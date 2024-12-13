# python scaling_benchmarks.py --base-config elie.yaml --debug
# python scaling_benchmarks.py --debug
import argparse
import math
import os

import pandas as pd
import yaml
from nanotron.logging import human_format
from tqdm import tqdm

ACCUMULATE_GRAD_IN_FP32 = True
NUM_KEY_VALUE_HEADS = 8


def estimate_num_params(layers, hidden_size, heads, intermediate_size, tie_word_embeddings, vocab, kv_heads=None):
    # params = 2*V*h + l(3*h*H + (2 + 2*q/kv_ratio)*h*h)
    # For GQA with 8 KV heads and 32 attention heads (4x ratio), it's: 2*V*h + l(3*h*H + (2 + 2/4)*h*h)
    vocab = vocab * hidden_size if tie_word_embeddings else 2 * vocab * hidden_size
    kv_ratio = kv_heads / heads if kv_heads is not None else 1
    qkv_params = (2 + 2 * kv_ratio) * hidden_size * hidden_size  # Account for GQA
    return vocab + layers * (3 * hidden_size * intermediate_size + qkv_params)


def create_config(
    dp: int,
    tp: int,
    pp: int,
    batch_accum: int,
    seq_len: int,
    micro_batch_size: int = 1,
    base_config_path: str = "examples/config_tiny_llama_bench.yaml",
    zero_stage: int = 0,
    num_layers: int = 24,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    intermediate_size=None,
    tp_mode: str = "REDUCE_SCATTER",
    vocab_size: int = 32768,
    profile: bool = False,
    benchmark_csv_path: str = "benchmark/results/bench_final.csv",
) -> dict:
    """Create a config with the specified parallelism settings."""
    # Load base config
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Modify parallelism settings
    config["parallelism"]["dp"] = dp
    config["parallelism"]["tp"] = tp
    config["parallelism"]["pp"] = pp

    # Modify batch and sequence settings
    config["tokens"]["batch_accumulation_per_replica"] = batch_accum
    config["tokens"]["sequence_length"] = seq_len
    config["model"]["model_config"]["max_position_embeddings"] = seq_len
    config["tokens"]["micro_batch_size"] = micro_batch_size

    # Modify model architecture settings
    config["model"]["model_config"]["num_hidden_layers"] = num_layers
    config["model"]["model_config"]["hidden_size"] = hidden_size
    config["model"]["model_config"]["num_attention_heads"] = num_attention_heads
    config["model"]["model_config"]["num_key_value_heads"] = (
        NUM_KEY_VALUE_HEADS if NUM_KEY_VALUE_HEADS is not None else num_attention_heads
    )
    config["model"]["model_config"]["intermediate_size"] = (
        intermediate_size if intermediate_size is not None else 4 * hidden_size
    )
    config["model"]["model_config"]["tie_word_embeddings"] = (
        True if intermediate_size < 10_000 else False
    )  # model < 4B

    # Set vocab_size
    config["model"]["model_config"]["vocab_size"] = vocab_size

    # Set zero stage
    config["optimizer"]["zero_stage"] = zero_stage

    # Set tp mode
    config["parallelism"]["tp_mode"] = tp_mode
    config["parallelism"]["tp_linear_async_communication"] = True if tp_mode == "REDUCE_SCATTER" else False

    num_params = estimate_num_params(
        num_layers,
        hidden_size,
        num_attention_heads,
        config["model"]["model_config"]["intermediate_size"],
        config["model"]["model_config"]["tie_word_embeddings"],
        vocab_size,
    )
    N = human_format(num_params)

    # Update run name to reflect configuration
    config["general"][
        "run"
    ] = f"{N}_dp{dp}_tp{tp}_pp{pp}_acc{batch_accum}_mbs{micro_batch_size}_seq{seq_len}_zero{zero_stage}_tpmode{tp_mode[:3]}_vocab{vocab_size//1000}k"

    # Update benchmark CSV path
    config["general"]["benchmark_csv_path"] = benchmark_csv_path

    if profile:
        config["profiler"] = {}
        config["profiler"]["profiler_export_path"] = "./tb_logs"
        config["tokens"]["train_steps"] = 10
        config["general"]["run"] += "_prof"

    return config


def generate_slurm_script(
    config: dict,
    dp: int,
    tp: int,
    pp: int,
    time: str = "00:02:00",
    partition: str = "hopper-prod",
    base_script_path: str = "run_multinode.sh",
) -> str:
    """Generate a SLURM script for the given configuration."""
    # Check if base script exists
    if not os.path.exists(base_script_path):
        raise FileNotFoundError(f"Base script file not found: {base_script_path}")

    # Load base script
    with open(base_script_path) as f:
        script = f.read()

    # Calculate required number of nodes
    gpus_per_node = 8
    total_gpus_needed = dp * tp * pp
    num_nodes = math.ceil(total_gpus_needed / gpus_per_node)

    # Replace SLURM parameters
    replacements = {
        "--nodes=2": f"--nodes={num_nodes}",
        "--time=00:02:00": f"--time={time}",
        "--partition=hopper-prod": f"--partition={partition}",
        "--job-name=smolm2-bench": f"--job-name=bench_{config['general']['run']}",
        "examples/config_tiny_llama.yaml": f"benchmark/configs/config_{config['general']['run']}.yaml",
    }

    for old, new in replacements.items():
        if old not in script:
            print(f"Warning: Could not find '{old}' in base script")
        script = script.replace(old, new)

    return script


def check_params(model_configs):
    for model_name, (num_layers, hidden_size, num_heads, intermediate_size) in model_configs.items():
        print(f"{model_name} model parameters:")
        tie = True if intermediate_size < 10_000 else False
        print(
            f"  Embedding params: {human_format(estimate_num_params(num_layers, hidden_size, num_heads, intermediate_size, tie, 131072, 8))}"
        )
        print()

    exit()


def save_experiment_configs(configs, output_path):
    """Save core experiment configurations for tracking"""
    records = []

    for config in configs:
        # Calculate total params
        tie_word_embeddings = True if config["model"]["model_config"]["intermediate_size"] < 10_000 else False
        estimate_num_params(
            config["model"]["model_config"]["num_hidden_layers"],
            config["model"]["model_config"]["hidden_size"],
            config["model"]["model_config"]["num_attention_heads"],
            config["model"]["model_config"]["intermediate_size"],
            tie_word_embeddings,
            config["model"]["model_config"]["vocab_size"],
            NUM_KEY_VALUE_HEADS,
        )
        record = {
            "name": config["general"]["run"],
            "nodes": config["parallelism"]["dp"] * config["parallelism"]["tp"] * config["parallelism"]["pp"] / 8,
            "seq_len": config["tokens"]["sequence_length"],
            "mbs": config["tokens"]["micro_batch_size"],
            "batch_accum": config["tokens"]["batch_accumulation_per_replica"],
            "gbs": config["tokens"]["sequence_length"]
            * config["tokens"]["micro_batch_size"]
            * config["tokens"]["batch_accumulation_per_replica"]
            * config["parallelism"]["dp"],
            "dp": config["parallelism"]["dp"],
            "pp": config["parallelism"]["pp"],
            "tp": config["parallelism"]["tp"],
            "tp_mode": f"TensorParallelLinearMode.{config['parallelism']['tp_mode']}",
            "hidden_size": config["model"]["model_config"]["hidden_size"],
            "num_layers": config["model"]["model_config"]["num_hidden_layers"],
            "num_heads": config["model"]["model_config"]["num_attention_heads"],
            "vocab_size": config["model"]["model_config"]["vocab_size"],
            "zero_stage": config["optimizer"]["zero_stage"],
        }
        records.append(record)

    # Save to CSV
    if os.path.exists(output_path):
        # Read existing data and append new records
        existing_df = pd.read_csv(output_path)
        df = pd.DataFrame(records)
        df = pd.concat([existing_df, df], ignore_index=True)
    else:
        df = pd.DataFrame(records)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(records)} experiment configurations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run scaling benchmarks with different parallelism configurations")
    parser.add_argument(
        "--configs-dir", type=str, default="benchmark/configs", help="Directory to store generated configs"
    )
    parser.add_argument(
        "--scripts-dir", type=str, default="benchmark/scripts", help="Directory to store generated SLURM scripts"
    )
    parser.add_argument("--partition", type=str, default="hopper-prod", help="SLURM partition to use")
    parser.add_argument("--time", type=str, default="00:10:00", help="Time limit for each job")
    parser.add_argument(
        "--base-config",
        type=str,
        default="examples/config_tiny_llama_bench.yaml",
        help="Base configuration file to use",
    )
    parser.add_argument("--base-script", type=str, default="run_multinode.sh", help="Base SLURM script to use")
    parser.add_argument(
        "--pending-csv",
        type=str,
        default="benchmark/results/pending_experiments_stress.csv",
        help="CSV file to store pending experiments",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=str,
        default="benchmark/results/bench_final_stress.csv",
        help="CSV file to store benchmark results",
    )
    parser.add_argument("--run", action="store_true", help="Automatically submit all generated SLURM scripts")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.base_config):
        raise FileNotFoundError(f"Base config file not found: {args.base_config}")
    if not os.path.exists(args.base_script):
        raise FileNotFoundError(f"Base script file not found: {args.base_script}")

    # Create directories if they don't exist
    for directory in [args.configs_dir, args.scripts_dir]:
        os.makedirs(directory, exist_ok=True)

    # Define model configurations
    model_configs = {
        # (layers, hidden_size, heads, intermediate_size)
        # "1B": (16, 2048, 32, 8192), # 1.2G
        # "3B": (28, 3072, 24, 8192), # 3.2G
        "8B": (32, 4096, 32, 14336),  # 8.0G
        # "70B": (80, 8192, 64, 28672), # 70G
        # "405B": (126, 16384, 128, 53248), # 406G
    }

    # Define configurations to test
    configurations = []

    # For each model size, test different GPU configurations
    for model_name, (num_layers, hidden_size, num_heads, intermediate_size) in model_configs.items():
        vocab_size = 32768
        zero_stage = 0
        tp_mode = "REDUCE_SCATTER"
        configs = [  # 64 nodes max
            # 2k, 8k, 32k
            # GBS: 1M, 4M
            # Format: (dp, tp, pp, batch_accum, seq_len, mbs, ...)
            # Using SP what's the biggest seqlen we can fit?
            # (1, 8, 1, 1, 2048, 1, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # (1, 8, 1, 1, 2048, 2, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # (1, 8, 1, 1, 2048, 8, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # (1, 8, 1, 1, 2048, 32, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # best run
            (
                2,
                8,
                1,
                1,
                2048,
                512,
                num_layers,
                hidden_size,
                num_heads,
                intermediate_size,
                vocab_size,
                zero_stage,
                tp_mode,
            ),
            # test zero
            # (3, 8, 1, 1, 2048, 64, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, 0, tp_mode),
            # (3, 8, 1, 1, 2048, 64, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, 1, tp_mode),
            # (24, 1, 1, 1, 2048, 8, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, 0, tp_mode),
            # (24, 1, 1, 1, 2048, 8, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, 1, tp_mode),
            # test tp mode
            # (1, 8, 1, 1, 2048, 64, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, "ALL_REDUCE"),
            # test pp
            # (1, 1, 8, 1, 2048, 64, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # (1, 8, 2, 1, 2048, 64, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # (1, 1, 8, 8, 2048, 8, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # (1, 2, 8, 8, 2048, 8, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # (1, 2, 64, 8, 2048, 8, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
            # (1, 2, 16, 8, 2048, 8, num_layers, hidden_size, num_heads, intermediate_size, vocab_size, zero_stage, tp_mode),
        ]
        configurations.extend(configs)

    # Duplicate configurations 100 times
    # configurations = configurations * 5000

    # Method 2: Parameter combinations
    PARALLEL_CONFIGS = [
        (dp, tp, pp)
        for dp in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        for tp in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        for pp in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ]  # Max 64 nodes
    # Sort PARALLEL_CONFIGS by total GPU count (dp*tp*pp) ascending
    PARALLEL_CONFIGS = sorted(PARALLEL_CONFIGS, key=lambda x: x[0] * x[1] * x[2])

    # for pp, tp, dp in PARALLEL_CONFIGS:
    #     for model_name, (num_layers, hidden_size, num_heads, intermediate_size) in model_configs.items():
    #         for seq_len in SEQUENCE_LENGTHS:
    #             for mbs in MBS:
    #                 for batch_accum in GRAD_ACCUM_STEPS:
    #                     for vocab_size in VOCAB_SIZES:
    #                         for zero_stage in ZERO_STAGES:
    #                             for tp_mode in TP_MODES:
    #                                 # Optional: Add conditions to filter out unwanted combinations
    #                                 total_gpus = dp * tp * pp
    #                                 if total_gpus < 8 or total_gpus/8 > 64: # max 64 nodes
    #                                     continue

    #                                 tokens_per_step = dp * mbs * batch_accum * seq_len
    #                                 if not tokens_per_step in [512*2048, 2048*2048]:
    #                                     continue

    #                                 # if dp=1 skip zero stage 1
    #                                 if dp == 1 and zero_stage == 1:
    #                                     continue

    #                                 # if tp=1 skip tp_mode=ALL_REDUCE
    #                                 if tp == 1 and tp_mode == "ALL_REDUCE":
    #                                     continue

    #                                 configurations.append((
    #                                     dp, tp, pp,
    #                                     batch_accum, seq_len, mbs,
    #                                     num_layers, hidden_size, num_heads, intermediate_size,
    #                                     vocab_size, zero_stage, tp_mode
    #                                 ))
    #                                 time += total_gpus * 1.5 / 8 / 64 # 1.5 minutes per config

    # print(len(configurations))
    # each config takes 1.5 minutes to run, print how many days
    # print(f"{time / 60 / 24:.2f} days ({time/60:.2f} hours)")
    # exit()

    if args.debug:
        print("Debug mode: only running 1 configuration")
        configurations = configurations[:1]

    # run first 100 configurations
    # configurations = configurations[:120+5000]

    # Validate configurations
    for (
        dp,
        tp,
        pp,
        batch_accum,
        seq_len,
        mbs,
        num_layers,
        hidden_size,
        num_heads,
        intermediate_size,
        vocab_size,
        zero_stage,
        tp_mode,
    ) in configurations:
        total_gpus = dp * tp * pp
        if total_gpus > 512:
            print(
                f"Warning: Configuration dp={dp}, tp={tp}, pp={pp} requires {total_gpus} GPUs, which might be too many"
            )

        # Calculate tokens per step to verify batch size
        # tokens_per_step = human_format(dp * mbs * batch_accum * seq_len)
        # print(f"Model {hidden_size}H_{num_layers}L: {total_gpus} GPUs, " f"{tokens_per_step} GBS")

    # Generate configs and scripts
    generated_scripts = []
    configs = []
    for (
        dp,
        tp,
        pp,
        batch_accum,
        seq_len,
        mbs,
        num_layers,
        hidden_size,
        num_heads,
        intermediate_size,
        vocab_size,
        zero_stage,
        tp_mode,
    ) in tqdm(configurations, desc="Generating configs and scripts"):
        try:
            # Create config
            config = create_config(
                dp=dp,
                tp=tp,
                pp=pp,
                batch_accum=batch_accum,
                seq_len=seq_len,
                micro_batch_size=mbs,
                base_config_path=args.base_config,
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
                zero_stage=zero_stage,
                tp_mode=tp_mode,
                profile=args.profile,
                benchmark_csv_path=args.benchmark_csv,
            )

            # Save config
            config_path = os.path.join(args.configs_dir, f"config_{config['general']['run']}.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            # Generate and save SLURM script
            script = generate_slurm_script(
                config, dp, tp, pp, time=args.time, partition=args.partition, base_script_path=args.base_script
            )

            script_path = os.path.join(args.scripts_dir, f"run_{config['general']['run']}.sh")
            with open(script_path, "w") as f:
                f.write(script)

            # Make script executable
            os.chmod(script_path, 0o755)

            generated_scripts.append(script_path)
            configs.append(config)

        except Exception as e:
            print(f"Error processing configuration (dp={dp}, tp={tp}, pp={pp}): {str(e)}")

    save_experiment_configs(configs, args.pending_csv)

    # Submit jobs if requested
    if args.run:
        import subprocess

        print("\nSubmitting jobs...")
        for script_path in tqdm(generated_scripts, desc="Submitting jobs"):
            try:
                result = subprocess.run(["sbatch", script_path], check=True, capture_output=True, text=True)
                print(f"Submitted {script_path}: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting {script_path}: {e.stderr}")
    else:
        print("\nTo run individual jobs:")
        for script_path in generated_scripts:
            print(f"sbatch {script_path}")


if __name__ == "__main__":
    main()
