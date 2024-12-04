# python scaling_benchmarks.py --base-config elie.yaml --debug
# python scaling_benchmarks.py --debug
import argparse
import math
import os

import yaml
from nanotron.logging import human_format

VOCAB_SIZE = 32768
NUM_KEY_VALUE_HEADS = None
TIE_WORD_EMBEDDINGS = True
ZERO_STAGE = 0
# TP_MODE = "REDUCE_SCATTER" # "REDUCE_SCATTER" "ALL_REDUCE"
TP_MODE = "ALL_REDUCE"  # "REDUCE_SCATTER" "ALL_REDUCE"
PROFILE = True


def estimate_num_params(layers, hidden_size, heads, intermediate_size, tie_word_embeddings):
    # params = 2*V*h + l(3*h*H + 4*h*h) = (2)Vh + 16lh^2
    vocab = VOCAB_SIZE * hidden_size if tie_word_embeddings else 2 * VOCAB_SIZE * hidden_size
    return vocab + layers * (3 * hidden_size * intermediate_size + 4 * hidden_size * hidden_size)


def create_config(
    dp: int,
    tp: int,
    pp: int,
    batch_accum: int,
    seq_len: int,
    micro_batch_size: int = 1,
    base_config_path: str = "examples/config_tiny_llama_bench.yaml",
    zero_stage: int = ZERO_STAGE,
    num_layers: int = 24,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    intermediate_size=None,
    tp_mode: str = TP_MODE,
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
    config["model"]["model_config"]["tie_word_embeddings"] = TIE_WORD_EMBEDDINGS

    # Set vocab_size to 32k to reduce memory usage
    config["model"]["model_config"]["vocab_size"] = VOCAB_SIZE

    # modify zero stage
    config["optimizer"]["zero_stage"] = zero_stage

    # modify tp mode
    config["parallelism"]["tp_mode"] = tp_mode
    config["parallelism"]["tp_linear_async_communication"] = True if tp_mode == "REDUCE_SCATTER" else False

    N = human_format(
        estimate_num_params(
            num_layers,
            hidden_size,
            num_attention_heads,
            config["model"]["model_config"]["intermediate_size"],
            config["model"]["model_config"]["tie_word_embeddings"],
        )
    )

    # Update run name to reflect configuration
    config["general"][
        "run"
    ] = f"{N}_dp{dp}_tp{tp}_pp{pp}_acc{batch_accum}_mbs{micro_batch_size}_seq{seq_len}_zero{zero_stage}_tpmode{tp_mode[:3]}_l{num_layers}_h{hidden_size}_heads{num_attention_heads}"

    # Update benchmark CSV path
    config["general"]["benchmark_csv_path"] = "bench_tp.csv"

    if PROFILE:
        config["profiler"] = {}
        config["profiler"]["profiler_export_path"] = "./tb_logs"
        config["tokens"]["train_steps"] = 10

    return config


def generate_slurm_script(
    config: dict,
    dp: int,
    tp: int,
    pp: int,
    time: str = "00:15:00",
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
        "--time=00:15:00": f"--time={time}",
        "--partition=hopper-prod": f"--partition={partition}",
        "--job-name=smolm2-bench": f"--job-name=bench_{config['general']['run']}",
        "examples/config_tiny_llama.yaml": f"benchmark/configs/config_{config['general']['run']}.yaml",
    }

    for old, new in replacements.items():
        if old not in script:
            print(f"Warning: Could not find '{old}' in base script")
        script = script.replace(old, new)

    return script


def main():
    parser = argparse.ArgumentParser(description="Run scaling benchmarks with different parallelism configurations")
    parser.add_argument(
        "--configs-dir", type=str, default="benchmark/configs", help="Directory to store generated configs"
    )
    parser.add_argument(
        "--scripts-dir", type=str, default="benchmark/scripts", help="Directory to store generated SLURM scripts"
    )
    parser.add_argument("--partition", type=str, default="hopper-prod", help="SLURM partition to use")
    parser.add_argument("--time", type=str, default="00:15:00", help="Time limit for each job")
    parser.add_argument(
        "--base-config", type=str, default="examples/config_tiny_llama.yaml", help="Base configuration file to use"
    )
    parser.add_argument("--base-script", type=str, default="run_multinode.sh", help="Base SLURM script to use")
    parser.add_argument("--run", action="store_true", help="Automatically submit all generated SLURM scripts")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
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
        # params = 2*V*h + l(3*h*H + 4*h*h) = (2)Vh + 16lh^2
        # (layers, hidden_size, heads, intermediate_size)
        # "138M": (12, 768, 12, 3072),
        # "200M": (12, 1024, 16, 4096),
        # "500M": (12, 1536, 16, 6144),
        # "1000M": (15, 2048, 16, 8192),
        # "1700M": (24, 2048, 16, 8192),  # (layers, hidden_size, heads, intermediate_size)
        # "4300M": (28, 3072, 20, 12288),
        # "8700M": (32, 4096, 32, 16384),
        # "11B": (42, 4096, 32, 16384),
        "3500M": (28, 3072, 24, 8192)
    }

    # Define configurations to test
    configurations = []

    # For each model size, test different GPU configurations
    for model_name, (num_layers, hidden_size, num_heads, intermediate_size) in model_configs.items():
        # Test each model with different GPU counts while maintaining 4M tokens/step
        model_configs = [
            # Format: (dp, tp, pp, batch_accum, seq_len, mbs, ...)
            # (1, 1, 1, 8, 2048, 1, num_layers, hidden_size, num_heads, intermediate_size),
            # (1, 2, 1, 1, 4096, 2, num_layers, hidden_size, num_heads, intermediate_size),
            # (1, 4, 1, 1, 4096, 2, num_layers, hidden_size, num_heads, intermediate_size),
            # (1, 8, 1, 1, 4096, 2, num_layers, hidden_size, num_heads, intermediate_size),
            # find best tput on 16 nodes with 4GBS
            (1, 8, 1, 1, 4096, 8, num_layers, hidden_size, num_heads, intermediate_size),  # test max MBS
            # (8, 1, 1, 1, 4096, 1, num_layers, hidden_size, num_heads, intermediate_size), # test max MBS
            # (1, 8, 1, 1, 4096, 64, num_layers, hidden_size, num_heads, intermediate_size), # test max MBS
            # (16, 8, 1, 1, 4096, 16, num_layers, hidden_size, num_heads, intermediate_size), # ideal run i guess
            # (32, 4, 1, 1, 4096, 8, num_layers, hidden_size, num_heads, intermediate_size), # TP=4
            # (64, 2, 1, 1, 4096, 4, num_layers, hidden_size, num_heads, intermediate_size), # TP=2
            # (128, 1, 1, 1, 4096, 2, num_layers, hidden_size, num_heads, intermediate_size), # TP=1
            # find best tput on 8 nodes with 1GBS
            # (8, 8, 1, 1, 4096, 32, num_layers, hidden_size, num_heads, intermediate_size),
            # (8, 8, 1, 2, 4096, 16, num_layers, hidden_size, num_heads, intermediate_size),
            # (16, 4, 1, 2, 4096, 8, num_layers, hidden_size, num_heads, intermediate_size),
            # (32, 2, 1, 2, 4096, 4, num_layers, hidden_size, num_heads, intermediate_size),
            # (64, 1, 1, 2, 4096, 2, num_layers, hidden_size, num_heads, intermediate_size),
            # same for 4 nodes
            # (4, 8, 1, 1, 4096, 16, num_layers, hidden_size, num_heads, intermediate_size),
            # (8, 4, 1, 1, 4096, 8, num_layers, hidden_size, num_heads, intermediate_size),
            # (16, 2, 1, 1, 4096, 4, num_layers, hidden_size, num_heads, intermediate_size),
            # (32, 1, 1, 1, 4096, 2, num_layers, hidden_size, num_heads, intermediate_size),
        ]
        configurations.extend(model_configs)

    if args.debug:
        print("Debug mode: only running 1 configuration")
        configurations = configurations[:1]

    # Validate configurations
    for dp, tp, pp, batch_accum, seq_len, mbs, num_layers, hidden_size, num_heads, intermediate_size in configurations:
        total_gpus = dp * tp * pp
        if total_gpus > 512:
            print(
                f"Warning: Configuration dp={dp}, tp={tp}, pp={pp} requires {total_gpus} GPUs, which might be too many"
            )

        # Calculate tokens per step to verify batch size
        tokens_per_step = dp * tp * pp * mbs * batch_accum * seq_len
        print(f"Model {hidden_size}H_{num_layers}L: {total_gpus} GPUs, " f"{tokens_per_step:,} GBS")

    # Generate configs and scripts
    generated_scripts = []
    for dp, tp, pp, batch_accum, seq_len, mbs, num_layers, hidden_size, num_heads, intermediate_size in configurations:
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

            print(f"Successfully generated config and script for {config_path}")
            generated_scripts.append(script_path)

        except Exception as e:
            print(f"Error processing configuration (dp={dp}, tp={tp}, pp={pp}): {str(e)}")

    # Submit jobs if requested
    if args.run:
        import subprocess

        print("\nSubmitting jobs...")
        for script_path in generated_scripts:
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
