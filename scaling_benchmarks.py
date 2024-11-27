#!/usr/bin/env python3
import argparse
import math
import os

import yaml


def create_config(
    dp: int,
    tp: int,
    pp: int,
    batch_accum: int,
    seq_len: int,
    micro_batch_size: int = 1,
    base_config_path: str = "examples/config_tiny_llama.yaml",
    zero_stage: int = 0,
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

    # modify zero stage
    config["optimizer"]["zero_stage"] = zero_stage

    # Update run name to reflect configuration
    config["general"][
        "run"
    ] = f"dp{dp}_tp{tp}_pp{pp}_acc{batch_accum}_mbs{micro_batch_size}_seq{seq_len}_zero{zero_stage}"

    # Update benchmark CSV path
    config["general"]["benchmark_csv_path"] = "bench.csv"

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
    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.base_config):
        raise FileNotFoundError(f"Base config file not found: {args.base_config}")
    if not os.path.exists(args.base_script):
        raise FileNotFoundError(f"Base script file not found: {args.base_script}")

    # Create directories if they don't exist
    for directory in [args.configs_dir, args.scripts_dir]:
        os.makedirs(directory, exist_ok=True)

    # Define configurations to test
    configurations = [
        # (dp, tp, pp, batch_accum, seq_len, mbs)
        # (1, 8, 1, 1, 2048, 1),    # Base configuration
        # (2, 4, 1, 1, 2048, 1),
        # (8, 1, 1, 1, 2048, 1),
        # (16, 1, 1, 1, 2048, 1),
        *[(2**i, 1, 1, 1, 2048, 1) for i in range(0, 8)],
        *[(2**i, 8, 1, 1, 2048, 1) for i in range(0, 7)],
        *[(2**i, 8, 1, 1, 2048, 8) for i in range(0, 7)],
        # 64k seq len
        *[(2**i, 8, 1, 1, 65536, 1) for i in range(0, 7)],  # 64 nodes max
    ]

    # Validate configurations
    for dp, tp, pp, batch_accum, seq_len, mbs in configurations:
        total_gpus = dp * tp * pp
        if total_gpus > 64:  # Assuming maximum of 8 nodes with 8 GPUs each
            print(
                f"Warning: Configuration dp={dp}, tp={tp}, pp={pp} requires {total_gpus} GPUs, which might be too many"
            )

    # Generate configs and scripts
    generated_scripts = []  # Keep track of generated script paths
    for dp, tp, pp, batch_accum, seq_len, mbs in configurations:
        try:
            # Create config
            config = create_config(dp, tp, pp, batch_accum, seq_len, mbs, base_config_path=args.base_config)

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

            print(f"Successfully generated config and script for {config['general']['run']}")
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
