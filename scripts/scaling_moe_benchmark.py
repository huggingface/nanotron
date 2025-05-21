import argparse
import os
import subprocess
from copy import deepcopy

import numpy as np
import pandas as pd
from nanotron.config import Config

SUBMIT_JOB_PATH = "/fsx/phuc/new_workspace/snippets/runner/submit_job.py"


def generate_scaling_configs(
    seq_len=4096,
    mbs=2,
    gpus_per_node=8,
    target_gbs_min=4_000_000,
    target_gbs_max=8_000_000,
    num_layers=[9, 10, 11],
    learning_rates=[0.00002, 0.00006, 0.0001, 0.0002],
):
    # Initialize lists to store configurations
    configs = []

    # Define node counts to explore
    node_counts = [1, 2, 4, 8, 16, 32, 64, 128]

    # Target the lower end of the range
    target_gbs = target_gbs_min

    for nodes in node_counts:
        for nl in num_layers:
            for lr in learning_rates:
                # Calculate data parallel size based on number of nodes
                world_size = gpus_per_node * nodes
                dp_replicas = world_size

                # Calculate accumulation steps needed to approach target GBS
                # GBS = seq_len * mbs * dp_size * accum
                ideal_accum = target_gbs / (seq_len * mbs * dp_replicas)

                # Round to nearest integer, minimum of 1
                # For smaller batch sizes, round up to ensure we meet minimum target
                accum = max(1, int(np.ceil(ideal_accum)))

                # Calculate actual GBS with this configuration
                actual_gbs = seq_len * mbs * dp_replicas * accum

                # Calculate batch size per replica
                bs_per_replica = seq_len * mbs

                # Calculate ep and edp values
                ep = 8
                edp = world_size // ep

                # Store configuration
                configs.append(
                    {
                        "Nodes": nodes,
                        "GPUs": world_size,
                        "MicroBatchSize": mbs,
                        "SequenceLength": seq_len,
                        "AccumSteps": accum,
                        "BatchSizePerReplica": bs_per_replica,
                        "GlobalBatchSize": actual_gbs,
                        "GlobalBatchSizeMillions": actual_gbs / 1_000_000,
                        "ep": ep,
                        "edp": edp,
                        "NumLayers": nl,
                        "LearningRate": lr,
                    }
                )

    # Create DataFrame
    df = pd.DataFrame(configs)

    # Format numbers for better readability
    df["GlobalBatchSizeMillions"] = df["GlobalBatchSizeMillions"].round(2)

    return df


def create_experiment_names(df):
    # Define mapping for node counts to alphabets
    node_to_alphabet = {1: "a", 2: "b", 4: "c", 8: "d", 16: "e", 32: "f", 64: "g", 128: "h"}

    # Create experiment names with alphabet code based on node count
    df["ExperimentName"] = df.apply(
        lambda row: f"exp19{node_to_alphabet[row['Nodes']]}a1_like_exp18aa1_and_{int(row['Nodes'])}_node_OLMoE-1B-7B_te_and_seq_len_{int(row['SequenceLength'])}_and_batch_accum{int(row['AccumSteps'])}_and_mbs{int(row['MicroBatchSize'])}_and_gbs{int(row['AccumSteps']*row['MicroBatchSize'])}_with_{int(row['GlobalBatchSizeMillions'])}m_and_elie_training_config_and_fineweb_numlayer{int(row['NumLayers'])}_and_seed_312_but_dp{int(row['GPUs'])}_tp1_ep{int(row['ep'])}_edp{int(row['edp'])}_and_lr{row['LearningRate']:.6f}_and_groupedgemm_and_allgather",
        axis=1,
    )

    return df


def create_scaled_configs(
    base_config: Config,
    scaling_df: pd.DataFrame,
    output_base_dir: str,
    benchmark_csv_path: str,
    brrr_repo_path: str,
    uv_env_path: str,
    script_path: str,
    reservation_name: str,
    launch_config: bool = False,
):
    """Create scaled config files and optionally launch jobs"""
    os.makedirs(output_base_dir, exist_ok=True)

    for _, row in scaling_df.iterrows():
        print(f"Generating config for {row['ExperimentName']}")
        new_config = deepcopy(base_config)

        # Config generation remains the same
        new_config.general.benchmark_csv_path = benchmark_csv_path
        new_config.general.run = row["ExperimentName"]
        new_config.model.model_config.num_hidden_layers = row["NumLayers"]
        new_config.parallelism.dp = row["GPUs"]
        new_config.parallelism.expert_parallel_size = row["ep"]
        new_config.parallelism.expert_data_parallel_size = row["edp"]
        new_config.tokens.sequence_length = row["SequenceLength"]
        new_config.tokens.micro_batch_size = row["MicroBatchSize"]
        new_config.tokens.batch_accumulation_per_replica = row["AccumSteps"]
        new_config.optimizer.learning_rate_scheduler.learning_rate = row["LearningRate"]

        config_path = os.path.join(output_base_dir, f"{row['ExperimentName']}.yaml")
        new_config.save_as_yaml(config_path)

        # Build launch command
        launch_command = [
            "python3",
            SUBMIT_JOB_PATH,
            "--config",
            config_path,
            "--nproc_per_node",
            "8",
            "--brrr_repo_path",
            brrr_repo_path,
            "--uv_env_path",
            uv_env_path,
            "--nodes",
            str(row["Nodes"]),
            "--script_path",
            script_path,
            "--is_brrr_config",
            "false",
            "--reservation_name",
            reservation_name,
        ]

        if launch_config:
            print("Launching:", " ".join(launch_command))
            subprocess.run(launch_command, check=True)
        else:
            print("Would launch:", " ".join(launch_command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scaled config files and launch jobs")
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument(
        "--output-base-dir", type=str, required=True, help="Shared directory for all configs and benchmark CSV"
    )
    parser.add_argument(
        "--benchmark-csv-name",
        type=str,
        default="benchmark_results.csv",
        help="Filename for EMPTY benchmark CSV in output-base-dir",
    )

    # Add new argument to parser
    parser.add_argument("--launch-config", action="store_true", help="Actually launch the jobs (dry run by default)")

    # Add new arguments for job submission
    parser.add_argument(
        "--brrr_repo_path", type=str, required=False, default="", help="[Optional] Path to BRRR repository"
    )
    parser.add_argument(
        "--uv_env_path", type=str, required=False, default="", help="[Optional] Path to UV environment"
    )
    parser.add_argument(
        "--script_path", type=str, default="run_train.py", help="Training script path (default: run_train.py)"
    )
    parser.add_argument(
        "--reservation_name", type=str, required=False, default="", help="[Optional] Cluster reservation name"
    )

    args = parser.parse_args()

    base_config = Config.load_from_yaml(args.base_config)
    os.makedirs(args.output_base_dir, exist_ok=True)

    # Path definitions
    benchmark_csv_path = os.path.join(args.output_base_dir, args.benchmark_csv_name)
    scaling_configs_path = os.path.join(args.output_base_dir, "scaling_configs.csv")

    # Generate configurations and names
    scaling_configs = generate_scaling_configs(seq_len=4096, mbs=2, num_layers=[10], learning_rates=[0.00002])
    scaling_config_names = create_experiment_names(scaling_configs.copy())

    # Save reference configurations with names
    scaling_config_names.to_csv(scaling_configs_path, index=False)

    # Create empty benchmark file
    open(benchmark_csv_path, "a").close()  # Creates empty file if it doesn't exist

    create_scaled_configs(
        base_config=base_config,
        scaling_df=scaling_config_names,
        output_base_dir=args.output_base_dir,
        benchmark_csv_path=benchmark_csv_path,
        brrr_repo_path=args.brrr_repo_path,
        uv_env_path=args.uv_env_path,
        script_path=args.script_path,
        reservation_name=args.reservation_name,
        launch_config=args.launch_config,
    )
