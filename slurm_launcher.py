#!/usr/bin/env python3
"""
Nanotron Slurm Launcher

This script simplifies launching multi-node Nanotron training jobs on Slurm clusters.
It handles configuration generation, resource allocation, and job submission.

Usage:
    python slurm_launcher.py --run_name my_experiment --nodes 4 [other options]

The script will:
1. Generate a Nanotron config based on your parameters
2. Create a Slurm job script with appropriate settings
3. Submit the job to the Slurm scheduler
4. Save configurations for reproducibility
"""

import argparse
import logging
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from typing import Optional

# Nanotron imports
from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LlamaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    ProfilerArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

logger = logging.getLogger(__name__)

# =============================================
# CONFIGURATION SECTION - MODIFY AS NEEDED
# =============================================

# Default paths - override with command line arguments if needed
DEFAULT_CONFIGS_PATH = "logs/configs"
DEFAULT_SLURM_LOGS_PATH = "logs/slurm_logs"
DEFAULT_SLURM_SCRIPTS_DIR = "logs/slurm_scripts"
DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_RUN_TRAIN_SCRIPT = "run_train.py"

# Default model sizes - predefined configurations for common model sizes
MODEL_CONFIGS = {
    "tiny": {
        "hidden_size": 16,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 256,
    },
    "small": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "vocab_size": 32000,
    },
    "base": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 32000,
    },
    "large": {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
    },
}


def generate_model_config(
    model_size: str = "tiny",
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    vocab_size: Optional[int] = None,
    max_position_embeddings: int = 4096,
) -> LlamaConfig:
    """
    Generate a model configuration based on predefined sizes or custom parameters.

    Args:
        model_size: Predefined size ('tiny', 'small', 'base', 'large')
        hidden_size: Custom hidden size (overrides model_size)
        intermediate_size: Custom intermediate size (overrides model_size)
        num_hidden_layers: Custom number of layers (overrides model_size)
        num_attention_heads: Custom number of attention heads (overrides model_size)
        num_key_value_heads: Custom number of KV heads (overrides model_size)
        vocab_size: Custom vocabulary size (overrides model_size)
        max_position_embeddings: Maximum sequence length

    Returns:
        LlamaConfig object with the specified parameters
    """
    # Start with the base configuration for the requested size
    config_params = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["tiny"]).copy()

    # Override with any explicitly specified parameters
    if hidden_size is not None:
        config_params["hidden_size"] = hidden_size
    if intermediate_size is not None:
        config_params["intermediate_size"] = intermediate_size
    if num_hidden_layers is not None:
        config_params["num_hidden_layers"] = num_hidden_layers
    if num_attention_heads is not None:
        config_params["num_attention_heads"] = num_attention_heads
    if num_key_value_heads is not None:
        config_params["num_key_value_heads"] = num_key_value_heads
    if vocab_size is not None:
        config_params["vocab_size"] = vocab_size

    # Create the model configuration
    model_config = LlamaConfig(
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        initializer_range=0.02,
        max_position_embeddings=max_position_embeddings,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        tie_word_embeddings=False,
        use_cache=True,
        **config_params,
    )

    return model_config


def launch_slurm_job(launch_file_contents: str, *args) -> str:
    """
    Save a sbatch script to a temporary file and submit it to Slurm.

    Args:
        launch_file_contents: Contents of the sbatch script
        *args: Additional arguments to pass to the sbatch command

    Returns:
        Job ID of the submitted Slurm job
    """
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(launch_file_contents)
        f.flush()
        return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]


def create_nanotron_config(args) -> Config:
    """
    Create a Nanotron configuration object based on the provided arguments.

    Args:
        args: Command line arguments

    Returns:
        Nanotron Config object
    """
    # Generate model configuration
    model_config = generate_model_config(
        model_size=args.model_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
    )

    # Calculate number of parameters for logging
    num_params = human_format(
        model_config.vocab_size * model_config.hidden_size * 2
        + model_config.num_hidden_layers
        * (
            3 * model_config.hidden_size * model_config.intermediate_size
            + 4 * model_config.hidden_size * model_config.hidden_size
        )
    ).replace(".", "p")

    print(f"Model has {num_params} parameters")

    # Generate a unique run ID
    job_id = str(uuid.uuid4())[:8]
    run_name = args.run_name.replace(" ", "_")
    run_id = f"{args.project}-{num_params}-{run_name}-seed-{args.seed}-{job_id}"

    # Use user-provided parallelism directly
    parallelism = ParallelismArgs(
        dp=args.dp,
        pp=args.pp,
        tp=args.tp,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )

    # Define tokens configuration
    tokens = TokensArgs(
        sequence_length=args.max_seq_len,
        train_steps=args.train_steps,
        micro_batch_size=args.micro_batch_size,
        batch_accumulation_per_replica=args.grad_accum_steps,
    )

    # Calculate global batch size for logging
    gbs = parallelism.dp * tokens.batch_accumulation_per_replica * tokens.micro_batch_size * tokens.sequence_length
    total_tokens = gbs * args.train_steps
    print(f"GBS: {(gbs)/1e6:.2f}M, total training tokens: {(total_tokens)/1e9:.2f}B")

    # Verify parallelism setting relative to available resources
    total_gpus = args.nodes * args.gpus_per_node
    required_gpus = parallelism.dp * parallelism.pp * parallelism.tp

    if required_gpus > total_gpus:
        logger.warning(
            f"Requested parallelism (DP={parallelism.dp}, PP={parallelism.pp}, TP={parallelism.tp}) "
            f"requires {required_gpus} GPUs, but only {total_gpus} are available. "
            "This may cause the job to fail."
        )
    elif required_gpus < total_gpus:
        logger.warning(
            f"Requested parallelism (DP={parallelism.dp}, PP={parallelism.pp}, TP={parallelism.tp}) "
            f"uses only {required_gpus} of {total_gpus} available GPUs. "
            "Consider adjusting parallelism for better resource utilization."
        )

    # Configure learning rate schedule
    lr_scheduler = LRSchedulerArgs(
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.warmup_steps,
        lr_warmup_style="linear",
        lr_decay_style="cosine",
        min_decay_lr=args.min_lr,
    )

    # Configure optimizer
    optimizer = OptimizerArgs(
        zero_stage=0,
        weight_decay=args.weight_decay,
        clip_grad=args.grad_clip,
        accumulate_grad_in_fp32=True,
        learning_rate_scheduler=lr_scheduler,
        optimizer_factory=AdamWOptimizerArgs(
            adam_eps=1e-08,
            adam_beta1=0.9,
            adam_beta2=0.95,
            torch_adam_is_fused=True,
        ),
    )

    # Configure datasets
    data_stages = [
        DatasetStageArgs(
            name="Stable Training Stage",
            start_training_step=1,
            data=DataArgs(
                dataset=PretrainDatasetsArgs(
                    hf_dataset_or_datasets=args.dataset,
                    text_column_name=args.text_column,
                ),
                seed=args.seed,
            ),
        ),
    ]

    # Configure checkpointing
    os.makedirs(args.checkpoints_path, exist_ok=True)
    checkpoints = CheckpointsArgs(
        checkpoints_path=os.path.join(args.checkpoints_path, run_id),
        checkpoint_interval=args.save_interval,
        save_initial_state=args.save_initial_state,
    )

    # Create the final config
    config = Config(
        general=GeneralArgs(project=args.project, run=run_id, seed=args.seed),
        checkpoints=checkpoints,
        parallelism=parallelism,
        model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
        tokenizer=TokenizerArgs(args.tokenizer),
        optimizer=optimizer,
        logging=LoggingArgs(log_level="info", iteration_step_info_interval=1),
        tokens=tokens,
        data_stages=data_stages,
        profiler=ProfilerArgs(profiler_export_path=args.profiler_export_path)
        if args.profiler_export_path is not None
        else None,
    )

    return config, run_id


def create_slurm_script(
    config_path: str,
    run_id: str,
    args,
    run_train_script: str = "run_train.py",
) -> str:
    """
    Create a Slurm job submission script.

    Args:
        config_path: Path to the Nanotron config YAML file
        run_id: Unique identifier for this run
        args: Command line arguments

    Returns:
        Contents of the Slurm script as a string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = os.path.join(args.slurm_logs_path, args.run_name.replace(" ", "_"))
    os.makedirs(logs_path, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name={args.run_name}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=60         # CPU cores per task
#SBATCH --exclusive               # Exclusive use of nodes
#SBATCH --gpus-per-node={args.gpus_per_node}
#SBATCH --partition={args.partition}
#SBATCH --output={logs_path}/{timestamp}-%x-%j.out
#SBATCH --time={args.time_limit}
#SBATCH --qos={args.qos}
#SBATCH --wait-all-nodes=1        # fail if any node is not ready
"""

    if args.email:
        script += f"""#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user={args.email}
"""

    script += f"""
set -x -e

echo "START TIME: $(date)"
secs_to_human() {{
    echo "$(( ${{1}} / 3600 )):$(( (${{1}} / 60) % 60 )):$(( ${{1}} % 60 ))"
}}
start=$(date +%s)
echo "$(date -d @${{start}} "+%Y-%m-%d %H:%M:%S"): ${{SLURM_JOB_NAME}} start id=${{SLURM_JOB_ID}}\\n"

# SLURM setup
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR={args.tmp_dir}
export CUDA_DEVICE_MAX_CONNECTIONS=1
{args.extra_env}

echo "Running on $COUNT_NODE nodes: $HOSTNAMES"

# Calculate total number of processes
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE={args.gpus_per_node}
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

# Set some environment variables for better distributed training
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN # INFO, WARN
# export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Nanotron specific
# export NANOTRON_BENCHMARK=1
{"" if args.wandb_disabled else "# "}export WANDB_MODE=disabled


CMD="{run_train_script} --config-file {config_path}"

LAUNCHER="torchrun \\
    --nproc_per_node {args.gpus_per_node} \\
    --nnodes $COUNT_NODE \\
    --rdzv_backend c10d \\
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \\
    --max_restarts 0 \\
    --tee 3 \\
    "

{args.pre_launch_commands}

srun -u bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"
end=$(date +%s)
elapsed=$((end - start))
echo "Total training time: $(secs_to_human $elapsed)"
"""

    return script


def parse_args():
    """Parse command line arguments for the Slurm launcher."""
    parser = argparse.ArgumentParser(description="Nanotron Slurm Launcher")

    # Required arguments
    parser.add_argument("--run_name", type=str, required=True, help="Name for this experiment run")

    # Slurm job configuration
    slurm_group = parser.add_argument_group("Slurm Configuration")
    slurm_group.add_argument("--nodes", type=int, default=2, help="Number of nodes to use")
    slurm_group.add_argument("--gpus_per_node", type=int, default=8, help="Number of GPUs per node")
    slurm_group.add_argument("--partition", type=str, default="hopper-prod", help="Slurm partition to use")
    slurm_group.add_argument("--qos", type=str, default="normal", help="Slurm QOS to use")
    slurm_group.add_argument("--time_limit", type=str, default="1:00:00", help="Time limit for the job (HH:MM:SS)")
    slurm_group.add_argument("--email", type=str, default=None, help="Email for job notifications")
    slurm_group.add_argument("--tmp_dir", type=str, default="/tmp", help="Temporary directory on compute nodes")
    slurm_group.add_argument("--pre_launch_commands", type=str, default="", help="Commands to run before job launch")
    slurm_group.add_argument("--extra_env", type=str, default="", help="Additional environment variables")

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base", "large"],
        help="Predefined model size",
    )
    model_group.add_argument("--hidden_size", type=int, default=None, help="Hidden size (overrides model_size)")
    model_group.add_argument(
        "--intermediate_size", type=int, default=None, help="Intermediate size (overrides model_size)"
    )
    model_group.add_argument("--num_layers", type=int, default=None, help="Number of layers (overrides model_size)")
    model_group.add_argument(
        "--num_heads", type=int, default=None, help="Number of attention heads (overrides model_size)"
    )
    model_group.add_argument(
        "--num_kv_heads", type=int, default=None, help="Number of KV heads (overrides model_size)"
    )
    model_group.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size (overrides model_size)")
    model_group.add_argument("--max_seq_len", type=int, default=4096, help="Maximum sequence length")

    # Training configuration
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    training_group.add_argument("--train_steps", type=int, default=10000, help="Number of training steps")
    training_group.add_argument("--micro_batch_size", type=int, default=2, help="Micro batch size")
    training_group.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    training_group.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    training_group.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate for decay")
    training_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    training_group.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    training_group.add_argument("--warmup_steps", type=int, default=1000, help="Learning rate warmup steps")

    # Parallelism strategy
    parallel_group = parser.add_argument_group("Parallelism Configuration")
    parallel_group.add_argument("--dp", type=int, default=8, help="Data parallelism (DP) degree")
    parallel_group.add_argument("--pp", type=int, default=1, help="Pipeline parallelism (PP) degree")
    parallel_group.add_argument("--tp", type=int, default=2, help="Tensor parallelism (TP) degree")

    # Dataset configuration
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset", type=str, default="stas/openwebtext-10k", help="Hugging Face dataset name or path"
    )
    data_group.add_argument("--text_column", type=str, default="text", help="Column name for text in the dataset")
    data_group.add_argument(
        "--tokenizer", type=str, default="robot-test/dummy-tokenizer-wordlevel", help="Tokenizer name or path"
    )

    # File paths
    paths_group = parser.add_argument_group("File Paths")
    paths_group.add_argument("--project", type=str, default="nanotron", help="Project name for logging")
    paths_group.add_argument(
        "--configs_path", type=str, default=DEFAULT_CONFIGS_PATH, help="Directory to save configuration files"
    )
    paths_group.add_argument(
        "--slurm_logs_path", type=str, default=DEFAULT_SLURM_LOGS_PATH, help="Directory for Slurm output logs"
    )
    paths_group.add_argument(
        "--checkpoints_path",
        type=str,
        default=DEFAULT_CHECKPOINTS_PATH,
        help="Base directory for saving model checkpoints",
    )
    slurm_group.add_argument(
        "--run_train_script",
        type=str,
        default=DEFAULT_RUN_TRAIN_SCRIPT,
        help="Path to the training script (default: run_train.py)",
    )
    slurm_group.add_argument(
        "--slurm_scripts_dir",
        type=str,
        default=DEFAULT_SLURM_SCRIPTS_DIR,
        help="Directory to save generated Slurm scripts (set to None to disable)",
    )
    paths_group.add_argument(
        "--save_interval", type=int, default=1000, help="Interval for saving checkpoints (in steps)"
    )
    paths_group.add_argument("--save_initial_state", action="store_true", help="Save initial state")

    # Logging configuration
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument("--wandb_disabled", action="store_true", help="Disable logging to Weights & Biases")
    logging_group.add_argument(
        "--profiler_export_path",
        type=str,
        default=None,
        help="Path to export the profiler tensorboard data. Use `tensorboard --logdir <path>` to view.",
    )

    # Execution control
    parser.add_argument("--dry_run", action="store_true", help="Generate configs but don't submit job")
    parser.add_argument("--show_logs", action="store_true", help="Show output of the job as it runs")
    return parser.parse_args()


def tail_output_file(output_file: str):
    """Tail the output file when available."""
    while not os.path.exists(output_file):
        time.sleep(1)
    with open(output_file, "r"):
        subprocess.run(["tail", "-f", output_file])


def main():
    """Main entry point for the Slurm launcher."""
    args = parse_args()

    # Create directories if they don't exist
    os.makedirs(args.configs_path, exist_ok=True)
    os.makedirs(args.slurm_logs_path, exist_ok=True)

    # Create Nanotron config
    config, run_id = create_nanotron_config(args)

    # Save config to YAML file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name.replace(" ", "_")
    config_dir = os.path.join(args.configs_path, run_name)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{timestamp}-{run_name}.yaml")
    config.save_as_yaml(config_path)
    print(f"Config saved to {config_path}")

    # Create Slurm script
    slurm_script = create_slurm_script(config_path, run_id, args, args.run_train_script)

    # Save Slurm script if requested
    if args.slurm_scripts_dir is not None:
        os.makedirs(args.slurm_scripts_dir, exist_ok=True)
        slurm_script_path = os.path.join(args.slurm_scripts_dir, f"{timestamp}-{run_name}.sh")
        with open(slurm_script_path, "w") as f:
            f.write(slurm_script)
        print(f"Slurm script saved to {slurm_script_path}")

    # Either submit the job or just print the script (dry run)
    if args.dry_run:
        print("DRY RUN - Job script:")
        print(slurm_script)
        print(f"Would submit job with config from {config_path}")
    else:
        job_id = launch_slurm_job(slurm_script)
        print(f"Slurm job submitted with JOBID: {job_id}")
        print(
            f"Logs will be available at: {os.path.join(args.slurm_logs_path, run_name, f'{timestamp}-{run_name}-{job_id}.out')}"
        )

        # Tail output file when available
        if args.show_logs:
            tail_output_file(os.path.join(args.slurm_logs_path, run_name, f"{timestamp}-{run_name}-{job_id}.out"))

    return 0


if __name__ == "__main__":
    main()
