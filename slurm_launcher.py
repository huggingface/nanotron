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
    NanosetDatasetsArgs,  # noqa
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,  # noqa
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
MODEL_SIZES = {
    # (layers, hidden, heads, kv_heads, ffn_size)
    "160m": (12, 768, 12, 12, 3072),  # ~160M params
    "410m": (24, 1024, 16, 16, 4096),  # ~410M params
    # Small to medium models
    "1b": (16, 2048, 16, 16, 5632),  # ~1B params
    "3b": (28, 3072, 32, 32, 8192),  # ~3B params
    # Standard sizes
    "7b": (32, 4096, 32, 32, 11008),  # ~7B params
    "13b": (40, 5120, 40, 40, 13824),  # ~13B params
    # Large models
    "30b": (60, 6656, 52, 52, 17920),  # ~30B params
    "70b": (80, 8192, 64, 8, 28672),  # ~70B params (MQA)
    # Custom model
    "custom": (12, 192, 4, 4, 768),
}


def parse_args():
    """Parse command line arguments for the Slurm launcher."""
    parser = argparse.ArgumentParser(
        description="Nanotron Slurm Launcher", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--run", type=str, default="nanotron", help="Name for this experiment run")

    # Slurm job configuration
    slurm_group = parser.add_argument_group("Slurm Configuration")
    slurm_group.add_argument("--gpus_per_node", type=int, default=8, help="Number of GPUs per node")
    slurm_group.add_argument("--partition", type=str, default="hopper-prod", help="Slurm partition to use")
    slurm_group.add_argument("--qos", type=str, default="normal", help="Slurm QOS to use")
    slurm_group.add_argument("--time_limit", type=str, default=None, help="Time limit for the job (HH:MM:SS)")
    slurm_group.add_argument("--email", type=str, default=None, help="Email for job notifications")
    slurm_group.add_argument("--tmp_dir", type=str, default="/tmp", help="Temporary directory on compute nodes")
    slurm_group.add_argument("--pre_launch_commands", type=str, default="", help="Commands to run before job launch")
    slurm_group.add_argument("--extra_env", type=str, default="", help="Additional environment variables")
    slurm_group.add_argument("--bench", type=str, default="", help="Benchmark csv path")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the Nanotron config file. If not provided, a config will be created automatically.",
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="custom",
        choices=MODEL_SIZES.keys(),
        help="Predefined model size",
    )
    model_group.add_argument("--hidden-size", type=int, default=None, help="Hidden size (overrides model)")
    model_group.add_argument("--intermediate-size", type=int, default=None, help="Intermediate size (overrides model)")
    model_group.add_argument("--num-layers", type=int, default=None, help="Number of layers (overrides model)")
    model_group.add_argument("--num-heads", type=int, default=None, help="Number of attention heads (overrides model)")
    model_group.add_argument("--num-kv-heads", type=int, default=None, help="Number of KV heads (overrides model)")
    model_group.add_argument("--vocab-size", type=int, default=65536, help="Vocabulary size (overrides model)")
    model_group.add_argument("--seq", type=int, default=4096, help="Maximum sequence length")

    # Training configuration
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    training_group.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    training_group.add_argument("--mbs", type=int, default=2, help="Micro batch size")
    training_group.add_argument("--acc", type=int, default=8, help="Gradient accumulation steps")
    training_group.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate")
    training_group.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate for decay")
    training_group.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    training_group.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    training_group.add_argument("--warmup-steps", type=int, default=1000, help="Learning rate warmup steps")

    # Parallelism strategy
    parallel_group = parser.add_argument_group("Parallelism Configuration")
    parallel_group.add_argument("--dp", type=int, default=8, help="Data parallelism (DP) degree")
    parallel_group.add_argument("--pp", type=int, default=1, help="Pipeline parallelism (PP) degree")
    parallel_group.add_argument("--tp", type=int, default=2, help="Tensor parallelism (TP) degree")
    parallel_group.add_argument("--cp", type=int, default=1, help="Context parallelism degree")
    parallel_group.add_argument("--ep", type=int, default=1, help="Expert parallelism degree")
    parallel_group.add_argument("--zero", type=int, default=0, choices=[0, 1], help="ZeRO stage")

    # Dataset configuration
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name or path")
    data_group.add_argument("--text-column", type=str, default="text", help="Column name for text in the dataset")
    data_group.add_argument(
        "--tokenizer", type=str, default="robot-test/dummy-tokenizer-wordlevel", help="Tokenizer name or path"
    )

    # File paths
    paths_group = parser.add_argument_group("File Paths")
    paths_group.add_argument("--project", type=str, default="nanotron", help="Project name for logging")
    paths_group.add_argument(
        "--configs-path", type=str, default=DEFAULT_CONFIGS_PATH, help="Directory to save configuration files"
    )
    paths_group.add_argument(
        "--slurm-logs-path", type=str, default=DEFAULT_SLURM_LOGS_PATH, help="Directory for Slurm output logs"
    )
    paths_group.add_argument(
        "--checkpoints-path",
        type=str,
        default=DEFAULT_CHECKPOINTS_PATH,
        help="Base directory for saving model checkpoints",
    )
    slurm_group.add_argument(
        "--run-train-script",
        type=str,
        default=DEFAULT_RUN_TRAIN_SCRIPT,
        help="Path to the training script (default: run_train.py)",
    )
    slurm_group.add_argument(
        "--slurm-scripts-dir",
        type=str,
        default=DEFAULT_SLURM_SCRIPTS_DIR,
        help="Directory to save generated Slurm scripts (set to None to disable)",
    )
    paths_group.add_argument(
        "--save-interval", type=int, default=1000, help="Interval for saving checkpoints (in steps)"
    )
    paths_group.add_argument("--save-initial-state", action="store_true", help="Save initial state")

    # Logging configuration
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument("--enable-wandb", action="store_true", help="Enable logging to Weights & Biases")
    logging_group.add_argument(
        "--profiler_export_path",
        type=str,
        default=None,
        help="Path to export the profiler tensorboard data. Use `tensorboard --logdir <path>` to view.",
    )
    logging_group.add_argument("--log-lvl", type=str, default="info", help="Log level")
    logging_group.add_argument("--no-sanity", action="store_true", help="Ignore sanity checks")

    # Execution control
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't submit job")
    parser.add_argument("--show-logs", action="store_true", help="Show output of the job as it runs")
    return parser.parse_args()


def generate_model_config(
    model_size: str = "custom",
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
    config_params = MODEL_SIZES.get(model_size, MODEL_SIZES["custom"])
    config_params = {
        "num_hidden_layers": config_params[0],
        "hidden_size": config_params[1],
        "num_attention_heads": config_params[2],
        "num_key_value_heads": config_params[3],
        "intermediate_size": config_params[4],
    }

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
        model_size=args.model,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.seq,
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

    # Use user-provided parallelism directly
    parallelism = ParallelismArgs(
        dp=args.dp,
        pp=args.pp,
        tp=args.tp,
        context_parallel_size=args.cp,
        expert_parallel_size=args.ep,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
        recompute_layer=False,
    )

    # Define tokens configuration
    tokens = TokensArgs(
        sequence_length=args.seq,
        train_steps=args.steps,
        micro_batch_size=args.mbs,
        batch_accumulation_per_replica=args.acc,
    )

    # Calculate global batch size for logging
    gbs = (
        parallelism.dp
        * tokens.batch_accumulation_per_replica
        * tokens.micro_batch_size
        * tokens.sequence_length
        * parallelism.context_parallel_size
        * parallelism.expert_parallel_size
    )
    total_tokens = gbs * args.steps
    print(f"GBS: {(gbs)/1e6:.2f}M, total training tokens: {(total_tokens)/1e9:.2f}B")

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
        zero_stage=args.zero,
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
                # For pretraining:
                # dataset=PretrainDatasetsArgs(
                #     hf_dataset_or_datasets=args.dataset,
                #     text_column_name=args.text_column,
                # ),
                # When using a Nanoset, we need to specify the vocab size of the tokenizer used to tokenize the dataset or larger
                dataset=NanosetDatasetsArgs(
                    dataset_folder="/fsx/loubna/tokenized_for_exps/mcf-dataset",  # 1.4T tokens
                ),
                # For SFT (uncomment to use):
                # dataset=SFTDatasetsArgs(
                #     hf_dataset_or_datasets=args.dataset,
                #     hf_dataset_splits="train",
                #     debug_max_samples=1000,
                # ),
                seed=args.seed,
            ),
        ),
    ]
    # Configure checkpointing
    os.makedirs(args.checkpoints_path, exist_ok=True)
    checkpoints = CheckpointsArgs(
        checkpoints_path=os.path.join(args.checkpoints_path, args.run),
        checkpoint_interval=args.save_interval,
        save_initial_state=args.save_initial_state,
    )

    # Create the final config
    config = Config(
        general=GeneralArgs(
            project=args.project,
            run=args.run,
            seed=args.seed,
            ignore_sanity_checks=args.no_sanity,
            benchmark_csv_path=args.bench,
        ),
        checkpoints=checkpoints,
        parallelism=parallelism,
        model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
        tokenizer=TokenizerArgs(args.tokenizer),
        optimizer=optimizer,
        logging=LoggingArgs(log_level=args.log_lvl, log_level_replica=args.log_lvl, iteration_step_info_interval=1),
        tokens=tokens,
        data_stages=data_stages,
        profiler=ProfilerArgs(profiler_export_path=args.profiler_export_path)
        if args.profiler_export_path is not None
        else None,
    )

    return config


def create_slurm_script(
    config_path: str,
    args,
    dp: int,
    pp: int,
    tp: int,
    cp: int,
    ep: int,
    run_train_script: str = "run_train.py",
) -> str:
    """
    Create a Slurm job submission script.

    Args:
        config_path: Path to the Nanotron config YAML file
        args: Command line arguments

    Returns:
        Contents of the Slurm script as a string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = os.path.join(args.slurm_logs_path, args.run.replace(" ", "_"))
    os.makedirs(logs_path, exist_ok=True)

    gpus_per_node = min(args.gpus_per_node, dp * pp * tp * cp * ep)
    assert dp * pp * tp * cp * ep % gpus_per_node == 0
    nodes = dp * pp * tp * cp * ep // gpus_per_node

    # Ensure config_path is a full path
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    script = f"""#!/bin/bash
#SBATCH --job-name={args.run}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=60         # CPU cores per task
#SBATCH --exclusive               # Exclusive use of nodes
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --partition={args.partition}
#SBATCH --output={logs_path}/{timestamp}-%x-%j.out
#SBATCH --qos={args.qos}
#SBATCH --wait-all-nodes=1        # fail if any node is not ready
{f"#SBATCH --time={args.time_limit}" if args.time_limit else ""}
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

# Get the actual slurm script path from the environment
echo "Slurm script path: $(scontrol show job $SLURM_JOB_ID | grep -oP 'Command=\\K[^ ]+')"

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
export GPUS_PER_NODE={gpus_per_node}
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

# Set some environment variables for better distributed training
# export NCCL_DEBUG=WARN # INFO, WARN
# export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Nanotron specific
{"export NANOTRON_BENCHMARK=1" if args.bench else ""}
{"# " if args.enable_wandb else ""}export WANDB_MODE=disabled
# export ENABLE_TIMERS=1
# export DEBUG_CPU=1


CMD="{run_train_script} --config-file {config_path}"

# echo nvcc version and assert we use cuda 12.4
echo "NVCC version: $(nvcc --version)"
if ! nvcc --version | grep -q "12.4"; then
    echo "ERROR: CUDA 12.4 is required to avoid dataloader issues"
    exit 1
fi

# Log system information
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Is debug build: $(python -c 'import torch; print(torch.version.debug)')"
echo "CUDA used to build PyTorch: $(python -c 'import torch; print(torch.version.cuda)')"
echo "ROCM used to build PyTorch: $(python -c 'import torch; print(torch.version.hip)')"

echo "PATH: $PATH"
# Log environment variables
echo "Environment variables:"
printenv | sort

# Log python path
echo "Python path: $(which python)"

# Log torchrun path
echo "Torchrun path: $(which torchrun)"

# Log installed Python packages
echo "Installed Python packages:"
python -m pip freeze


# Log GPU information
nvidia-smi


LAUNCHER="torchrun \\
    --nproc_per_node {gpus_per_node} \\
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

    # Create Nanotron config if not provided
    if args.config is None:
        config = create_nanotron_config(args)
        dp, pp, tp, cp, ep = (args.dp, args.pp, args.tp, args.cp, args.ep)
    else:
        print(f"🔍 Loading config from {args.config}")
        config = Config.load_from_yaml(args.config)
        dp = config.parallelism.dp
        pp = config.parallelism.pp
        tp = config.parallelism.tp
        cp = config.parallelism.context_parallel_size
        ep = config.parallelism.expert_parallel_size
        # bench
        if args.bench:
            config.general.benchmark_csv_path = args.bench

    # Save config to YAML file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run.replace(" ", "_")
    config_dir = os.path.join(args.configs_path, run_name)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{timestamp}-{run_name}.yaml")
    config.save_as_yaml(config_path)
    print(f"💾 Config saved to {config_path}")
    config.print_config_details()

    # Create Slurm script
    slurm_script = create_slurm_script(config_path, args, dp, pp, tp, cp, ep, args.run_train_script)

    # Save Slurm script if requested
    if args.slurm_scripts_dir is not None:
        os.makedirs(args.slurm_scripts_dir, exist_ok=True)
        slurm_script_path = os.path.join(args.slurm_scripts_dir, f"{timestamp}-{run_name}.sh")
        with open(slurm_script_path, "w") as f:
            f.write(slurm_script)
        print(f"💾 Slurm script saved to {slurm_script_path}")

    # Either submit the job or just print the script (dry run)
    if args.dry_run:
        print("DRY RUN - Job script:")
        print(slurm_script)
        print(f"🔍 Would submit job with config from {config_path}")
    else:
        job_id = launch_slurm_job(slurm_script)
        print(f"🚀 Slurm job submitted with JOBID: {job_id}")
        print(
            f"🔍 Logs will be available at: {os.path.join(args.slurm_logs_path, run_name, f'{timestamp}-{run_name}-{job_id}.out')}"
        )

        # Tail output file when available
        if args.show_logs:
            tail_output_file(os.path.join(args.slurm_logs_path, run_name, f"{timestamp}-{run_name}-{job_id}.out"))

    return 0


if __name__ == "__main__":
    main()
