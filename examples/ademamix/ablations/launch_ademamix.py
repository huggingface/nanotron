""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os
import subprocess
import tempfile
from datetime import datetime

import torch
from nanotron.config import (
    AdEMAMixOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LlamaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    NanosetDatasetsArgs,
    OptimizerArgs,
    ParallelismArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

NODES = int(os.environ.get("NODES", 1))
NUM_GPUS = int(os.environ.get("NUM_GPUS", 8))

TORCH_COMPILE_DISABLED = int(os.environ.get("TORCH_COMPILE_DISABLED", 1))
QOS = os.environ.get("QOS", "high")

print(
    f"""
Global Environment Variables:
──────────────────────────────────────────────
NODES: {NODES}
NUM_GPUS: {NUM_GPUS}
QOS: {QOS}
──────────────────────────────────────────────
"""
)


PROJECT = "optimizer-ablation"
REPO_ID = f"HuggingFaceTB/elie-{PROJECT}"

CONFIG_PATH = "./configs"
LOGS_PATH = "./logs"
NANOTRON_FOLDER = "."

if __name__ == "__main__":

    # 1.7B model config (replacing the original 360M config)
    model_config = LlamaConfig(
        bos_token_id=0,
        eos_token_id=0,
        hidden_act="silu",
        hidden_size=2048,  # Updated from 960
        initializer_range=0.02,
        intermediate_size=8192,  # Updated from 2560
        max_position_embeddings=2048,
        num_attention_heads=32,  # Updated from 15
        num_hidden_layers=24,  # Updated from 32
        num_key_value_heads=32,  # Updated from 5
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        rope_theta=10000.0,  # Added
        rope_interleaved=False,  # Added
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=49152,
    )

    num_params = human_format(
        model_config.vocab_size * model_config.hidden_size * 2
        + model_config.num_hidden_layers
        * (
            3 * model_config.hidden_size * model_config.intermediate_size
            + 4 * model_config.hidden_size * model_config.hidden_size
        )
    ).replace(".", "p")

    print(f"Model has {num_params} parameters")

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    job_id = os.environ.get("SLURM_JOB_ID", "")

    SEED = 6
    TRAIN_STEPS = 50000
    nb_warmup_steps = 2000
    nb_decay_steps = 10000
    starting_decay_step = TRAIN_STEPS - nb_decay_steps
    learning_rate = 0.0004

    def launch_slurm_job(launch_file_contents, *args):
        """
            Small helper function to save a sbatch script and call it.
        Args:
            launch_file_contents: Contents of the sbatch script
            *args: any other arguments to pass to the sbatch command

        Returns: the id of the launched slurm job

        """
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(launch_file_contents)
            f.flush()
            return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]

    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", help="dataset folder", type=str)
    parser.add_argument("--run_name", help="run name", type=str)
    args = parser.parse_args()

    run_name = args.run_name.replace(" ", "_")

    # Specific name for this run (checkpoints/logs/tensorboard)
    RUN = f"{PROJECT}-{num_params}-{run_name}-seed-{SEED}-{job_id}"

    data_stages = [
        DatasetStageArgs(
            data=DataArgs(
                dataset=NanosetDatasetsArgs(
                    dataset_folder="/fsx/elie_bakouch/data/fw-edu-og",
                ),
                num_loading_workers=0,
                seed=SEED,
            ),
            name="training stage",
            start_training_step=1,
        ),
    ]

    general = GeneralArgs(
        project=PROJECT,
        run=RUN,
        ignore_sanity_checks=True,
        seed=SEED,
    )

    checkpoints = CheckpointsArgs(
        checkpoints_path=f"./checkpoints/{RUN}",
        checkpoints_path_is_shared_file_system=False,
        resume_checkpoint_path=None,
        checkpoint_interval=10000,
        save_initial_state=False,
        save_final_state=False,
    )

    parallelism = ParallelismArgs(
        dp=8,
        pp=1,
        tp=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
        tp_recompute_allgather=True,
        recompute_layer=False,
        expert_parallel_size=1,
    )
    tokens = TokensArgs(
        batch_accumulation_per_replica=4,  # Updated from 4
        micro_batch_size=4,  # Updated from 6
        sequence_length=2048,
        train_steps=TRAIN_STEPS,
        val_check_interval=1000,
        limit_val_batches=0,
        limit_test_batches=0,
    )

    model = ModelArgs(
        model_config=model_config,
        make_vocab_size_divisible_by=1,
        init_method=RandomInit(
            std=0.02,
        ),
        dtype=torch.bfloat16,
        ddp_bucket_cap_mb=50,
    )

    logging = LoggingArgs(
        # 'debug', 'info', '    rning', 'error', 'critical' and 'passive'
        log_level="info",
        log_level_replica="info",
        iteration_step_info_interval=1,
    )

    learning_rate_scheduler = LRSchedulerArgs(
        learning_rate=learning_rate,
        lr_warmup_steps=nb_warmup_steps,
        lr_warmup_style="linear",
        lr_decay_style="linear",
        lr_decay_steps=nb_decay_steps,
        lr_decay_starting_step=starting_decay_step,
        min_decay_lr=0,
    )

    optimizer = OptimizerArgs(
        zero_stage=0,
        weight_decay=0.01,
        clip_grad=0.05,
        accumulate_grad_in_fp32=True,
        learning_rate_scheduler=learning_rate_scheduler,
        optimizer_factory=AdEMAMixOptimizerArgs(
            beta1=0.9,
            beta2=0.999,
            beta3=0.9999,
            alpha=8.0,
            eps=1e-08,
            beta3_warmup=100000,
            alpha_warmup=100000,
        ),
    )

    tokenizer = TokenizerArgs(
        tokenizer_name_or_path="HuggingFaceTB/cosmo2-tokenizer",
    )

    config = Config(
        general=general,
        checkpoints=checkpoints,
        parallelism=parallelism,
        model=model,
        tokenizer=tokenizer,
        logging=logging,
        tokens=tokens,
        optimizer=optimizer,
        data_stages=data_stages,
        profiler=None,
    )

    #### DEBUG MODE
    if os.environ.get("DEBUG_MODE", "0") != "0":
        print("##### WARNING DEBUG MODE #####")
        config.parallelism.dp = 2
        config.parallelism.pp = 2
        config.parallelism.tp = 2
        config.tokens.micro_batch_size = 3
        config.tokens.batch_accumulation_per_replica = 2
        config.checkpoints.save_initial_state = True
        NODES = 1

    # Sanity check that we can load, save to YAML and reload the config
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"{CONFIG_PATH}/{run_name}", exist_ok=True)
    config_path_yaml = f"{CONFIG_PATH}/{run_name}/{timestamp}.yaml"
    config.save_as_yaml(config_path_yaml)

    os.makedirs(f"{LOGS_PATH}/{run_name}", exist_ok=True)

    log_path = f"{LOGS_PATH}/{run_name}/train-{timestamp}-%x-%j.out"

    # Replace individual prints with a comprehensive summary
    print(
        f"""
🔍 Configuration Recap:
──────────────────────────────────────────────
Model: {str(num_params)} params, {str(model_config.num_hidden_layers)} layers, {str(model_config.num_attention_heads)} heads
Parallelism: {str(NODES)} nodes, {str(parallelism.dp * parallelism.pp * parallelism.tp)} GPUs
Training: {str(TRAIN_STEPS)} steps, {str(tokens.micro_batch_size)} batch size, {str(tokens.sequence_length)} seq length
Learning Rate: Initial {str(learning_rate_scheduler.learning_rate)}, Warmup {str(nb_warmup_steps)} steps, Decay {str(nb_decay_steps)} steps
Optimizer: {str(optimizer.optimizer_factory.__class__.__name__)}, Weight Decay {str(optimizer.weight_decay)}
Paths: Config {str(config_path_yaml)}, Log {str(log_path)}, Checkpoint {str(checkpoints.checkpoints_path)}
──────────────────────────────────────────────
"""
    )

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=88
#SBATCH --gres=gpu:{NUM_GPUS}
#SBATCH --partition=hopper-prod
#SBATCH --output={LOGS_PATH}/{run_name}/train-{timestamp}-%x-%j.out
#SBATCH --qos={QOS}
#SBATCH --begin=now+0minutes

source /fsx/elie_bakouch/nanotron/nanotron-env/bin/activate
module load cuda/12.1

TRAINER_PYTHON_FILE={NANOTRON_FOLDER}/run_train.py
set -x -e

echo "START TIME: $(date)"
secs_to_human(){{
    echo "$(( ${{1}} / 3600 )):$(( (${{1}} / 60) % 60 )):$(( ${{1}} % 60 ))"
}}
start=$(date +%s)
echo "$(date -d @${{start}} "+%Y-%m-%d %H:%M:%S"): ${{SLURM_JOB_NAME}} start id=${{SLURM_JOB_ID}}\n"

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR=/scratch
export CUDA_DEVICE_MAX_CONNECTIONS="1"

export WANDB_MODE="disabled"
export NCCL_P2P_LEVEL="LOC" # disable NVLink

echo go $COUNT_NODE
echo $HOSTNAMES

#### SANITY CHECK ####

echo "Python version:"
which python
python --version

echo "CUDA version:"
nvcc --version

echo "Environment variables:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_MAX_CONNECTIONS: $CUDA_DEVICE_MAX_CONNECTIONS"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "COUNT_NODE: $COUNT_NODE"

echo "System information:"
nvidia-smi

##### LAUNCHER ######

CMD=" \
    $TRAINER_PYTHON_FILE \
    --config-file {config_path_yaml}
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node {NUM_GPUS} \\
    --nnodes $COUNT_NODE \\
    --rdzv_backend c10d \\
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \\
    --rdzv_id $SLURM_JOB_ID \\
    --node_rank $SLURM_PROCID \\
    --role $SLURMD_NODENAME: \\
    --max_restarts 0 \\
    --tee 3 \\
    "

# Wait a random number between 0 and 1000 (milliseconds) to avoid too many concurrent requests to the hub
random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

launch_args="srun $SRUN_ARGS -u bash -c $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"


echo "END TIME: $(date)"
"""
    # Save the sbatch script to a file in the slurm-script directory
    import os

    # Create the directory if it doesn't exist
    os.makedirs("./slurm-script", exist_ok=True)
    script_path = f"./slurm-script/job_{timestamp}_{run_name}.slurm"

    # Write the script to the file
    with open(script_path, "w") as f:
        f.write(sbatch_script)

    print(f"Saved sbatch script to {script_path}")
    print(f"Slurm job launched with id={launch_slurm_job(sbatch_script)}")
