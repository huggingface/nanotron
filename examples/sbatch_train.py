"""
Launches a training using SLURM. This script is meant to be used as a template for launching training jobs on HPC clusters.
```
python examples/sbatch_train.py my_run_name
# After the job runs you can attach to the logs by using
sattach <job_id>.0
```
"""
import argparse
import os
import subprocess
import tempfile
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path

import torch
from nanotron.config import (
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LightEvalConfig,
    LightEvalLoggingArgs,
    LightEvalTasksArgs,
    LlamaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
    get_config_from_file,
)
from nanotron.logging import human_format

###########################################
# CHANGE THIS SECTION
BRRR_FOLDER = Path(os.getcwd())
RUN_EVAL_SLURM_TEMPLATE = BRRR_FOLDER / "examples/run_eval.slurm.jinja"
EVAL_SLURM_SCRIPT_DIR = BRRR_FOLDER / "eval-scripts"
LOCAL_TMP_PATH_ON_NODE = BRRR_FOLDER / "tmp"

SLURM_LOGS_PATH = BRRR_FOLDER / "slurm-logs"
BRRR_CONFIGS_PATH = BRRR_FOLDER / "launch-configs"

EMAIL = ""  # "nouamane@huggingface.co"

NODES = 1

# General name to gather the runs on the hub
PROJECT = "fineweb-exps"

REPO_ID = f"HuggingFaceBR4/nouatest-{PROJECT}"
# END CHANGE THIS SECTION
###########################################

# uncomment whatever model you want to use
model_config = LlamaConfig(
    # Config for a 1.82/1.61B model
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=2048,
    initializer_range=0.02,
    intermediate_size=8192,
    max_position_embeddings=2048,
    num_attention_heads=32,
    num_hidden_layers=24,
    num_key_value_heads=32,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=50272,  # GPT2 tokenizer rounded to next multiple of 8
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

# You can SLURM_ARRAY_TASK_ID to run multiple runs with predefined HP
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
job_id = os.environ.get("SLURM_JOB_ID", "")

# Seed for model and data
SEED = [5, 6][task_id % 2]


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name", type=str)
    args = parser.parse_args()

    def print_differences(target, updates, assert_equal=False):
        if not is_dataclass(target) or not is_dataclass(updates):
            raise ValueError("Both target and updates should be dataclass instances")

        for field in fields(target):
            update_value = getattr(updates, field.name)

            if update_value is not None:
                if is_dataclass(update_value):
                    print_differences(getattr(target, field.name), update_value)
                else:
                    target_value = getattr(target, field.name)
                    if update_value != target_value:
                        if update_value.__class__.__module__ != "builtins":
                            continue
                        print(f"{field.name}: {target_value} -> {update_value}")
                        if assert_equal:
                            assert (
                                target_value == update_value
                            ), f"{field.name} is different. {target_value} != {update_value}"

    dataset_name = run_name = args.run_name.replace(" ", "_")

    # Specific name for this run (checkpoints/logs/tensorboard)
    RUN = f"{PROJECT}-{num_params}-{dataset_name}-seed-{SEED}-{job_id}"

    dataset = PretrainDatasetsArgs(hf_dataset_or_datasets="HuggingFaceTB/cosmopedia-100k", text_column_name="text")

    data_stages = [
        DatasetStageArgs(
            name="Stable Training Stage", start_training_step=1, data=DataArgs(dataset=dataset, seed=SEED)
        ),
    ]

    general = GeneralArgs(
        project=PROJECT,
        run=RUN,
        ignore_sanity_checks=True,
    )

    lighteval = LightEvalConfig(
        tasks=LightEvalTasksArgs(
            tasks="early-signal",  # "generatives", "all"
            custom_tasks="brrr.lighteval.evaluation_tasks",
            max_samples=1000,  # Cap very large evals or for debugging
            dataset_loading_processes=8,
        ),
        parallelism=ParallelismArgs(
            dp=8,
            pp=1,
            tp=1,
            pp_engine="1f1b",
            tp_mode="ALL_REDUCE",
            tp_linear_async_communication=False,
        ),
        batch_size=16,
        # wandb=LightEvalWandbLoggerConfig(
        #     wandb_project=PROJECT,
        #     wandb_entity="huggingface",
        #     wandb_run_name=f"{RUN}_evals",
        # ),
        logging=LightEvalLoggingArgs(
            local_output_path=f"{LOCAL_TMP_PATH_ON_NODE}/lighteval/{RUN}",
            push_details_to_hub=False,
            push_results_to_hub=True,
            push_results_to_tensorboard=True,
            # hub_repo_details=REPO_ID,
            hub_repo_results=REPO_ID,
            hub_repo_tensorboard=REPO_ID,
            tensorboard_metric_prefix="e",
        ),
        slurm_template=RUN_EVAL_SLURM_TEMPLATE,
        slurm_script_dir=EVAL_SLURM_SCRIPT_DIR,
    )

    checkpoints = CheckpointsArgs(
        checkpoints_path=f"{LOCAL_TMP_PATH_ON_NODE}/checkpoints/{RUN}",
        checkpoints_path_is_shared_file_system=False,
        resume_checkpoint_path=None,
        checkpoint_interval=500,
        save_initial_state=True,
    )

    parallelism = ParallelismArgs(
        dp=0,
        pp=1,
        tp=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )
    parallelism.dp = int(
        NODES * 8 // parallelism.pp // parallelism.tp
    )  # How many remaining GPU when taking into account PP, TP and 8 GPUs per node

    tokens = TokensArgs(
        batch_accumulation_per_replica=4,
        micro_batch_size=4,
        sequence_length=2048,
        train_steps=1,
        val_check_interval=100,
    )

    model = ModelArgs(
        model_config=model_config,
        make_vocab_size_divisible_by=8,
        init_method=RandomInit(
            std=0.02,
            # std=1 / math.sqrt(model_config.hidden_size)  # 0.01275  # Basically 1/sqrt(N),
        ),
        dtype=torch.bfloat16,
    )

    logging = LoggingArgs(
        # 'debug', 'info', 'warning', 'error', 'critical' and 'passive'
        log_level="info",
        log_level_replica="info",
        iteration_step_info_interval=1,
    )

    optimizer = OptimizerArgs(
        accumulate_grad_in_fp32=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0,
        torch_adam_is_fused=True,
        weight_decay=0.1,
        zero_stage=0,
        learning_rate_scheduler=LRSchedulerArgs(
            learning_rate=3e-4,
            lr_warmup_steps=500,
            lr_warmup_style="linear",
            lr_decay_style="cosine",
            # lr_decay_steps=10000-500,  # Keeping it to 10k for comparison for now
            min_decay_lr=3.0e-5,
        ),
    )

    tokenizer = TokenizerArgs(
        tokenizer_name_or_path="gpt2",
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
        lighteval=lighteval,
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
    os.makedirs(f"{BRRR_CONFIGS_PATH}/{run_name}", exist_ok=True)
    config_path_yaml = f"{BRRR_CONFIGS_PATH}/{run_name}/{timestamp}.yaml"
    config.save_as_yaml(config_path_yaml)
    print(f"Config saved to {config_path_yaml}")
    config2 = get_config_from_file(config_path_yaml, config_class=Config)
    print_differences(config, config2)

    os.makedirs(f"{SLURM_LOGS_PATH}/{run_name}", exist_ok=True)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output={SLURM_LOGS_PATH}/{run_name}/train-{timestamp}-%x-%j
#SBATCH --array=1-1%1
#SBATCH --qos=prod
#SBATCH --begin=now+0minutes
#SBATCH --mail-type=ALL
#SBATCH --mail-user={EMAIL}

TRAINER_PYTHON_FILE={BRRR_FOLDER}/run_train.py
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

module load cuda/12.1

echo go $COUNT_NODE
echo $HOSTNAMES

##### MOVE TO YAML ######

CMD=" \
    $TRAINER_PYTHON_FILE \
    --config-file {config_path_yaml}
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes $COUNT_NODE \
    --rdzv-backend etcd-v2 \
    --rdzv-endpoint etcd.hpc-cluster-hopper.hpc.internal.huggingface.tech:2379 \
    --rdzv-id $SLURM_JOB_ID \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --max_restarts 0 \
    --tee 3 \
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
    print(f"Slurm job launched with id={launch_slurm_job(sbatch_script)}")
