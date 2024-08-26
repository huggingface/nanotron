import os
import subprocess
import tempfile
from datetime import datetime
import math
import torch

import argparse

from nanotron.logging import human_format
from nanotron.models.llama import LlamaConfig

from nanotron.config import (
    Config,
    DataArgs,
    NanosetDatasetsArgs,
    S3UploadArgs,
    SlurmArgs,
    CheckpointsArgs,
    GeneralArgs,
    LightEvalConfig,
    LightEvalLoggingArgs,
    LightEvalTasksArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    AdamWOptimizerArgs,
    ParallelismArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
    LightEvalWandbLoggerConfig,
    DatasetStageArgs,
)

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
    parser.add_argument("project", help="project name", type=str)
    parser.add_argument("--slurm", help="use slurm", action="store_true")
    parser.add_argument("--name", help="run name", type=str, default=None)
    parser.add_argument("--seed", help="seed", type=int, default=8)
    parser.add_argument("--priority", "--qos", "-p", help="qos to use", type=str, default="normal")
    args = parser.parse_args()

    PROJECT = args.project
    if args.name is not None:
        RUN = f"{PROJECT}-{args.name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        RUN = f"{PROJECT}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    ## FOR SANITY CHECK LATER
    # from dataclasses import fields, is_dataclass

    # def print_differences(target, updates):
    #     if not is_dataclass(target) or not is_dataclass(updates):
    #         raise ValueError("Both target and updates should be dataclass instances")

    #     for field in fields(target):
    #         update_value = getattr(updates, field.name)

    #         if update_value is not None:
    #             if is_dataclass(update_value):
    #                 print_differences(getattr(target, field.name), update_value)
    #             else:
    #                 target_value = getattr(target, field.name)
    #                 if update_value != target_value:
    #                     if update_value.__class__.__module__ != "builtins":
    #                         continue
    #                     print(f"{field.name}: {target_value} -> {update_value}")


    general = GeneralArgs(
        project=PROJECT,
        run=RUN,
        repo_id="HuggingFaceSmol/test-nanotron",
        seed=args.seed,
        temp_dir="/scratch",
    )
    if args.slurm:
        slurm = SlurmArgs(
            gpu_partition="hopper-prod",
            job_name=f"{PROJECT}-{args.name}",
            nodes=2,
            logs_path=f"/fsx/elie_bakouch/nanotron/debug",
            conda_path="/fsx/elie_bakouch/miniconda3/etc/profile.d/conda.sh",
            conda_env_path="/fsx/elie_bakouch/miniconda3/envs/smollm",
            exclude_nodes=["ip-26-0-161-138", "ip-26-0-161-178"],
            torchrun_args={
                "rdzv_backend": "etcd-v2",
                "rdzv_endpoint": "etcd.hpc-cluster-hopper.hpc.internal.huggingface.tech:2379",
                "rdzv_id": "$SLURM_JOB_ID"
            },
            qos="normal",
            mail_type="FAIL",
            mail_user="bakouch.elie@gmail.com",
            begin="now+0minutes"
        )

    model_config = LlamaConfig(
        bos_token_id=0,
        eos_token_id=0,
        hidden_act="silu",
        hidden_size=576,
        initializer_range=0.02,
        intermediate_size=1536,
        max_position_embeddings=2048,
        num_attention_heads=9,
        num_hidden_layers=30,
        num_key_value_heads=3,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=49152,
    )
    if model_config.tie_word_embeddings ==True:
        tie_word_embeddings_multiplier = 1
    else:
        tie_word_embeddings_multiplier = 2

    num_params = human_format(
        model_config.vocab_size * model_config.hidden_size * tie_word_embeddings_multiplier
        + model_config.num_hidden_layers
        * (
            3 * model_config.hidden_size * model_config.intermediate_size
            + 4 * model_config.hidden_size * model_config.hidden_size
        )
    ).replace(".", "p")

    # Do we  have a SLURM task ID?
    # You can SLURM_ARRAY_TASK_ID to run multiple runs with predefined HP
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    job_id = os.environ.get("SLURM_JOB_ID", "")
    


    lighteval = LightEvalConfig(
        tasks=LightEvalTasksArgs(
            tasks="early-signal",  # "generatives", "all"
            custom_tasks="nanotron.lighteval.evaluation_tasks",
            max_samples=1000,  # Cap very large evals or for debugging
            dataset_loading_processes=8,
        ),
        parallelism=ParallelismArgs(
            dp=8,
            pp=1,
            tp=1,
            pp_engine="1f1b",
            tp_mode="ALL_REDUCE",
            # recompute_granularity="selective",
            tp_linear_async_communication=False,
        ),
        batch_size=16,
        wandb=LightEvalWandbLoggerConfig(
            wandb_project=PROJECT,
            wandb_entity="eliebak",
            wandb_run_name=f"{RUN}_evals",
        ),
        logging=LightEvalLoggingArgs(
            local_output_path=f"{general.temp_dir}/lighteval/{RUN}",
            push_details_to_hub=False,
            push_results_to_hub=True,
            push_results_to_tensorboard=True,
            #hub_repo_details=REPO_ID,
            hub_repo_results=general.repo_id,
            hub_repo_tensorboard=general.repo_id,
            tensorboard_metric_prefix="e",
        ),
        slurm_template="/fsx/elie_bakouch/nanotron/src/nanotron/lighteval/run_eval.slurm.jinja",
    )


    checkpoints = CheckpointsArgs(
        checkpoints_path=f"checkpoints/{RUN}",
        checkpoints_path_is_shared_file_system=False,
        resume_checkpoint_path=None,
        checkpoint_interval=20,
        save_initial_state=False,
    )

    parallelism = ParallelismArgs(
        dp=16,
        pp=1,
        tp=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )
    #Add sanity check for the number of GPUs and the number of nodes ?

    tokens = TokensArgs(
        batch_accumulation_per_replica=8,
        micro_batch_size=16,
        sequence_length=2048,
        train_steps=100,
        val_check_interval=-1,
    )
    BS = tokens.micro_batch_size*tokens.batch_accumulation_per_replica*tokens.sequence_length
    GBS = BS * parallelism.dp
    
    total_tokens = tokens.train_steps * GBS
    total_tokens_billions = total_tokens / 1e9

    model = ModelArgs(
        model_config=model_config,
        make_vocab_size_divisible_by=1,
        init_method=RandomInit(
            std=math.sqrt(model_config.hidden_size),
        ),
        dtype=torch.bfloat16,
    )

    logging = LoggingArgs(
        # 'debug', 'info', 'warning', 'error', 'critical' and 'passive'
        log_level="info",
        log_level_replica="info",
        iteration_step_info_interval=1,
    )

    learning_rate_scheduler = LRSchedulerArgs(
        learning_rate=1e-4, #llama one
        lr_warmup_steps=10,
        lr_warmup_style="linear",
        lr_decay_style="linear",            
        lr_decay_steps = 20,
        lr_decay_starting_step= 80,
        min_decay_lr=0,
    )
    # Calculate and print learning rate and global batch size information
    lr_initial = learning_rate_scheduler.learning_rate
    lr_min = learning_rate_scheduler.min_decay_lr
    lr_warmup_steps = learning_rate_scheduler.lr_warmup_steps
    lr_decay_steps = learning_rate_scheduler.lr_decay_steps
    lr_decay_start = learning_rate_scheduler.lr_decay_starting_step
    lr_decay_style = learning_rate_scheduler.lr_decay_style
    
    optimizer = OptimizerArgs(
        zero_stage=0,
        weight_decay=0.01,
        clip_grad=1.0,
        accumulate_grad_in_fp32=True,
        learning_rate_scheduler=learning_rate_scheduler,
        optimizer_factory=AdamWOptimizerArgs(
            adam_eps=1e-08,
            adam_beta1=0.9,
            adam_beta2=0.95,
            torch_adam_is_fused=True,
        ),
    )

    tokenizer = TokenizerArgs(
        tokenizer_name_or_path="lvwerra/the-tokenizer-v1",
    )

    s3_upload = S3UploadArgs(
        upload_s3_path=f"s3://elie-exp/debug_nanotron/test/",
        remove_after_upload=True,
        s5cmd_numworkers=16,
        s5cmd_concurrency=5,
        s5cmd_path=os.path.join(slurm.conda_env_path, "bin/s5cmd"),
    )

    data_stages=[
        DatasetStageArgs(
            data=DataArgs(
                dataset=NanosetDatasetsArgs(
                    dataset_folder="/fsx/elie_bakouch/nanotron/datasets/cosmopedia-v2",
                ),
                num_loading_workers=0,
                seed=general.seed,
            ),
            name="training stage",
            start_training_step=1,
        ),
    ]

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
        s3_upload=s3_upload,
        lighteval=lighteval,
        slurm=slurm,
    )

    print(f"""
🏋️  Model Parameters:
┌───────────────────────┬───────────────────────────┐
│ Total Parameters      │ {num_params:>25} │
│ Layers                │ {model_config.num_hidden_layers:>25d} │
│ Attention Heads       │ {model_config.num_attention_heads:>25d} │
│ Hidden Size           │ {model_config.hidden_size:>25d} │
│ Intermediate Size     │ {model_config.intermediate_size:>25d} │
│ Context Length        │ {model_config.max_position_embeddings:>25d} │
│ Tokenizer             │ {tokenizer.tokenizer_name_or_path[:25]:>25} │
│ Vocab Size            │ {model_config.vocab_size:>25d} │
└───────────────────────┴───────────────────────────┘
""")

    print(f"""
🤖 Parallelism Configuration:
┌───────────────────────┬───────────────────┐
│ Nodes                 │ {slurm.nodes:>17d} │
│ Total GPUs            │ {parallelism.dp*parallelism.pp*parallelism.tp:>17d} │
│ Data Parallel (DP)    │ {parallelism.dp:>17d} │
│ Pipeline Parallel (PP)│ {parallelism.pp:>17d} │
│ Tensor Parallel (TP)  │ {parallelism.tp:>17d} │
└───────────────────────┴───────────────────┘
""")

    print(f"""
📙 Training Configuration:
┌───────────────────────┬───────────────────┐
│ Total Tokens          │ {total_tokens_billions:>16.2f}B │
│ Global Batch Size     │ {GBS:>17,d} │
│ Batch Size (per GPU)  │ {BS:>17,d} │
└───────────────────────┴───────────────────┘
""")

    print(f"""
📊 Learning Rate Schedule:
┌───────────────────────┬───────────────────┐
│ Initial LR            │ {lr_initial:>17.2e} │
│ Warmup Style          │ {learning_rate_scheduler.lr_warmup_style[:17]:>17} │
│ Warmup Steps          │ {lr_warmup_steps:>17d} │
│ Decay Style           │ {lr_decay_style[:17]:>17} │
│ Decay Start Step      │ {lr_decay_start:>17d} │
│ Decay Steps           │ {lr_decay_steps:>17d} │
│ Final LR              │ {lr_min:>17.2e} │
└───────────────────────┴───────────────────┘
""")

    if slurm is not None:
        dir = os.path.dirname(__file__)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(config.slurm.config_logs_path, exist_ok=True)
        config_path_yaml = f"{config.slurm.config_logs_path}/{timestamp}.yaml"
        config.save_as_yaml(config_path_yaml)
    
        os.makedirs(f"{config.slurm.slurm_logs_path}/", exist_ok=True)

        def format_sbatch_option(option, value):
            return f"#SBATCH --{option}={value}" if value is not None else ""
        
        torchrun_args = ""
        if hasattr(slurm, 'torchrun_args') and slurm.torchrun_args:
            torchrun_args = " ".join([f"--{k} {v}" for k, v in slurm.torchrun_args.items()])

        sbatch_script = f"""#!/bin/bash
{format_sbatch_option("job-name", slurm.job_name)}
{format_sbatch_option("nodes", slurm.nodes)}
{format_sbatch_option("ntasks-per-node", slurm.n_tasks_per_node)}
{format_sbatch_option("cpus-per-task", slurm.cpus_per_task)}
{format_sbatch_option("gres", f"gpu:{slurm.gpu_per_node}")}
{format_sbatch_option("partition", slurm.gpu_partition)}
{format_sbatch_option("output", f"{slurm.slurm_logs_path}/train-{timestamp}-%x-%j.out")}
{format_sbatch_option("array", slurm.array)}
{format_sbatch_option("qos", slurm.qos)}
{format_sbatch_option("mail-type", slurm.mail_type)}
{format_sbatch_option("mail-user", slurm.mail_user)}
{format_sbatch_option("exclude", ",".join(slurm.exclude_nodes) if slurm.exclude_nodes else None)}
{format_sbatch_option("time", slurm.time)}
{format_sbatch_option("mem", slurm.mem)}
{format_sbatch_option("constraint", slurm.constraint)}
{format_sbatch_option("account", slurm.account)}
{format_sbatch_option("reservation", slurm.reservation)}
{format_sbatch_option("begin", slurm.begin)}

set -x -e

TRAINER_PYTHON_FILE=/fsx/elie_bakouch/nanotron/run_train.py
nvidia-smi
source ~/.bashrc
source /fsx/elie_bakouch/miniconda3/etc/profile.d/conda.sh
conda activate {slurm.conda_env_path} #Modify this line if you use something different than conda


#Show some environment variables
echo python3 version = `python3 --version`
echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"

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
export LAUNCHER="torchrun \
    --nproc_per_node {slurm.gpu_per_node} \
    --nnodes $COUNT_NODE \
    {torchrun_args} \
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