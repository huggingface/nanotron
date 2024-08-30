import os
import subprocess
import tempfile
from datetime import datetime
import torch

import argparse

from nanotron.logging import human_format

from nanotron.config import (
    Config,
    get_config_from_file,
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

def set_nested_attribute(obj, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        if not hasattr(obj, part):
            setattr(obj, part, type('', (), {})())  # Create empty object if attribute doesn't exist
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="path to the configuration file", type=str)
    parser.add_argument("--override", nargs="+", metavar="KEY=VALUE",
                        help="Override config values. Use dot notation for nested keys.")
    args = parser.parse_args()

    # Load the configuration using get_config_from_file
    config = get_config_from_file(args.config_path, config_class=Config)

    if config.model.model_config.tie_word_embeddings ==True:
        tie_word_embeddings_multiplier = 1
    else:
        tie_word_embeddings_multiplier = 2

    num_params = human_format(
        config.model.model_config.vocab_size * config.model.model_config.hidden_size * tie_word_embeddings_multiplier
        + config.model.model_config.num_hidden_layers
        * (
            3 * config.model.model_config.hidden_size * config.model.model_config.intermediate_size
            + 4 * config.model.model_config.hidden_size * config.model.model_config.hidden_size
        )
    ).replace(".", "p")
    # Apply overrides
    if args.override:
        for item in args.override:
            if '=' not in item:
                raise ValueError(f"Invalid override format: {item}. Use KEY=VALUE.")
            key, value = item.split('=', 1)
            try:
                # Try to evaluate the value as a Python literal
                value = eval(value)
            except:
                # If eval fails, treat it as a string
                pass
            
            set_nested_attribute(config, key, value)

        print("Applied overrides:")
        for item in args.override:
            print(f"  {item}")
        
    # Calculate and print learning rate and global batch size information
    lr_initial = config.optimizer.learning_rate_scheduler.learning_rate
    lr_min = config.optimizer.learning_rate_scheduler.min_decay_lr
    lr_warmup_steps = config.optimizer.learning_rate_scheduler.lr_warmup_steps
    lr_decay_steps = config.optimizer.learning_rate_scheduler.lr_decay_steps
    lr_decay_start = config.optimizer.learning_rate_scheduler.lr_decay_starting_step
    lr_decay_style = config.optimizer.learning_rate_scheduler.lr_decay_style

    BS = config.tokens.micro_batch_size*config.tokens.batch_accumulation_per_replica*config.tokens.sequence_length
    GBS = BS * config.parallelism.dp
    
    total_tokens = config.tokens.train_steps * GBS
    total_tokens_billions = total_tokens / 1e9

    print(f"""
🏋️  Model Parameters:
┌───────────────────────┬────────────────────────┐
│ Total Parameters      │ {num_params:>22} │
│ Layers                │ {config.model.model_config.num_hidden_layers:>22d} │
│ Attention Heads       │ {config.model.model_config.num_attention_heads:>22d} │
│ Hidden Size           │ {config.model.model_config.hidden_size:>22d} │
│ Intermediate Size     │ {config.model.model_config.intermediate_size:>22d} │
│ Context Length        │ {config.model.model_config.max_position_embeddings:>22d} │
│ Tokenizer             │ {config.tokenizer.tokenizer_name_or_path[:22]:>22} │
│ Vocab Size            │ {config.model.model_config.vocab_size:>22d} │
└───────────────────────┴────────────────────────┘
""")

    num_nodes = config.slurm.nodes if config.slurm else torch.cuda.device_count()
    print(f"""
🤖 Parallelism Configuration:
┌───────────────────────┬────────────────────────┐
│ Nodes                 │ {num_nodes:>22d} │
│ Total GPUs            │ {config.parallelism.dp*config.parallelism.pp*config.parallelism.tp:>22d} │
│ Data Parallel (DP)    │ {config.parallelism.dp:>22d} │
│ Pipeline Parallel (PP)│ {config.parallelism.pp:>22d} │
│ Tensor Parallel (TP)  │ {config.parallelism.tp:>22d} │ 
└───────────────────────┴────────────────────────┘
""")

    print(f"""
📙 Training Configuration:
┌───────────────────────┬────────────────────────┐
│ Total Tokens          │ {total_tokens_billions:>21.2f}B │
│ Global Batch Size     │ {GBS:>22,d} │
│ Batch Size (per GPU)  │ {BS:>22,d} │
└───────────────────────┴────────────────────────┘
""")

    print(f"""
📊 Learning Rate Schedule:
┌───────────────────────┬────────────────────────┐
│ Initial LR            │ {lr_initial:>22.2e} │
│ Warmup Style          │ {config.optimizer.learning_rate_scheduler.lr_warmup_style[:22]:>22} │
│ Warmup Steps          │ {lr_warmup_steps:>22d} │
│ Decay Style           │ {lr_decay_style[:22]:>22} │
│ Decay Start Step      │ {lr_decay_start:>22d} │
│ Decay Steps           │ {lr_decay_steps:>22d} │
│ Final LR              │ {lr_min:>22.2e} │
└───────────────────────┴────────────────────────┘
""")
    print(f"""
🔧 Optimization Configuration:
┌───────────────────────┬────────────────────────┐
│ Optimizer             │ {config.optimizer.optimizer_factory.__class__.__name__:>22} │
│ Weight Decay          │ {config.optimizer.weight_decay:>22.2e} │
│ Gradient Clipping     │ {config.optimizer.clip_grad:>22.2f} │
│ Adam Epsilon          │ {config.optimizer.optimizer_factory.adam_eps:>22.2e} │
│ Adam Beta1            │ {config.optimizer.optimizer_factory.adam_beta1:>22.2f} │
│ Adam Beta2            │ {config.optimizer.optimizer_factory.adam_beta2:>22.2f} │
│ ZeRO Stage            │ {config.optimizer.zero_stage:>22d} │
│ FP32 Grad Accumulation│ {str(config.optimizer.accumulate_grad_in_fp32):>22} │
└───────────────────────┴────────────────────────┘
""")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.slurm:
        dir = os.path.dirname(__file__)
        
        os.makedirs(config.general.config_logs_path, exist_ok=True)
        config_path_yaml = f"{config.general.config_logs_path}/{timestamp}_launch.yaml"
        config.save_as_yaml(config_path_yaml)
    
        os.makedirs(f"{config.general.slurm_logs_path}/", exist_ok=True)

        def format_sbatch_option(option, value):
            return f"#SBATCH --{option}={value}" if value is not None else ""
        
        torchrun_args = ""
        if hasattr(config.slurm, 'torchrun_args') and config.slurm.torchrun_args:
            torchrun_args = " ".join([f"--{k} {v}" for k, v in config.slurm.torchrun_args.items()])

        sbatch_script = f"""#!/bin/bash
{format_sbatch_option("job-name", config.slurm.job_name)}
{format_sbatch_option("nodes", config.slurm.nodes)}
{format_sbatch_option("ntasks-per-node", config.slurm.n_tasks_per_node)}
{format_sbatch_option("cpus-per-task", config.slurm.cpus_per_task)}
{format_sbatch_option("gres", f"gpu:{config.slurm.gpu_per_node}")}
{format_sbatch_option("partition", config.slurm.gpu_partition)}
{format_sbatch_option("output", f"{config.general.slurm_logs_path}/train-{timestamp}-%j.out")}
{format_sbatch_option("error", f"{config.general.slurm_logs_path}/train-{timestamp}-%j.err")}
{format_sbatch_option("qos", config.slurm.qos)}
{format_sbatch_option("mail-type", config.slurm.mail_type)}
{format_sbatch_option("mail-user", config.slurm.mail_user)}
{format_sbatch_option("exclude", ",".join(config.slurm.exclude_nodes) if config.slurm.exclude_nodes else None)}
{format_sbatch_option("time", config.slurm.time)}
{format_sbatch_option("mem", config.slurm.mem)}
{format_sbatch_option("constraint", config.slurm.constraint)}
{format_sbatch_option("account", config.slurm.account)}
{format_sbatch_option("reservation", config.slurm.reservation)}
{format_sbatch_option("begin", config.slurm.begin)}

set -x -e

TRAINER_PYTHON_FILE=/fsx/elie_bakouch/nanotron/run_train.py
nvidia-smi


#Show some environment variables
echo python3 version = `python3 --version`
echo "Python path: $(which python3)"
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

CMD=" $TRAINER_PYTHON_FILE \
    --config-file {config_path_yaml} \
    "
export LAUNCHER="torchrun \
    --nproc_per_node {config.slurm.gpu_per_node} \
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
        # Save the Slurm script
        print(f"🚀 Slurm job launched with id={launch_slurm_job(sbatch_script)}")
        if config.general.launch_script_path:
            os.makedirs(config.general.launch_script_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_filename = f"slurm_script_{timestamp}.slurm"
            script_path = os.path.join(config.general.launch_script_path, script_filename)
            
            with open(script_path, 'w') as f:
                f.write(sbatch_script)
            
        print(f"    💾 Logs are saved to : {config.general.logs_path}")

    else:
        # Check if running on an interactive node
        try:
            gpu_count = torch.cuda.device_count()
            is_interactive = gpu_count > 0
        except:
            is_interactive = False

        if is_interactive:
            print("💻 Running on an interactive node with GPUs.")
            
            # Check if the parallelism configuration matches the available GPUs
            total_gpus = gpu_count
            config_gpus = config.parallelism.dp * config.parallelism.tp * config.parallelism.pp
            
            if total_gpus != config_gpus:
                raise ValueError(f"The parallelism configuration (dp={config.parallelism.dp}, tp={config.parallelism.tp}, pp={config.parallelism.pp}) "
                                 f"doesn't match the number of available GPUs ({total_gpus}). "
                                 f"Please adjust your configuration to match the available resources.")
            
            # Save config
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs("/fsx/elie_bakouch/nanotron/config_logs", exist_ok=True)
            config_path_yaml = f"/fsx/elie_bakouch/nanotron/config_logs/{timestamp}.yaml"
            config.save_as_yaml(config_path_yaml)

            # Prepare command
            trainer_python_file = "/fsx/elie_bakouch/nanotron/run_train.py"
            cmd = f"{trainer_python_file} --config-file {args.config_path}"

            # Launch job
            launch_cmd = f"CUDA_DEVICE_MAX_CONNECTIONS='1' torchrun --nproc_per_node {gpu_count} {cmd}"
            print(f"🚀 Launching interactive job with command: {launch_cmd}")
            
            # Execute the command
            subprocess.run(launch_cmd, shell=True, check=True)
        else:
            print("❌ Not running on a Slurm cluster or an interactive node with GPUs. Please submit a Slurm job or use an interactive node with GPUs.")