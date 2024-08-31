import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime
import torch
import argparse
import json
from jinja2 import Template

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
            setattr(obj, part, type('', (), {})()) 
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="path to the configuration file", type=str)
    parser.add_argument("--override", nargs="+", metavar="KEY=VALUE",
                        help="Override config values. Use dot notation for nested keys.")
    parser.add_argument("--slurm", action="store_true", help="Launch the job on Slurm")
    parser.add_argument("--nodes", type=int, help="Number of nodes to use for the job")
    args = parser.parse_args()

    if args.slurm:
        if args.nodes is None:
            raise ValueError("When using Slurm (--slurm), you must specify the number of nodes (--nodes)")

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

    num_nodes = args.nodes if args.slurm else 1
    print(f"""
🎛️ Parallelism Configuration:
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
    if args.slurm:
        
        nodes = args.nodes

        launch_slurm_config_path = os.path.join(os.path.dirname(__file__), "src/nanotron/slurm/launch_slurm_config.json")
        eval_slurm_config_path = os.path.join(os.path.dirname(__file__), "src/nanotron/slurm/eval_slurm_config.json")
        
        with open(launch_slurm_config_path, 'r') as f:
            launch_slurm_config = json.load(f)
        
        with open(eval_slurm_config_path, 'r') as f:
            eval_slurm_config = json.load(f)
        
            
        total_gpus = config.parallelism.dp * config.parallelism.pp * config.parallelism.tp
        gpus_per_node = launch_slurm_config.get('gpus_per_node')
        required_nodes = (total_gpus + gpus_per_node - 1) // gpus_per_node  # Ceiling division

        if args.nodes != required_nodes:
            raise ValueError(f"Number of nodes in config ({args.nodes}) does not match the required number of nodes ({required_nodes}) based on the parallelism configuration.")

            
        # Create necessary folders
        project_log_folder = Path(config.general.logs_path)
        log_folder = project_log_folder / f"{config.general.run}-{config.general.project}"
        subfolders = ['launch-script', 'slurm']
        if hasattr(config, 'lighteval') and config.lighteval is not None:
            subfolders.append('evals')

        for subfolder in subfolders:
            folder_path = os.path.join(log_folder, subfolder)
            os.makedirs(folder_path, exist_ok=True)
            if subfolder == 'launch-script':
                config.general.launch_script_path = folder_path
            elif subfolder == 'slurm':
                config.general.slurm_logs_path = folder_path
            elif subfolder == 'evals':
                config.general.evals_logs_path = folder_path
                for evals_subfolder in ['launch-config', 'logs']:
                    evals_subfolder_path = os.path.join(config.general.evals_logs_path, evals_subfolder)
                    os.makedirs(evals_subfolder_path, exist_ok=True)
        
        
        torchrun_args = ""
        if 'torchrun_args' in launch_slurm_config and launch_slurm_config['torchrun_args']:
            torchrun_args = " ".join([f"--{k} {v}" for k, v in launch_slurm_config['torchrun_args'].items()])
        
        launch_slurm_config.update({
            "job_name": f"{config.general.project}-{config.general.run}",
            "nodes": args.nodes,
            "slurm_logs_path": config.general.slurm_logs_path,
            "timestamp": timestamp,
            "path_to_trainer_python_file": os.path.join(os.path.dirname(__file__), "run_train.py"),
            "config_path_yaml": f"{config.general.config_logs_path}/{timestamp}_launch.yaml",
            "torchrun_args": torchrun_args,
        })

        # Load Jinja2 template
        template_path = os.path.join(os.path.dirname(__file__), "src/nanotron/slurm/launch_training.slurm.jinja")
        with open(template_path, 'r') as f:
            template = Template(f.read())

        # Render the template
        sbatch_script = template.render(**launch_slurm_config)

        config.general.launch_slurm_config = launch_slurm_config
        config.general.eval_slurm_config = eval_slurm_config

        config.save_as_yaml(launch_slurm_config["config_path_yaml"])
        
        # Launch the Slurm job
        job_id = launch_slurm_job(sbatch_script)
        print(f"🚀 Slurm job launched with id={job_id}")

        # Save the Slurm script if a path is provided
        if config.general.launch_script_path:
            os.makedirs(config.general.launch_script_path, exist_ok=True)
            script_filename = f"slurm_script_{timestamp}.slurm"
            script_path = os.path.join(config.general.launch_script_path, script_filename)
            
            with open(script_path, 'w') as f:
                f.write(sbatch_script)
            
        print(f"    💾 Logs are saved to : {config.general.logs_path}/{config.general.run}-{config.general.project}")
        print(f"    🤖 Slurm Configuration Details:")

        slurm_config_keys = ['qos', 'gpus_per_node', 'cpus_per_task', 'constraint', 'account', 'reservation']
        for key in slurm_config_keys:
            if key in launch_slurm_config:
                if launch_slurm_config[key] is not None:
                    print(f"        {key}: {launch_slurm_config[key]}")

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