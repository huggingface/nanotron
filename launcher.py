import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from jinja2 import Template
from nanotron.config import (
    Config,
    get_config_from_file,
    save_as_yaml,
)
from nanotron.logging import human_format


def count_subdirectories(path):
    return sum(os.path.isdir(os.path.join(path, item)) for item in os.listdir(path))


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
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(obj, part):
            setattr(obj, part, type("", (), {})())
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", help="path to the configuration file", type=str, default=None, required=True)
    parser.add_argument("--project", help="name of the project", type=str)
    parser.add_argument("--run", help="name of the run", type=str)
    parser.add_argument("--logs-path", help="path to the logs folder", type=str, default="logs")
    parser.add_argument(
        "--override", nargs="+", metavar="KEY=VALUE", help="Override config values. Use dot notation for nested keys."
    )
    parser.add_argument("--slurm", action="store_true", help="Launch the job on Slurm")
    parser.add_argument("--nodes", type=int, help="Number of nodes to use for the job")
    args = parser.parse_args()

    if args.config_path is None:
        raise ValueError("Please provide a config path")

    if args.slurm:
        if args.nodes is None:
            raise ValueError("When using Slurm (--slurm), you must specify the number of nodes (--nodes)")

    # Load the configuration using get_config_from_file
    config = get_config_from_file(args.config_path, config_class=Config)

    if config.general.logs_path is None and args.logs_path is None:
        raise ValueError("Please provide a logs path")
    if config.general.project is None and args.project is None:
        raise ValueError("Please provide a project name")
    elif args.project is not None:
        config.general.project = args.project

    if config.general.run is None and args.run is None:
        raise ValueError("Please provide a run name")
    elif args.run is not None:
        config.general.run = args.run

    num_params = human_format(config.model.model_config.get_llama_param_count()).replace(".", ",")

    if args.override:
        for item in args.override:
            if "=" not in item:
                raise ValueError(f"Invalid override format: {item}. Use KEY=VALUE.")
            key, value = item.split("=", 1)
            try:
                value = eval(value)
            except Exception as e:
                print(f"Warning: Could not evaluate '{value}': {e}")

            set_nested_attribute(config, key, value)

        print("⇄ Applied overrides:")
        for item in args.override:
            print(f"  {item}")

    # Calculate and print learning rate and global batch size information
    lr_initial = config.optimizer.learning_rate_scheduler.learning_rate
    lr_min = config.optimizer.learning_rate_scheduler.min_decay_lr
    lr_warmup_steps = config.optimizer.learning_rate_scheduler.lr_warmup_steps
    lr_decay_steps = config.optimizer.learning_rate_scheduler.lr_decay_steps
    lr_decay_start = config.optimizer.learning_rate_scheduler.lr_decay_starting_step
    lr_decay_style = config.optimizer.learning_rate_scheduler.lr_decay_style

    # Sample/Token per GPU (at once)
    bs_gpu_sample = config.tokens.micro_batch_size
    bs_gpu_token = bs_gpu_sample * config.tokens.sequence_length

    # Sample/Token in one step
    gbs_sample = bs_gpu_sample * config.parallelism.dp * config.tokens.batch_accumulation_per_replica
    gbs_token = gbs_sample * config.tokens.sequence_length

    total_tokens = config.tokens.train_steps * gbs_token
    total_tokens_billions = human_format(total_tokens).replace(".", ",")

    print(
        f"""
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
"""
    )

    num_nodes = args.nodes if args.slurm else 1
    print(
        f"""
🎛️ Parallelism Configuration:
┌───────────────────────┬────────────────────────┐
│ Nodes                 │ {num_nodes:>22d} │
│ Total GPUs            │ {config.parallelism.dp*config.parallelism.pp*config.parallelism.tp:>22d} │
│ Data Parallel (DP)    │ {config.parallelism.dp:>22d} │
│ Pipeline Parallel (PP)│ {config.parallelism.pp:>22d} │
│ Tensor Parallel (TP)  │ {config.parallelism.tp:>22d} │
└───────────────────────┴────────────────────────┘
"""
    )

    print(
        f"""
📙 Training Configuration:
┌───────────────────────┬────────────────────────┐
│ Total Tokens          │ {total_tokens_billions:>22} │
│ Batch Size (per GPU)  │ {bs_gpu_token:>15,d} Tokens │
│ Global Batch Size     │ {gbs_token:>15,d} Tokens │
└───────────────────────┴────────────────────────┘
"""
    )

    print(
        f"""
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
"""
    )
    print(
        f"""
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
"""
    )

    config.general.logs_path = args.logs_path

    path = Path(args.logs_path) / f"{config.general.run}"
    path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_number = count_subdirectories(f"{args.logs_path}/{config.general.run}") + 1
    timestamp_with_run = f"run{run_number:03d}_{timestamp}"
    config.general.timestamp_with_run = timestamp_with_run

    config.general.config_logs_path = str(
        Path(config.general.logs_path) / config.general.run / timestamp_with_run / "config"
    )
    Path(config.general.config_logs_path).mkdir(parents=True, exist_ok=True)

    if config.checkpoints.checkpoints_path is None:
        config.checkpoints.checkpoints_path = str(
            Path(config.general.logs_path) / config.general.run / timestamp_with_run / "checkpoints"
        )
        Path(config.checkpoints.checkpoints_path).mkdir(parents=True, exist_ok=True)

    if args.slurm:

        nodes = args.nodes

        launch_slurm_config_path = Path("slurm/launch_slurm_config.json")
        if config.lighteval is not None:
            eval_slurm_config_path = Path("slurm/eval_slurm_config.json")
            if eval_slurm_config_path.exists():
                config.general.eval_slurm_config = str(eval_slurm_config_path.resolve())
            else:
                raise ValueError("Lighteval SLURM configuration is required but not provided.")
            if config.general.is_s3_available:
                config.general.eval_slurm_template = "slurm/run_eval_s3.slurm.jinja"
            else:
                config.general.eval_slurm_template = "slurm/run_eval.slurm.jinja"

        with open(launch_slurm_config_path, "r") as f:
            launch_slurm_config = json.load(f)

        total_gpus = config.parallelism.dp * config.parallelism.pp * config.parallelism.tp
        gpus_per_node = launch_slurm_config.get("gpus_per_node")
        if total_gpus < gpus_per_node:
            required_nodes = 1
            gpus_per_node = total_gpus
            print(
                "Warning: The total number of GPUs is less than the GPUs per node. You need to adjust to use all available GPUs."
            )
        else:
            required_nodes = (total_gpus + gpus_per_node - 1) // gpus_per_node  # Ceiling division

        if args.nodes != required_nodes:
            raise ValueError(
                f"Number of nodes in config ({args.nodes}) does not match the required number of nodes ({required_nodes}) based on the parallelism configuration."
            )

        # Create necessary folders
        project_log_folder = Path(config.general.logs_path)
        log_folder = project_log_folder / f"{config.general.run}" / f"{timestamp_with_run}"
        subfolders = ["launch-script", "slurm-logs"]
        if hasattr(config, "lighteval") and config.lighteval is not None:
            subfolders.append("evals")

        for subfolder in subfolders:
            folder_path = str(log_folder / subfolder)
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            if subfolder == "launch-script":
                config.general.launch_script_path = folder_path
            elif subfolder == "slurm-logs":
                config.general.slurm_logs_path = folder_path
            elif subfolder == "evals":
                config.general.evals_logs_path = folder_path
                for evals_subfolder in ["launch-config", "logs", "lighteval-logs"]:
                    if evals_subfolder == "lighteval-logs":
                        if config.lighteval.logging.output_dir is None:
                            evals_subfolder_path = str(Path(config.general.evals_logs_path) / evals_subfolder)
                            Path(evals_subfolder_path).mkdir(parents=True, exist_ok=True)
                            config.lighteval.logging.output_dir = evals_subfolder_path
                    else:
                        evals_subfolder_path = str(Path(config.general.evals_logs_path) / evals_subfolder)
                        Path(evals_subfolder_path).mkdir(parents=True, exist_ok=True)

        torchrun_args = ""
        if "torchrun_args" in launch_slurm_config and launch_slurm_config["torchrun_args"]:
            torchrun_args = " ".join([f"--{k} {v}" for k, v in launch_slurm_config["torchrun_args"].items()])

        launch_slurm_config.update(
            {
                "job_name": f"{config.general.project}-{config.general.run}",
                "nodes": args.nodes,
                "slurm_logs_path": config.general.slurm_logs_path,
                "path_to_trainer_python_file": os.path.join(os.path.dirname(__file__), "run_train.py"),
                "config_path_yaml": f"{config.general.config_logs_path}/launch_config.yaml",
                "torchrun_args": torchrun_args,
            }
        )

        # Load Jinja2 template
        template_path = Path("slurm/launch_training.slurm.jinja")
        with open(template_path, "r") as f:
            template = Template(f.read())

        # Render the template
        sbatch_script = template.render(**launch_slurm_config)
        if launch_slurm_config_path.exists():
            config.general.launch_slurm_config = str(launch_slurm_config_path.resolve())
        else:
            config.general.launch_slurm_config = None

        if config.lighteval is not None:
            # Save the lighteval configuration
            lighteval_config = config.lighteval
            Path(config.general.config_logs_path).mkdir(parents=True, exist_ok=True)
            config.general.lighteval_config_path = str(Path(config.general.config_logs_path) / "lighteval_config.yaml")
            save_as_yaml(lighteval_config, config.general.lighteval_config_path)

        config_path_yaml = str(Path(config.general.config_logs_path) / "launch_config.yaml")
        Path(config.general.config_logs_path).mkdir(parents=True, exist_ok=True)
        config.save_as_yaml(config_path_yaml)

        # Launch the Slurm job
        job_id = launch_slurm_job(sbatch_script)
        print(f"🚀 Slurm job launched with id={job_id}")

        # Save the Slurm script if a path is provided
        if config.general.launch_script_path:
            Path(config.general.launch_script_path).mkdir(parents=True, exist_ok=True)
            script_filename = "slurm_launch_script.slurm"
            script_path = str(Path(config.general.launch_script_path) / script_filename)
            script_path = os.path.join(config.general.launch_script_path, script_filename)

            with open(script_path, "w") as f:
                f.write(sbatch_script)
        print("    🤖 Slurm Configuration Details:")

        slurm_config_keys = ["qos", "gpus_per_node", "cpus_per_task", "constraint", "account", "reservation"]
        for key in slurm_config_keys:
            if key in launch_slurm_config:
                if launch_slurm_config[key] is not None:
                    print(f"        {key}: {launch_slurm_config[key]}")

        print("        ")
        print("    📁 Log structure:")
        print(f"    {config.general.logs_path}/{config.general.run}/")
        print(f"    └── {timestamp_with_run}/")
        if config.checkpoints.checkpoints_path == str(
            Path(config.general.logs_path) / config.general.run / timestamp_with_run / "checkpoints"
        ):
            print("        ├── checkpoints/")
        print("        ├── config/")
        print("        ├── launch-script/")
        print("        ├── slurm-logs/")
        if hasattr(config, "lighteval") and config.lighteval is not None:
            print("        └── evals/")
            print("            ├── launch-config/")
            print("            └── logs/")
            if config.lighteval.logging.output_dir == str(Path(config.general.evals_logs_path) / "lighteval-logs"):
                print("            └── lighteval-logs/")

    else:
        # Check if running on an interactive node
        try:
            gpu_count = torch.cuda.device_count()
            is_interactive = gpu_count > 0
        except Exception as e:
            print(f"Warning: Could not get GPU count: {e}")
            is_interactive = False

        if is_interactive:
            print("💻 Running on an interactive node with GPUs.")
            gpu_config = config.parallelism.dp * config.parallelism.tp * config.parallelism.pp
            if gpu_count < gpu_config:
                raise ValueError(
                    f"Error: Your configuration (dp={config.parallelism.dp}, tp={config.parallelism.tp}, pp={config.parallelism.pp}) "
                    f"requires {gpu_config} GPUs, but only {gpu_count} are available."
                )
            elif gpu_count == gpu_config:
                print(
                    f"🚀 Running on {gpu_count} GPUs, which matches your configuration (dp={config.parallelism.dp}, tp={config.parallelism.tp}, pp={config.parallelism.pp})"
                )
                total_gpus = gpu_count
            elif gpu_count > gpu_config:
                total_gpus = gpu_config
                print(
                    f"⚠️  Warning: Your configuration (dp={config.parallelism.dp}, tp={config.parallelism.tp}, pp={config.parallelism.pp}) "
                    f"uses {total_gpus} GPUs, but {gpu_count} are available. "
                    f"You are not fully utilizing all available GPUs on this device."
                )

            config_path_yaml = str(Path(config.general.config_logs_path) / "launch_config.yaml")
            os.makedirs(config.general.config_logs_path, exist_ok=True)
            config.save_as_yaml(config_path_yaml)

            trainer_python_file = "run_train.py"
            cmd = f"{trainer_python_file} --config-file {config_path_yaml}"

            launch_cmd = f"CUDA_DEVICE_MAX_CONNECTIONS='1' torchrun --nproc_per_node {total_gpus} {cmd}"
            print(f"🚀 Launching interactive job with command: {launch_cmd}")

            subprocess.run(launch_cmd, shell=True, check=True)
        else:
            print(
                "❌ Not running on a Slurm cluster or an interactive node with GPUs. Please submit a Slurm job or use an interactive node with GPUs."
            )
