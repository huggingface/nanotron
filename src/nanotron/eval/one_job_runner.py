""" Mostly complete a SLURM template with a link to a single checkpoint on s3 and launch it
"""
import datetime
import math
import os
import subprocess
from typing import List, Optional, Tuple

from nanotron import logging
from nanotron.config import Config, LightEvalConfig
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext

logger = logging.get_logger(__name__)


class LightEvalRunner:
    def __init__(self, config: Config, parallel_context: Optional[ParallelContext] = None):
        self.config = config
        assert config.lighteval is not None, "LightEval config is required"
        self.lighteval_config = config.lighteval
        self.parallel_context = parallel_context

    def eval_single_checkpoint_no_s3(self, checkpoint_path: str) -> Tuple[str, str]:
        raise NotImplementedError("Not implemented")

    def eval_single_checkpoint(self, uploaded_files: List[dict]) -> Tuple[str, str]:
        """Run light evaluation on uploaded files."""
        logger.warning(f"Lighteval Runner got {len(uploaded_files)} files. Checking configs.")
        config_files = [
            f for f in uploaded_files if "config.py" in f["destination"] or "config.yaml" in f["destination"]
        ]
        # Sanity check on the config files len (we want only one)
        if len(config_files) == 0:
            log_rank(
                "No config files founds in uploaded checkpoints. Not running evaluation.",
                logger=logger,
                level=logging.ERROR,
                group=self.parallel_context.dp_pg if self.parallel_context is not None else None,
                rank=0,
            )
            return
        if len(config_files) > 1:
            log_rank(
                "Found multiple config files in uploaded checkpoints.",
                logger=logger,
                level=logging.ERROR,
                group=self.parallel_context.dp_pg if self.parallel_context is not None else None,
                rank=0,
            )
            return
        checkpoint_path = config_files[0]["destination"].replace("config.yaml", "")

        slurm_job_id, slurm_log = run_slurm_one_job(
            config=self.config,
            lighteval_config=self.lighteval_config,
            model_checkpoint_path=checkpoint_path,
            current_step=self.config.general.step,
        )

        return slurm_job_id, slurm_log


def run_slurm_one_job(
    config: Config,
    lighteval_config: LightEvalConfig,
    model_checkpoint_path: str,
    current_step: int,
):
    """Launch a single job on Slurm with the given mapping"""
    # Default evaluation config
    default_slurm_config = {
        "gpus_per_node": 8,
        "partition": "hopper-prod",
        "hf_cache": "/fsx/nouamane/.cache/huggingface",
        "cpus_per_task": 88,
        "qos": "high",
        "time": "24:00:00",
        "reservation": "smollm",
    }

    # Use lighteval config paths if available, otherwise use defaults
    eval_launch_script_path = os.path.join(
        lighteval_config.slurm_script_dir
        if lighteval_config.slurm_script_dir
        else "/fsx/nouamane/projects/nanotron/eval_results/launch-config",
        str(current_step),
    )
    eval_logs_path = os.path.join(
        lighteval_config.checkpoints_path
        if lighteval_config.checkpoints_path
        else "/fsx/nouamane/projects/nanotron/eval_results/logs",
        str(current_step),
    )

    # Create directories
    os.makedirs(eval_launch_script_path, exist_ok=True)
    os.makedirs(eval_logs_path, exist_ok=True)

    # Calculate the number of nodes based on parallelism config
    total_gpus_needed = (
        lighteval_config.parallelism.dp * lighteval_config.parallelism.pp * lighteval_config.parallelism.tp
    )
    gpus_per_node = default_slurm_config["gpus_per_node"]
    nodes = math.ceil(total_gpus_needed / gpus_per_node)

    # Get timestamp for log files
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"eval_{current_step}".replace(" ", "_")

    # Create log directory with run name subdirectory
    logs_path = os.path.join(eval_logs_path, run_name)
    os.makedirs(logs_path, exist_ok=True)

    # Create the SLURM script content
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --partition={default_slurm_config["partition"]}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={default_slurm_config["cpus_per_task"]}
#SBATCH --gpus={gpus_per_node}
#SBATCH --exclusive
#SBATCH --qos={default_slurm_config["qos"]}
#SBATCH --time={default_slurm_config["time"]}
#SBATCH --output={logs_path}/{timestamp}-%x-%j.out"""

    if default_slurm_config.get("reservation"):
        slurm_script += f"\n#SBATCH --reservation={default_slurm_config['reservation']}"

    # Add the rest of the script content
    local_path = os.path.join("/tmp", f"eval_{config.general.run}", str(current_step))

    slurm_script += f"""

set -x -e

LOCAL_DOWNLOAD_CHECKPOINT_FOLDER={local_path}

echo "START TIME: $(date)"
#Show some environment variables
echo python3 version = `python3 --version`
echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Hugging Face token setup
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
  if TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null); then
    export HUGGING_FACE_HUB_TOKEN=$TOKEN
  else
    echo "Error: The environment variable HUGGING_FACE_HUB_TOKEN is not set and the token cache could not be read."
    exit 1
  fi
fi

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Set HuggingFace cache locations
export HUGGINGFACE_HUB_CACHE={default_slurm_config["hf_cache"]}
export HF_DATASETS_CACHE={default_slurm_config["hf_cache"]}
export HF_MODULES_CACHE={default_slurm_config["hf_cache"]}
export HF_HOME={default_slurm_config["hf_cache"]}

echo "Running on $COUNT_NODE nodes: $HOSTNAMES"

# Create checkpoint directory
mkdir -p $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER

# Handle S3 paths
if [[ "{model_checkpoint_path}" == s3://* ]]; then
    echo "Downloading checkpoint from S3: {model_checkpoint_path}"
    s5cmd sync \
        --concurrency=50 \
        --size-only \
        --exclude "optimizer/*" \
        --exclude "random/*" \
        --exclude "lr_scheduler/*" \
        --part-size 100 \
        "{model_checkpoint_path}/*" "$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/"
else
    echo "Copying checkpoint files from {model_checkpoint_path} to $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER"
    rsync -av --progress --inplace --no-whole-file \
        --exclude 'optimizer/' \
        --exclude 'random/' \
        --exclude 'lr_scheduler/' \
        {model_checkpoint_path} $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/
fi

echo "Contents of checkpoint directory:"
ls -la $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/

# Add random sleep to avoid hub request conflicts
# sleep $(( RANDOM % 300 ))

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \\
    --nproc_per_node {gpus_per_node} \\
    --nnodes $COUNT_NODE \\
    --node_rank $SLURM_PROCID \\
    --master_addr $MASTER_ADDR \\
    --master_port $MASTER_PORT \\
    run_evals.py \\
    --checkpoint-config-path $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/config.yaml \\
    --lighteval-override smollm3_eval.yaml"""

    #     if lighteval_config.batch_size:
    #         slurm_script += f" \\\n    --batch-size {lighteval_config.batch_size}"

    #     if lighteval_config.tasks:
    #         slurm_script += """
    # if [ -n "${TASKS}" ]; then
    #     CMD="$CMD --tasks ${TASKS}"
    # fi
    # if [ -n "${CUSTOM_TASKS}" ]; then
    #     CMD="$CMD --custom-tasks ${CUSTOM_TASKS}"
    # fi
    # if [ -n "${MAX_SAMPLES}" ]; then
    #     CMD="$CMD --max-samples ${MAX_SAMPLES}"
    # fi"""

    slurm_script += """

echo "END TIME: $(date)"
"""

    # Write the script to file
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    launch_script_path = os.path.join(eval_launch_script_path, f"launch_script-{current_time}.slurm")
    os.makedirs(os.path.dirname(launch_script_path), exist_ok=True)

    with open(launch_script_path, "w") as f:
        f.write(slurm_script)

    # Preserve important environment variables
    env = {
        "PATH": os.environ["PATH"],
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "HOME": os.path.expanduser("~"),
    }

    try:
        # Use subprocess.run instead of check_output for better error handling
        result = subprocess.run(["sbatch", launch_script_path], env=env, check=True, capture_output=True, text=True)
        output = result.stdout
        job_ids = output.split()[-1]

        output_log = os.path.join(logs_path, f"{timestamp}-{run_name}-{job_ids}.out")

        logger.warning(
            f"""🚀 Slurm job launched successfully:
            Job name: {run_name}
            Job ID: {job_ids}
            Launch script: {launch_script_path}
            Log file: {output_log}"""
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error while launching Slurm job: {e}")
        logger.error(f"Command output: {e.output}")
        logger.error(f"Command stderr: {e.stderr}")
        job_ids = None
        output_log = None

    return job_ids, output_log


if __name__ == "__main__":

    from nanotron.config.config import Config

    # Load existing config from checkpoint
    # checkpoint_path = "/fsx/nouamane/projects/nanotron/checkpoints/smollm3-training-test-tps-48nn-seed-6-/10"
    # config_path = os.path.join(checkpoint_path, "config.yaml")
    checkpoint_path = "s3://smollm3/smollm3-3B-final/3B-final-GQA-noTP-2k-seq/20000/"
    config_path = "/fsx/nouamane/projects/nanotron/checkpoints/smollm3-training-test-tps-48nn-seed-6-/10/config.yaml"
    try:
        # Load the existing config
        print(f"\nLoading config from: {config_path}")
        config = Config.load_from_yaml(config_path)

        # Print config details
        print("\nConfig details:")
        print(f"Project: {config.general.project}")
        print(f"Run: {config.general.run}")
        print(f"Step: {config.general.step}")

        if config.lighteval:
            print("\nLightEval config:")
            print(
                f"Parallelism: dp={config.lighteval.parallelism.dp}, tp={config.lighteval.parallelism.tp}, pp={config.lighteval.parallelism.pp}"
            )
            print(f"Batch size: {config.lighteval.batch_size}")
            print(f"Slurm template: {config.lighteval.slurm_template}")
            print(f"Checkpoints path: {config.lighteval.checkpoints_path}")
            if config.lighteval.tasks:
                print(f"Tasks: {config.lighteval.tasks.tasks}")
                print(f"Custom tasks: {config.lighteval.tasks.custom_tasks}")
                print(f"Max samples: {config.lighteval.tasks.max_samples}")

        # Create test files structure
        test_files = [
            {
                "destination": os.path.join(checkpoint_path, "config.yaml"),
                "source": "existing_config",
            }
        ]

        if config.lighteval is None:
            config.lighteval = LightEvalConfig()

        print("\nInitializing LightEvalRunner...")
        runner = LightEvalRunner(config=config)

        print("\nTesting LightEvalRunner.eval_single_checkpoint()...")
        job_id, log_path = runner.eval_single_checkpoint(test_files)

    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        print("\nTest completed")
