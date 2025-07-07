""" Mostly complete a SLURM template with a link to a single checkpoint on s3 and launch it
"""
import datetime
import math
import os
import subprocess
from typing import List, Optional, Tuple

from datasets.download.streaming_download_manager import xPath

from nanotron import logging
from nanotron.config import Config, LightEvalConfig
from nanotron.data.s3_utils import _get_s3_path_components
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext

logger = logging.get_logger(__name__)


class LightEvalRunner:
    def __init__(self, config: Config, parallel_context: Optional[ParallelContext] = None):
        self.config = config
        assert config.lighteval is not None, "LightEval config is required"
        self.lighteval_config = config.lighteval
        self.parallel_context = parallel_context

    def eval_single_checkpoint(self, uploaded_files: List[dict]) -> Tuple[str, str]:
        """Run light evaluation on uploaded files."""
        if (
            self.config.lighteval.eval_interval is not None
            and self.config.general.step % self.config.lighteval.eval_interval != 0
        ):
            logger.debug(
                f"Skipping evaluation at step {self.config.general.step} because eval_interval is {self.config.lighteval.eval_interval}"
            )
            return
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
                f"Found multiple config files in uploaded checkpoints: {config_files}",
                logger=logger,
                level=logging.ERROR,
                group=self.parallel_context.dp_pg if self.parallel_context is not None else None,
                rank=0,
            )
            return
        checkpoint_path = config_files[0]["destination"].replace("config.yaml", "")
        logger.warning(
            f"Lighteval Runner got {len(uploaded_files)} files. Using {checkpoint_path} as checkpoint path."
        )
        if self.config.general.step % self.lighteval_config.eval_interval == 0:
            slurm_job_id, slurm_log = run_slurm_one_job(
                config=self.config,
                lighteval_config=self.lighteval_config,
                model_checkpoint_path=checkpoint_path,
                current_step=self.config.general.step,
            )
        else:
            logger.warning(f"Skipping evaluation at step {self.config.general.step} because it's not a multiple of {self.lighteval_config.eval_interval}")
            return None, None

        return slurm_job_id, slurm_log


def normalize_s3_path(path: str) -> str:
    """Normalize S3 path using existing s3_utils"""
    # Use existing utility to normalize path components
    path = xPath(path)
    bucket, prefix = _get_s3_path_components(path)
    # Reconstruct normalized path
    return f"s3://{bucket}/{prefix}".rstrip("/")


def run_slurm_one_job(
    config: Config,
    lighteval_config: LightEvalConfig,
    model_checkpoint_path: str,
    current_step: int,
):
    """Launch a single job on Slurm with the given mapping"""
    # Normalize S3 path if needed
    if model_checkpoint_path.startswith(("s3:/", "s3://")):
        model_checkpoint_path = normalize_s3_path(model_checkpoint_path)
        logger.info(f"Normalized S3 path: {model_checkpoint_path}")

    # Use config values instead of hardcoded defaults
    slurm_config = lighteval_config.slurm

    # Calculate the number of nodes based on parallelism config
    total_gpus_needed = (
        lighteval_config.parallelism.dp * lighteval_config.parallelism.pp * lighteval_config.parallelism.tp
    )
    nodes = math.ceil(total_gpus_needed / slurm_config.gpus_per_node)

    # Get timestamp for log files
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    general_run_name = config.general.run
    run_name = f"{timestamp}-eval_{general_run_name}".replace(" ", "_")

    # Use lighteval config paths if available, otherwise use defaults
    eval_launch_script_path = lighteval_config.slurm_script_dir
    eval_logs_path = lighteval_config.logs_path
    eval_launch_script_path = os.path.join(eval_launch_script_path, general_run_name, f"step-{current_step}")
    eval_logs_path = os.path.join(eval_logs_path, general_run_name, f"step-{current_step}")

    # Create directories
    os.makedirs(eval_launch_script_path, exist_ok=True)
    os.makedirs(eval_logs_path, exist_ok=True)

    # Use configured local path instead of hardcoded /tmp
    local_path = os.path.join(lighteval_config.local_checkpoint_dir, run_name, str(current_step))
    nanotron_path = lighteval_config.nanotron_path
    # Create the SLURM script content
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=eval_{current_step}_{run_name}
#SBATCH --partition={slurm_config.partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={slurm_config.cpus_per_task}
#SBATCH --gpus={slurm_config.gpus_per_node}
#SBATCH --exclusive
#SBATCH --qos={slurm_config.qos}
#SBATCH --time={slurm_config.time}
#SBATCH --output={eval_logs_path}/%j-{timestamp}.out
#SBATCH --requeue"""

    if slurm_config.reservation:
        slurm_script += f"\n#SBATCH --reservation={slurm_config.reservation}"

    # Rest of the script content
    slurm_script += f"""

set -x

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
export HUGGINGFACE_HUB_CACHE={slurm_config.hf_cache}
export HF_DATASETS_CACHE={slurm_config.hf_cache}
export HF_MODULES_CACHE={slurm_config.hf_cache}
export HF_HOME={slurm_config.hf_cache}

echo "Running on $COUNT_NODE nodes: $HOSTNAMES"

# Create checkpoint directory
mkdir -p $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER

# Handle S3 paths
if [[ "{model_checkpoint_path}" == s3://* ]]; then
    echo "Downloading checkpoint from S3: {model_checkpoint_path}"

    # First check if the S3 path exists
    if ! s5cmd ls "{model_checkpoint_path}" &>/dev/null; then
        echo "Error: S3 path {model_checkpoint_path} does not exist"
        exit 1
    fi

    # Try sync command and check its exit status
    s5cmd cp \\
        --concurrency=50 \\
        --exclude "optimizer/*" \\
        --exclude "random/*" \\
        --exclude "lr_scheduler/*" \\
        --part-size 100 \\
        "{model_checkpoint_path}/*" "$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to sync files from S3"
        exit 1
    fi

    # Verify that config.yaml was downloaded
    if [ ! -f "$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/config.yaml" ]; then
        echo "Error: config.yaml not found in downloaded checkpoint"
        echo "Contents of $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER:"
        ls -la $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/
        exit 1
    fi
else
    echo "Copying checkpoint files from {model_checkpoint_path} to $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER"
    rsync -av --progress --inplace --no-whole-file \\
        --exclude 'optimizer/' \\
        --exclude 'random/' \\
        --exclude 'lr_scheduler/' \\
        {model_checkpoint_path} $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/

    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy files using rsync"
        exit 1
    fi

    # Verify that config.yaml was copied
    if [ ! -f "$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/config.yaml" ]; then
        echo "Error: config.yaml not found in copied checkpoint"
        echo "Contents of $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER:"
        ls -la $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/
        exit 1
    fi
fi

echo "Contents of checkpoint directory:"
ls -la $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/

# Add random sleep to avoid hub request conflicts
# sleep $(( RANDOM % 300 ))

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \\
    --nproc_per_node {slurm_config.gpus_per_node} \\
    --nnodes $COUNT_NODE \\
    --node_rank $SLURM_PROCID \\
    --master_addr $MASTER_ADDR \\
    --master_port $MASTER_PORT \\
    {nanotron_path}/run_evals.py \\
    --checkpoint-config-path $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER/config.yaml \\
    --lighteval-override {lighteval_config.eval_config_override}
    --cache-dir {slurm_config.hf_cache}"""
    if lighteval_config.output_dir is not None and lighteval_config.s3_save_path is not None:
        slurm_script += f"""
s5cmd cp --if-size-differ "{lighteval_config.output_dir}*" {lighteval_config.s3_save_path}/
"""
    if lighteval_config.upload_to_wandb:
        gbs_tok = config.parallelism.dp * config.tokens.micro_batch_size * config.tokens.sequence_length * config.tokens.batch_accumulation_per_replica
        slurm_script += f"""
python {nanotron_path}/src/nanotron/eval/upload_to_wandb.py \\
    --wandb_project {lighteval_config.wandb_project} \\
    --wandb_entity {lighteval_config.wandb_entity} \\
    --model_name {general_run_name} \\
    --results_path {lighteval_config.s3_save_path}/results/results/{general_run_name}/{current_step}/ \\
    --train_step {current_step} \\
    --consumed_tokens {current_step*gbs_tok}
"""
    slurm_script += """
echo "Cleaning up downloaded checkpoints..."
rm -rf "$LOCAL_DOWNLOAD_CHECKPOINT_FOLDER"
echo "Cleanup completed"

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

        output_log = os.path.join(eval_logs_path, f"{timestamp}-{run_name}-{job_ids}.out")

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
    # checkpoint_path = "checkpoints/smollm3-training-test-tps-48nn-seed-6-/10"
    # config_path = os.path.join(checkpoint_path, "config.yaml")
    checkpoint_path = "s3://smollm3/smollm3-3B-final/3B-final-GQA-noTP-2k-seq/20000/"
    config_path = "checkpoints/smollm3-training-test-tps-48nn-seed-6-/10/config.yaml"
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
