""" Mostly complete a SLURM template with a link to a single checkpoint on s3 and launch it
"""
import datetime
import os
import re
import subprocess
from typing import List, Optional, Tuple, Union

import jinja2
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext

from nanotron.config import Config, LightEvalConfig

logger = logging.get_logger(__name__)


class LightEvalRunner:
    def __init__(self, config: Config, parallel_context: Optional[ParallelContext] = None):
        self.config = config
        self.lighteval_config = config.lighteval
        self.parallel_context = parallel_context

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
            config = self.config,
            slurm_template=self.lighteval_config.slurm_template,
            model_checkpoint_path=checkpoint_path,
        )

        return slurm_job_id, slurm_log


def run_slurm_one_job(
    config: Config,
    model_checkpoint_path: str,
    slurm_template: str,
    slurm_name: Optional[str] = "eval",
    slurm_kwargs: Optional[dict] = None, #add slurm_kwargs and modify the jinja template in case you need to adapt it to your slurm cluster.
):
    """Launch a single job on Slurm with the given mapping
    Args:
        slurm_config: Slurm configuration
        mapping: Mapping to use for the job script (see SLURM_ONE_JOB_MAPPING)
    """

    eval_launch_script_path=os.path.join(config.slurm.evals_logs_path, "launch-config")
    eval_logs_path= os.path.join(config.slurm.evals_logs_path, "logs")

    environment = jinja2.Environment(
        comment_start_string="{=",
        comment_end_string="=}",
    )

    with open(slurm_template, "r") as f:
        SLURM_JOBS_ARRAY_TEMPLATE = environment.from_string(f.read())

    launch_string = SLURM_JOBS_ARRAY_TEMPLATE.render(
        model_checkpoint_path=model_checkpoint_path,
        job_name=f"{slurm_name}-eval",
        n_tasks_per_node=config.slurm.n_tasks_per_node,
        partition=config.slurm.gpu_partition,
        gpu_per_node=config.slurm.gpu_per_node,
        cpus_per_task=config.slurm.cpus_per_task,
        eval_path=eval_logs_path,
        mail=config.slurm.mail,
        conda_path=config.slurm.conda_path,
        conda_env_path=config.slurm.conda_env_path,
        local_path=config.checkpoints.checkpoints_path,
        **(slurm_kwargs if slurm_kwargs else {}),
    )

    match = re.match(r"#SBATCH --output=(.*)", launch_string)
    slurm_output_path = match.group(1) if match else ""

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    launch_script_path = os.path.join(eval_launch_script_path, f"launch_script-{current_time}.slurm")

    # make sure the folder exists before write
    # Extract the folder path from launch_script_path
    folder_path = os.path.dirname(launch_script_path)

    # Check if the folder exists. If not, create it.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(launch_script_path, "w") as f:
        f.write(launch_string)

    logger.warning(f'Launching Slurm job {slurm_name} with launch script "{launch_script_path}"')

    # Preserve important environment variables
    env = {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
        'HOME': os.path.expanduser("~"),
    }

    try:
        # Use subprocess.run instead of check_output for better error handling
        result = subprocess.run(
            ["sbatch", launch_script_path],
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        output = result.stdout
        job_ids = output.split()[-1]
        output_log = (
            slurm_output_path.replace("%x", slurm_name).replace("%j", job_ids).replace("%n", "0").replace("%t", "0")
        )
        logger.warning(f'Slurm job launched successfully with id={job_ids}, logging outputs at "{output_log}"')
    except subprocess.CalledProcessError as e:
        logger.error(f"Error while launching Slurm job: {e}")
        logger.error(f"Command output: {e.output}")
        logger.error(f"Command stderr: {e.stderr}")
        job_ids = None
        output_log = None

    return job_ids, output_log
