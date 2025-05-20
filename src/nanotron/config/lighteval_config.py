from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
import os
from dataclasses import field

from nanotron.generation.sampler import SamplerType
from nanotron.logging import get_logger
from nanotron.config.parallelism_config import ParallelismArgs
logger = get_logger(__name__)

DEFAULT_GENERATION_SEED = 42


@dataclass
class GenerationArgs:
    sampler: Optional[Union[str, SamplerType]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n_samples: Optional[int] = None
    eos: Optional[str] = None
    seed: Optional[int] = None
    use_cache: Optional[bool] = False

    def __post_init__(self):
        if isinstance(self.sampler, str):
            self.sampler = SamplerType[self.sampler.upper()]
        if self.seed is None:
            self.seed = DEFAULT_GENERATION_SEED


@dataclass
class LightEvalLoggingArgs:
    """Arguments related to logging for LightEval"""

    local_output_path: Optional[Path] = None
    push_results_to_hub: Optional[bool] = None
    push_details_to_hub: Optional[bool] = None
    push_results_to_tensorboard: Optional[bool] = None
    hub_repo_results: Optional[str] = None
    hub_repo_details: Optional[str] = None
    hub_repo_tensorboard: Optional[str] = None
    tensorboard_metric_prefix: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.local_output_path, str):
            self.local_output_path = Path(self.local_output_path)


@dataclass
class LightEvalTasksArgs:
    """Arguments related to tasks for LightEval"""

    tasks: str
    custom_tasks: Optional[str] = None
    max_samples: Optional[int] = None
    num_fewshot_seeds: Optional[int] = None

    dataset_loading_processes: Optional[int] = 8
    multichoice_continuations_start_space: Optional[bool] = None
    no_multichoice_continuations_start_space: Optional[bool] = None


@dataclass
class LightEvalWandbLoggerConfig:
    """Arguments related to the local Wandb logger"""

    wandb_project: str = ""
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def __post_init__(self):
        assert self.wandb_project != "", "Please specify a wandb_project"


@dataclass
class LightEvalSlurm:
    """Arguments related to SLURM configuration for LightEval"""

    gpus_per_node: int = 8
    partition: str = "hopper-prod"
    hf_cache: Optional[str] = None
    cpus_per_task: int = 88
    qos: str = "low"
    time: str = "24:00:00"
    reservation: Optional[str] = None

    def __post_init__(self):
        self.hf_cache = str(Path(self.hf_cache).expanduser())


@dataclass
class LightEvalConfig:
    """Arguments related to running LightEval on checkpoints.

    All is optional because you can also use this class to later supply arguments to override
    the saved config when running LightEval after training.
    """

    logs_path: Path
    nanotron_path: str
    output_dir: str
    local_checkpoint_dir: Path = Path(
        "/scratch"
    )  # Base directory for temporary checkpoint storage, will store under {local_checkpoint_dir}/{run_name}/{step}
    slurm: LightEvalSlurm = field(default_factory=LightEvalSlurm)
    tasks: LightEvalTasksArgs = field(default_factory=LightEvalTasksArgs)
    wandb: Optional[LightEvalWandbLoggerConfig] = None
    s3_save_path: Optional[str] = None  # should not be dependent of the run_name
    upload_to_wandb: Optional[bool] = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    parallelism: Optional[ParallelismArgs] = field(default_factory=lambda: ParallelismArgs(dp=1, pp=1, tp=1, tp_linear_async_communication=True))
    batch_size: Optional[int] = None
    eval_interval: Optional[
        int
    ] = None  # Must be multiple of checkpoint_interval. If None, eval will be done after each checkpoint upload to s3
    eval_interval_file: Optional[
        Path
    ] = None  # If specified, eval_interval will be read from this file upon the next evaluation.

    def __post_init__(self):
        self.local_checkpoint_dir = str(Path(self.local_checkpoint_dir).expanduser())
        if self.upload_to_wandb:
            assert self.s3_save_path is not None, " We should have a s3_save_path if we want to upload to wandb" # todo: add the option to read from local folder i guess
            assert self.wandb_project is not None, "wandb_project must be specified if upload_to_wandb is True"
            assert self.wandb_entity is not None, "wandb_entity must be specified if upload_to_wandb is True"
        if self.eval_interval_file is not None and Path(self.eval_interval_file).exists():
            logger.warning(
                f"Eval interval file {self.eval_interval_file} exists. `eval_interval` will be replaced by the value in the file upon the next evaluation. You should probably delete this file if that's not what you want."
            )
