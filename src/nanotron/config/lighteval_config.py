from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from nanotron.config.config import ParallelismArgs, GenerationArgs


@dataclass
class LightEvalTasksArgs:
    """Arguments related to tasks for LightEval"""

    tasks: Optional[str] = None
    custom_tasks_file: Optional[Path] = None
    max_samples: Optional[int] = None
    num_fewshot_seeds: Optional[int] = None

    dataset_loading_processes: Optional[int] = 8
    multichoice_continuations_start_space: Optional[bool] = None
    no_multichoice_continuations_start_space: Optional[bool] = None

    def __post_init__(self):
        if isinstance(self.custom_tasks_file, str):
            self.custom_tasks_file = Path(self.custom_tasks_file)


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
class LightEvalSlurmArgs:
    """
    Args for SlurmExecutor.
    """

    logging_dir: str
    exclusive: bool = False
    ntasks_per_node: Optional[int] = 1
    job_name: Optional[str] = "eval"
    cpus_per_task: Optional[int] = None
    mem_per_cpu_gb: Optional[int] = None
    partition: Optional[str] = "production-cluster"
    qos: Optional[str] = "normal"
    sbatch_args: Optional[dict] = None
    time: Optional[str] = None
    nodes: Optional[int] = None
    lighteval_path: Optional[Path] = None
    env_setup_command: Optional[str] = None
    tokenizer: Optional[str] = None
    s5cmd_path: Optional[str] = None
    s5cmd_numworkers: Optional[int] = None
    s5cmd_concurrency: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.lighteval_path, str):
            self.lighteval_path = Path(self.lighteval_path)


@dataclass
class LightEvalConfig:
    """Arguments related to running LightEval on checkpoints.

    All is optional because you can also use this class to later supply arguments to override
    the saved config when running LightEval after training.
    """

    checkpoints_path: Optional[str] = None
    parallelism: Optional[ParallelismArgs] = None
    batch_size: Optional[int] = None
    generation: Optional[Union["GenerationArgs", Dict[str, "GenerationArgs"]]] = None
    tasks: Optional[LightEvalTasksArgs] = None
    logging: Optional[LightEvalLoggingArgs] = None
    slurm: Optional[LightEvalSlurmArgs] = None

    def __post_init__(self):
        # Automatically setup our node count
        if self.slurm.nodes is None:
            self.slurm.nodes = self.parallelism.dp * self.parallelism.pp * self.parallelism.tp // 8  # 8 GPUs per node
