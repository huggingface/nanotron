from dataclasses import dataclass
from typing import Dict, Optional, Union

from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.generation.sampler import SamplerType
from nanotron.logging import get_logger

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

    output_dir: Optional[str] = None
    save_details: bool = True
    push_to_hub: bool = False
    push_to_tensorboard: bool = False
    public_run: bool = False
    results_org: str | None = None
    tensorboard_metric_prefix: str = "eval"


@dataclass
class LightEvalTasksArgs:
    """Arguments related to tasks for LightEval"""

    tasks: str
    custom_tasks: Optional[str] = None
    max_samples: Optional[int] = None
    num_fewshot_seeds: Optional[int] = None

    dataset_loading_processes: int = 8
    multichoice_continuations_start_space: Optional[bool] = None
    pair_wise_tokenization: bool = False


@dataclass
class LightEvalConfig:
    """Arguments related to running LightEval on checkpoints.

    All is optional because you can also use this class to later supply arguments to override
    the saved config when running LightEval after training.
    """

    logging: LightEvalLoggingArgs
    tasks: LightEvalTasksArgs
    parallelism: ParallelismArgs
    batch_size: int = 0
    generation: Optional[Union[GenerationArgs, Dict[str, GenerationArgs]]] = None
