import datetime
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Type, Union

import dacite
import torch
import yaml
from dacite import from_dict
from yaml.loader import SafeLoader

from nanotron.config.lighteval_config import LightEvalConfig
from nanotron.config.models_config import ExistingCheckpointInit, NanotronConfigs, RandomInit, SpectralMupInit
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.config.utils_config import (
    RecomputeGranularity,
    cast_str_to_pipeline_engine,
    cast_str_to_torch_dtype,
    serialize,
)
from nanotron.generation.sampler import SamplerType
from nanotron.logging import get_logger
from nanotron.parallel.pipeline_parallel.engine import PipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode

logger = get_logger(__name__)

DEFAULT_SEED = 42


@dataclass
class BenchArgs:
    model_name: str
    sequence_length: int
    micro_batch_size: int
    batch_accumulation_per_replica: int
    benchmark_csv_path: str


@dataclass
class LoggingArgs:
    """Arguments related to logging"""

    log_level: Optional[str] = None
    log_level_replica: Optional[str] = None
    iteration_step_info_interval: Optional[int] = 1

    def __post_init__(self):
        if self.log_level is None:
            self.log_level = "info"
        if self.log_level not in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "passive",
        ]:
            raise ValueError(
                f"log_level should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level}"
            )
        if self.log_level_replica is None:
            self.log_level_replica = "info"
        if self.log_level_replica not in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "passive",
        ]:
            raise ValueError(
                f"log_level_replica should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level_replica}"
            )


@dataclass
class PretrainDatasetsArgs:
    hf_dataset_or_datasets: Union[str, list, dict]
    hf_dataset_splits: Optional[Union[str, list]] = None
    hf_dataset_config_name: Optional[str] = None
    dataset_processing_num_proc_per_process: Optional[int] = 1
    dataset_overwrite_cache: Optional[bool] = False
    text_column_name: Optional[str] = None

    def __post_init__(self):
        if self.text_column_name is None:
            self.text_column_name = "text"
        if self.hf_dataset_splits is None:
            self.hf_dataset_splits = "train"


@dataclass
class NanosetDatasetsArgs:
    dataset_folder: Union[str, dict, List[str]]

    def __post_init__(self):
        if isinstance(self.dataset_folder, str):  # Case 1: 1 Dataset folder
            self.dataset_folder = [self.dataset_folder]
            self.dataset_weights = [1]
        elif isinstance(self.dataset_folder, List):  # Case 2: > 1 Dataset folder
            self.dataset_weights = None  # Set to None so we consume all the samples randomly
        elif isinstance(self.dataset_folder, dict):  # Case 3: dict with > 1 dataset_folder and weights
            tmp_dataset_folder = self.dataset_folder.copy()
            self.dataset_folder = list(tmp_dataset_folder.keys())
            self.dataset_weights = list(tmp_dataset_folder.values())


@dataclass
class DataArgs:
    """Arguments related to the data and data files processing"""

    dataset: Optional[Union[PretrainDatasetsArgs, NanosetDatasetsArgs]]
    seed: Optional[int]
    num_loading_workers: Optional[int] = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = DEFAULT_SEED


@dataclass
class DatasetStageArgs:
    """Arguments for loading dataset in different stages of the training process"""

    name: str
    start_training_step: int
    data: DataArgs

    def __post_init__(self):
        if self.start_training_step < 0:
            raise ValueError(f"training_steps should be a positive integer and not {self.start_training_step}")


@dataclass
class CheckpointsArgs:
    """Arguments related to checkpoints:
    checkpoints_path: where to save the checkpoints
    checkpoint_interval: how often to save the checkpoints
    resume_checkpoint_path: if you want to load from a specific checkpoint path

    """

    checkpoints_path: Path
    checkpoint_interval: int
    save_initial_state: Optional[bool] = False
    save_final_state: Optional[bool] = False
    resume_checkpoint_path: Optional[Path] = None
    checkpoints_path_is_shared_file_system: Optional[bool] = False

    def __post_init__(self):
        if isinstance(self.checkpoints_path, str):
            self.checkpoints_path = Path(self.checkpoints_path)
        if isinstance(self.resume_checkpoint_path, str):
            self.resume_checkpoint_path = Path(self.resume_checkpoint_path)


@dataclass
class GeneralArgs:
    """General training experiment arguments

    Args:
        project: Name of the project (a project gather several runs in common tensorboard/hub-folders)
        run: Name of the run
        step: Global step (updated when we save the checkpoint)
        consumed_train_samples: Number of samples consumed during training (should be actually just step*batch_size)
        ignore_sanity_checks: Whether to ignore sanity checks
    """

    project: str
    run: Optional[str] = None
    seed: Optional[int] = None
    step: Optional[int] = None
    consumed_train_samples: Optional[int] = None
    benchmark_csv_path: Optional[Path] = None
    ignore_sanity_checks: bool = True

    def __post_init__(self):
        if self.seed is None:
            self.seed = DEFAULT_SEED
        if self.benchmark_csv_path is not None:
            assert (
                os.environ.get("NANOTRON_BENCHMARK", None) is not None
            ), f"Please set NANOTRON_BENCHMARK to 1 when using benchmark_csv_path. Got {os.environ.get('NANOTRON_BENCHMARK', None)}"

        if self.run is None:
            self.run = "%date_%jobid"
        self.run.replace("%date", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run.replace("%jobid", os.environ.get("SLURM_JOB_ID", "local"))


@dataclass
class ProfilerArgs:
    """Arguments related to profiling"""

    profiler_export_path: Optional[Path]


@dataclass
class ModelArgs:
    """Arguments related to model architecture"""

    model_config: NanotronConfigs
    init_method: Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]
    dtype: Optional[torch.dtype] = None
    make_vocab_size_divisible_by: int = 1
    ddp_bucket_cap_mb: int = 25

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16
        if isinstance(self.dtype, str):
            self.dtype = cast_str_to_torch_dtype(self.dtype)

        self.model_config._is_using_mup = isinstance(self.init_method, SpectralMupInit)

        # if self.model_config.max_position_embeddings is None:
        #     self.model_config.max_position_embeddings = 0


@dataclass
class TokenizerArgs:
    """Arguments related to the tokenizer"""

    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tokenizer_max_length: Optional[int] = None


@dataclass
class TokensArgs:
    """Arguments related to the tokens, sequence, batch and steps of the training"""

    sequence_length: int
    train_steps: int
    micro_batch_size: int
    batch_accumulation_per_replica: int

    val_check_interval: Optional[int] = -1
    limit_val_batches: Optional[int] = 0
    limit_test_batches: Optional[int] = 0


@dataclass
class LRSchedulerArgs:
    """Arguments related to the learning rate scheduler

    lr_warmup_steps: number of steps to warmup the learning rate
    lr_warmup_style: linear or constant
    lr_decay_style: linear, cosine or 1-sqrt
    min_decay_lr: minimum learning rate after decay
    lr_decay_steps: optional number of steps to decay the learning rate otherwise will default to train_steps - lr_warmup_steps
    lr_decay_starting_step: optional number of steps to decay the learning rate otherwise will default to train_steps - lr_warmup_steps
    """

    learning_rate: float
    lr_warmup_steps: int = 0
    lr_warmup_style: str = None
    lr_decay_style: str = None
    lr_decay_steps: Optional[int] = None
    lr_decay_starting_step: Optional[int] = None
    min_decay_lr: float = None

    def __post_init__(self):
        if self.lr_warmup_style not in ["linear", "constant"]:
            raise ValueError(
                f"lr_warmup_style should be a string selected in ['linear', 'constant'] and not {self.lr_warmup_style}"
            )
        if self.lr_warmup_style is None:
            self.lr_warmup_style = "linear"
        if self.lr_decay_style is None:
            self.lr_decay_style = "linear"
        if self.lr_decay_style not in ["linear", "cosine", "1-sqrt"]:
            raise ValueError(
                f"lr_decay_style should be a string selected in ['linear', 'cosine', '1-sqrt'] and not {self.lr_decay_style}"
            )
        if self.min_decay_lr is None:
            self.min_decay_lr = self.learning_rate


@dataclass
class SGDOptimizerArgs:
    name: str = "sgd"


@dataclass
class AdamWOptimizerArgs:
    adam_eps: float
    adam_beta1: float
    adam_beta2: float
    torch_adam_is_fused: bool
    name: str = "adamW"


@dataclass
class OptimizerArgs:
    """Arguments related to the optimizer and learning rate"""

    optimizer_factory: Union[SGDOptimizerArgs, AdamWOptimizerArgs]
    zero_stage: int
    weight_decay: float
    clip_grad: Optional[float]
    accumulate_grad_in_fp32: bool
    learning_rate_scheduler: LRSchedulerArgs


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
            self.seed = DEFAULT_SEED


@dataclass
class Config:
    """Main configuration class"""

    general: GeneralArgs
    parallelism: ParallelismArgs
    model: ModelArgs
    tokenizer: TokenizerArgs
    checkpoints: Optional[CheckpointsArgs] = None
    logging: Optional[LoggingArgs] = None
    tokens: Optional[TokensArgs] = None
    optimizer: Optional[OptimizerArgs] = None
    data_stages: Optional[List[DatasetStageArgs]] = None
    profiler: Optional[ProfilerArgs] = None
    lighteval: Optional[LightEvalConfig] = None

    @classmethod
    def create_empty(cls):
        cls_fields = fields(cls)
        return cls(**{f.name: None for f in cls_fields})

    def __post_init__(self):
        # Some final sanity checks across separate arguments sections:
        if self.profiler is not None and self.profiler.profiler_export_path is not None:
            assert self.tokens.train_steps < 10

        if self.optimizer is not None and self.optimizer.learning_rate_scheduler.lr_decay_steps is None:
            self.optimizer.learning_rate_scheduler.lr_decay_steps = (
                self.tokens.train_steps - self.optimizer.learning_rate_scheduler.lr_warmup_steps
            )

        if self.data_stages is not None:
            self.data_stages = sorted(self.data_stages, key=lambda stage: stage.start_training_step)
            names = [stage.name for stage in self.data_stages]
            training_steps = [stage.start_training_step for stage in self.data_stages]
            assert any(
                stage.start_training_step == 1 for stage in self.data_stages
            ), "You must have a training stage starting at 1 in the config's data_stages"

            for stage in self.data_stages:
                if names.count(stage.name) > 1:
                    raise ValueError(f"Each stage should have unique names and not {names}")

                if training_steps.count(stage.start_training_step) > 1:
                    raise ValueError(
                        f"Each stage should have unique starting training step, please change the starting training step for stage {stage.name}"
                    )

            # NOTE: must order the stages by start_training_step from lowest to highest
            assert all(
                self.data_stages[i].start_training_step < self.data_stages[i + 1].start_training_step
                for i in range(len(self.data_stages) - 1)
            ), "The stages are not sorted by start_training_step in increasing order"

        # # if lighteval, we need tokenizer to be defined
        # if self.checkpoints.lighteval is not None:
        #     assert self.tokenizer.tokenizer_name_or_path is not None

    @property
    def global_batch_size(self):
        return self.tokens.micro_batch_size * self.tokens.batch_accumulation_per_replica * self.parallelism.dp

    def save_as_yaml(self, file_path: str):
        config_dict = serialize(self)
        file_path = str(file_path)
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f)

        # Sanity test config can be reloaded
        _ = get_config_from_file(file_path, config_class=self.__class__)

    def as_dict(self) -> dict:
        return serialize(self)


def get_config_from_dict(
    config_dict: dict, config_class: Type = Config, skip_unused_config_keys: bool = False, skip_null_keys: bool = False
):
    """Get a config object from a dictionary

    Args:
        args: dictionary of arguments
        config_class: type of the config object to get as a ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
        skip_unused_config_keys: whether to skip unused first-nesting-level keys in the config file (for config with additional sections)
        skip_null_keys: whether to skip keys with value None at first and second nesting level
    """
    if skip_unused_config_keys:
        logger.warning("skip_unused_config_keys set")
        config_dict = {
            field.name: config_dict[field.name] for field in fields(config_class) if field.name in config_dict
        }
    if skip_null_keys:
        logger.warning("Skip_null_keys set")
        config_dict = {
            k: {kk: vv for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else v
            for k, v in config_dict.items()
            if v is not None
        }
    return from_dict(
        data_class=config_class,
        data=config_dict,
        config=dacite.Config(
            cast=[Path],
            type_hooks={
                torch.dtype: cast_str_to_torch_dtype,
                PipelineEngine: cast_str_to_pipeline_engine,
                TensorParallelLinearMode: lambda x: TensorParallelLinearMode[x.upper()],
                RecomputeGranularity: lambda x: RecomputeGranularity[x.upper()],
                SamplerType: lambda x: SamplerType[x.upper()],
            },
            # strict_unions_match=True,
            strict=True,
        ),
    )


def get_config_from_file(
    config_path: str,
    config_class: Type = Config,
    model_config_class: Optional[Type] = None,
    skip_unused_config_keys: bool = False,
    skip_null_keys: bool = False,
) -> Config:
    """Get a config object from a file (python or YAML)

    Args:
        config_path: path to the config file
        config_type: if the file is a python file, type of the config object to get as a
            ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
            if None, will default to Config
        skip_unused_config_keys: whether to skip unused first-nesting-level keys in the config file (for config with additional sections)
        skip_null_keys: whether to skip keys with value None at first and second nesting level
    """
    # Open the file and load the file
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    config = get_config_from_dict(
        config_dict,
        config_class=config_class,
        skip_unused_config_keys=skip_unused_config_keys,
        skip_null_keys=skip_null_keys,
    )
    if model_config_class is not None:
        if not isinstance(config.model.model_config, (dict, model_config_class)):
            raise ValueError(
                f"model_config should be a dictionary or a {model_config_class} and not {config.model.model_config}"
            )
        config.model.model_config = model_config_class(**config.model.model_config)
    return config
