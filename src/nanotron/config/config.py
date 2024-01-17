import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Type

import dacite
import torch
import yaml
from dacite import from_dict
from yaml.loader import SafeLoader

from nanotron.config.models_config import ExistingCheckpointInit, NanotronConfigs, RandomInit
from nanotron.config.utils_config import (
    RecomputeGranularity,
    cast_str_to_pipeline_engine,
    cast_str_to_torch_dtype,
    serialize,
)
from nanotron.parallel.pipeline_parallelism.engine import (
    AllForwardAllBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.tensor_parallelism.nn import TensorParallelLinearMode
from nanotron.generate.sampler import SamplerType
from nanotron.logging import get_logger

logger = get_logger(__name__)

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
class DataArgs:
    """Arguments related to the data and data files processing"""

    dataset: PretrainDatasetsArgs
    seed: int
    num_loading_workers: Optional[int] = 1


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
        kill_switch_path: Path to the kill switch file
        ignore_sanity_checks: Whether to ignore sanity checks
    """

    project: str
    run: Optional[str] = None
    seed: Optional[int] = None
    step: Optional[int] = None
    consumed_train_samples: Optional[int] = None
    # If you want to signal the training script to stop, you just need to touch the following file
    # We force users to set one in order to programmatically be able to remove it.
    kill_switch_path: Optional[Path] = None
    # If you want to signal the training script to pause, you just need to add the following file
    benchmark_csv_path: Optional[Path] = None
    ignore_sanity_checks: bool = False

    def __post_init__(self):
        if self.seed is None:
            self.seed = 42
        if isinstance(self.kill_switch_path, str):
            self.kill_switch_path = Path(self.kill_switch_path)
        if self.benchmark_csv_path is not None:
            assert (
                os.environ.get("NANOTRON_BENCHMARK", None) is not None
            ), f"Please set NANOTRON_BENCHMARK to 1 when using benchmark_csv_path. Got {os.environ.get('NANOTRON_BENCHMARK', None)}"

        if self.run is None:
            self.run = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if os.environ.get("SLURM_JOB_ID", None) is not None:
                self.run += f"_{os.environ['SLURM_JOB_ID']}"
        else:
            self.run = self.run.replace("%d", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            if os.environ.get("SLURM_JOB_ID", None) is not None:
                self.run = self.run.replace("%j", os.environ["SLURM_JOB_ID"])


@dataclass
class ProfilerArgs:
    """Arguments related to profiling"""

    profiler_export_path: Optional[Path]


@dataclass
class ParallelismArgs:
    """Arguments related to TP/PP/DP

    Args:
        dp: Number of DP replicas
        pp: Number of PP stages
        tp: Number of TP replicas
        pp_engine: Pipeline engine to use between "1f1b" and "afab"
        tp_mode: TP mode to use between "all_reduce" and "reduce_scatter": all_reduce is normal, reduce_scatter activate sequence parallelism
        recompute_granularity: Recompute granularity to use between "full" and "selective"
        tp_linear_async_communication: Whether to use async communication in TP linear layers
    """

    dp: int
    pp: int
    tp: int
    pp_engine: Optional[PipelineEngine] = None
    tp_mode: Optional[TensorParallelLinearMode] = None
    recompute_granularity: Optional[RecomputeGranularity] = None
    tp_linear_async_communication: Optional[bool] = None

    def __post_init__(self):
        # Conservative defaults
        if self.pp_engine is None:
            self.pp_engine = AllForwardAllBackwardPipelineEngine()
        if self.tp_mode is None:
            self.tp_mode = TensorParallelLinearMode.ALL_REDUCE
        if self.tp_linear_async_communication is None:
            self.tp_linear_async_communication = False

        if isinstance(self.pp_engine, str):
            self.pp_engine = cast_str_to_pipeline_engine(self.pp_engine)
        if isinstance(self.tp_mode, str):
            self.tp_mode = TensorParallelLinearMode[self.tp_mode.upper()]
        if isinstance(self.recompute_granularity, str):
            self.recompute_granularity = RecomputeGranularity[self.recompute_granularity.upper()]


@dataclass
class ModelArgs:
    """Arguments related to model architecture"""

    model_config: NanotronConfigs
    init_method: Union[RandomInit, ExistingCheckpointInit]
    dtype: Optional[torch.dtype] = None
    make_vocab_size_divisible_by: int = 1
    ddp_bucket_cap_mb: int = 25

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16
        if isinstance(self.dtype, str):
            self.dtype = cast_str_to_torch_dtype(self.dtype)

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
    lr_decay_style: linear or cosine
    min_decay_lr: minimum learning rate after decay
    lr_decay_steps: optional number of steps to decay the learning rate otherwise will default to train_steps - lr_warmup_steps
    """

    learning_rate: float
    lr_warmup_steps: int = 0
    lr_warmup_style: str = None
    lr_decay_style: str = None
    min_decay_lr: float = None
    lr_decay_steps: Optional[int] = None

    def __post_init__(self):
        if self.lr_warmup_style not in ["linear", "constant"]:
            raise ValueError(
                f"lr_warmup_style should be a string selected in ['linear', 'constant'] and not {self.lr_warmup_style}"
            )
        if self.lr_warmup_style is None:
            self.lr_warmup_style = "linear"
        if self.lr_decay_style is None:
            self.lr_decay_style = "linear"
        if self.lr_decay_style not in ["linear", "cosine"]:
            raise ValueError(
                f"lr_decay_style should be a string selected in ['linear', 'cosine'] and not {self.lr_decay_style}"
            )
        if self.min_decay_lr is None:
            self.min_decay_lr = self.learning_rate


@dataclass
class OptimizerArgs:
    """Arguments related to the optimizer and learning rate"""

    zero_stage: int
    weight_decay: float
    clip_grad: Optional[float]

    accumulate_grad_in_fp32: bool

    adam_eps: float
    adam_beta1: float
    adam_beta2: float
    torch_adam_is_fused: bool
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


@dataclass
class Config:
    """Main configuration class"""

    general: GeneralArgs
    checkpoints: CheckpointsArgs
    parallelism: ParallelismArgs
    model: ModelArgs
    tokenizer: TokenizerArgs
    logging: LoggingArgs
    tokens: TokensArgs
    optimizer: OptimizerArgs
    data: DataArgs
    profiler: Optional[ProfilerArgs]

    def __post_init__(self):
        # Some final sanity checks across separate arguments sections:
        if self.profiler is not None and self.profiler.profiler_export_path is not None:
            assert self.tokens.train_steps < 10

        if self.optimizer.learning_rate_scheduler.lr_decay_steps is None:
            self.optimizer.learning_rate_scheduler.lr_decay_steps = (
                self.tokens.train_steps - self.optimizer.learning_rate_scheduler.lr_warmup_steps
            )

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


def get_config_from_file(config_path: str, config_class: Type[Config] = Config) -> Config:
    """Get a config objet from a file (python or YAML)

    Args:
        config_path: path to the config file
        config_type: if the file is a python file, type of the config object to get as a
            ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
            if None, will default to Config
    """
    # Open the file and load the file
    with open(config_path) as f:
        args = yaml.load(f, Loader=SafeLoader)

    # Make a nice dataclass from our yaml
    try:
        config = from_dict(
            data_class=config_class,
            data=args,
            config=dacite.Config(
                cast=[Path],
                type_hooks={
                    torch.dtype: cast_str_to_torch_dtype,
                    PipelineEngine: cast_str_to_pipeline_engine,
                    TensorParallelLinearMode: lambda x: TensorParallelLinearMode[x.upper()],
                    RecomputeGranularity: lambda x: RecomputeGranularity[x.upper()],
                    SamplerType: lambda x: SamplerType[x.upper()],
                },
                strict_unions_match=True,
                strict=True,
            ),
        )
    except Exception as e:
        raise ValueError(f"Error parsing config file {config_path}: {e}")

    return config
