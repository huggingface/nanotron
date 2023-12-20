import datetime
import importlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar, Union

import dacite
import torch
import yaml
from dacite import from_dict
from pathlib import Path
from transformers import AutoConfig
from yaml.loader import SafeLoader

from nanotron.logging import get_logger
from nanotron.core.parallel.pipeline_parallelism.engine import (
    AllForwardAllBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.config.checkpoints_config import CheckpointsArgs
from nanotron.config.data_config import DataArgs
from nanotron.config.logging_config import LoggingArgs
from nanotron.config.lighteval_config import LightEvalConfig, LightEvalSlurmArgs
from nanotron.config.models_config import NanotronConfigs
from nanotron.config.utils_config import cast_str_to_pipeline_engine, cast_str_to_torch_dtype, RecomputeGranularity, serialize
from nanotron.core.parallel.tensor_parallelism.nn import TensorParallelLinearMode
from nanotron.generate.sampler import SamplerType

logger = get_logger(__name__)



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
    step: Optional[int] = None
    consumed_train_samples: Optional[int] = None
    # If you want to signal the training script to stop, you just need to touch the following file
    # We force users to set one in order to programmatically be able to remove it.
    kill_switch_path: Optional[Path] = None
    # If you want to signal the training script to pause, you just need to add the following file
    benchmark_csv_path: Optional[Path] = None
    ignore_sanity_checks: bool = False

    def __post_init__(self):
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
class RandomInit:
    std: float


@dataclass
class ExistingCheckpointInit:
    """This is used to initialize from an already existing model (without optimizer, lr_scheduler...)"""

    path: Path

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)


@dataclass
class ModelArgs:
    """Arguments related to model architecture"""

    dtype: torch.dtype
    init_method: Union[RandomInit, ExistingCheckpointInit]
    seed: Optional[int]
    model_config: Optional[NanotronConfigs] = None
    make_vocab_size_divisible_by: int
    ddp_bucket_cap_mb: int = 25

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = cast_str_to_torch_dtype(self.dtype)

        if self.model_config is not None:
            if self.model_config.max_position_embeddings is None:
                self.model_config.max_position_embeddings = 0


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
    lr_warmup_steps: int
    lr_warmup_style: str
    lr_decay_style: str
    min_decay_lr: float
    lr_decay_steps: Optional[int] = None

    def __post_init__(self):
        if self.lr_warmup_style not in ["linear", "constant"]:
            raise ValueError(
                f"lr_warmup_style should be a string selected in ['linear', 'constant'] and not {self.lr_warmup_style}"
            )
        if self.lr_decay_style not in ["linear", "cosine"]:
            raise ValueError(
                f"lr_decay_style should be a string selected in ['linear', 'cosine'] and not {self.lr_decay_style}"
            )


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

    def __post_init__(self):
        if isinstance(self.sampler, str):
            self.sampler = SamplerType[self.sampler.upper()]


@dataclass
class Config:
    """Main configuration class"""

    general: GeneralArgs
    profiler: Optional[ProfilerArgs]
    checkpoints: CheckpointsArgs
    parallelism: ParallelismArgs
    model: ModelArgs 
    tokenizer: TokenizerArgs
    logging: LoggingArgs
    tokens: TokensArgs
    optimizer: OptimizerArgs
    data: DataArgs

    def __post_init__(self):
        # Some final sanity checks across separate arguments sections:
        if self.profiler is not None and self.profiler.profiler_export_path is not None:
            assert self.tokens.train_steps < 10

        if self.learning_rate_scheduler.lr_decay_steps is None:
            self.learning_rate_scheduler.lr_decay_steps = (
                self.tokens.train_steps - self.learning_rate_scheduler.lr_warmup_steps
            )

        # if lighteval, we need tokenizer to be defined
        if self.checkpoints.lighteval is not None:
            assert self.model.tokenizer_name_or_path is not None

    @property
    def global_batch_size(self):
        return self.tokens.micro_batch_size * self.tokens.batch_accumulation_per_replica * self.parallelism.dp

    def save_as_yaml(self, file_path: str):
        config_dict = serialize(self)
        file_path = str(file_path)
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f)

        # Sanity test config can be reloaded
        _ = get_config_from_file(file_path)

    def as_dict(self) -> dict:
        return serialize(self)



class ConfigTypes(Enum):
    """Enum class for the different types of config files
    Name is the name of the class
    Value is the name of the object
    """

    LightEvalConfig = "lighteval"
    Config = "config"
    LightEvalSlurmArgs = "slurm"


# All the config types in ConfigTypes as a type
ConfigTypesClasses = TypeVar("ConfigTypesClasses", LightEvalConfig, Config, LightEvalSlurmArgs)


def get_config_from_file(config_path: str, config_type: Union[ConfigTypes, str, None] = None) -> ConfigTypesClasses:
    """Get a config objet from a file (python or YAML)

    Args:
        config_path: path to the config file
        config_type: if the file is a python file, type of the config object to get as a
            ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
            if None, will default to Config
    """
    if config_type is None:
        config_type = ConfigTypes.Config
    if isinstance(config_type, str):
        config_type = ConfigTypes(config_type)

    # Open the file and load the file
    with open(config_path) as f:
        args = yaml.load(f, Loader=SafeLoader)

    # Make a nice dataclass from our yaml
    try:
        config = from_dict(
            data_class=globals()[config_type.name],
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


@dataclass
class AllTrainerConfigs:
    config: Config
    model_config: Union[NanotronConfigs, AutoConfig]


def get_all_trainer_configs(config_or_config_file: Union[Config, str]) -> AllTrainerConfigs:
    """Get the config and the config file path from either a config object or a config file path."""
    if isinstance(config_or_config_file, str):
        config: Config = get_config_from_file(config_or_config_file)
    else:
        config = config_or_config_file

    if config.model.hf_model_name:
        trust_remote_code = (
            config.model.remote_code.trust_remote_code if hasattr(config.model, "remote_code") else None
        )
        model_config = AutoConfig.from_pretrained(config.model.hf_model_name, trust_remote_code=trust_remote_code)
    else:
        assert config.model.model_config is not None, "Either model.hf_model_name or model.model_config must be set"
        model_config = config.model.model_config

    return AllTrainerConfigs(config=config, model_config=model_config)

