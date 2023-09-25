from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

import dacite
import torch
import yaml
from dacite import from_dict
from datasets.download.streaming_download_manager import xPath
from yaml.loader import SafeLoader

from nanotron.core.parallelism.pipeline_parallelism.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.core.parallelism.tensor_parallelism.nn import TensorParallelLinearMode


class RecomputeGranularity(Enum):
    SELECTIVE = auto()
    FULL = auto()


@dataclass
class GeneralArgs:
    """General training experiment arguments"""

    name: str
    # If you want to signal the training script to stop, you just need to touch the following file
    # We force users to set one in order to programmatically be able to remove it.
    kill_switch_path: xPath
    ignore_sanity_checks: bool = False


@dataclass
class ProfileArgs:
    """Arguments related to profiling"""

    profiler_export_path: Optional[str]


@dataclass
class CheckpointsArgs:
    """Arguments related to checkpoints"""

    checkpoints_path: xPath
    checkpoint_interval: int
    load_from_specific_checkpoint: Optional[int]


@dataclass
class ParallelismArgs:
    """Arguments related to TP/PP/DP"""

    dp: int
    pp: int
    tp: int
    pp_engine: PipelineEngine
    tp_mode: TensorParallelLinearMode
    recompute_granularity: Optional[RecomputeGranularity]
    tp_linear_async_communication: bool


@dataclass
class PPOParallelismArgs(ParallelismArgs):
    """Arguments related to TP/PP/DP"""

    ref_model_pp_size: int  # For PPO, number of PP stages in the reference model


@dataclass
class RandomInit:
    std: float


@dataclass
class ExistingCheckpointInit:
    """This is used to initialize from an already existing model (without optimizer, lr_scheduler...)"""

    path: xPath


@dataclass
class RemoteCodeArgs:
    """Arguments related to model architecture"""

    trust_remote_code: bool


@dataclass
class ModelArgs:
    """Arguments related to model architecture"""

    hf_model_name: str
    make_vocab_size_divisible_by: int
    dtype: torch.dtype
    init_method: Union[RandomInit, ExistingCheckpointInit]
    seed: Optional[int]
    remote_code: Optional[RemoteCodeArgs]


@dataclass
class HubLoggerConfig:
    """Arguments related to the HF Tensorboard logger"""

    tensorboard_dir: xPath
    repo_id: str
    push_to_hub_interval: int
    repo_public: bool = False


@dataclass
class TensorboardLoggerConfig:
    """Arguments related to the local Tensorboard logger"""

    tensorboard_dir: xPath
    flush_secs: int = 120


@dataclass
class LoggingArgs:
    """Arguments related to logging"""

    log_level: str
    log_level_replica: str
    iteration_step_info_interval: int
    tensorboard_logger: Optional[Union[HubLoggerConfig, TensorboardLoggerConfig]]

    def __post_init__(self):
        if self.log_level not in ["debug", "info", "warning", "error", "critical", "passive"]:
            raise ValueError(
                f"log_level should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level}"
            )
        if self.log_level_replica not in ["debug", "info", "warning", "error", "critical", "passive"]:
            raise ValueError(
                f"log_level_replica should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level_replica}"
            )


@dataclass
class TokensArgs:
    """Arguments related to the tokens, sequence, batch and steps of the training"""

    sequence_length: int
    train_steps: int
    micro_batch_size: int
    batch_accumulation_per_replica: int

    val_check_interval: int
    limit_val_batches: int
    limit_test_batches: int = 0


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

    learning_rate: float


@dataclass
class LRSchedulerArgs:
    lr_warmup_steps: int
    lr_warmup_style: str
    lr_decay_style: str
    lr_decay_steps: Optional[int]
    min_decay_lr: float

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
class PretrainNemoArgs:
    data_prefix: Union[list, dict]
    index_mapping_dir: Optional[
        str
    ]  # path to save index mapping .npy files, by default will save in the same location as data_prefix
    splits_string: str
    skip_warmup: bool
    dataloader_type: str
    validation_drop_last: bool  # Set to false if the last partial validation samples is to be consumed
    eod_mask_loss: bool  # Mask loss for the end of document tokens
    no_seqlen_plus_one_input_tokens: bool  # Set to True to disable fetching (sequence length + 1) input tokens, instead get (sequence length) input tokens and mask the last token
    pad_samples_to_global_batch_size: bool  # Set to True if you want to pad the last partial batch with -1's to equal global batch size

    def __post_init__(self):
        # TODO @thomasw21: Should probably be an enum
        if self.dataloader_type not in ["single", "cyclic"]:
            raise ValueError(
                f"dataloader_type should be a string selected in ['single', 'cyclic'] and not {self.dataloader_type}"
            )

        if self.eod_mask_loss:
            raise NotImplementedError("`eod_mask_loss` support is not implemented yet")


@dataclass
class PretrainDatasetsArgs:
    hf_dataset_mixer: Union[str, list, dict]
    hf_dataset_config_name: Optional[str]
    hf_dataset_splits: Union[str, list]
    dataset_processing_num_proc_per_process: int
    dataset_overwrite_cache: Optional[bool]
    text_column_name: Optional[str]


@dataclass
class DPOPretrainDatasetsArgs:
    hf_dataset_name: str
    hf_dataset_split: str

    # TODO @nouamane: this gives unclear error: https://github.com/huggingface/brrr/issues/515
    # def __post_init__(self):
    #     if self.hf_dataset_name not in ["Anthropic/hh-rlhf"]:
    #         raise ValueError(
    #             f"hf_dataset_name={self.hf_dataset_name} is not supported yet. Only Anthropic/hh-rlhf is supported."
    #         )


@dataclass
class DataArgs:
    """Arguments related to the data and data files processing"""

    # TODO @thomasw21: Would have been great to have sealed class of something (kotlin concept)
    seed: Optional[int]
    num_loading_workers: int
    dataset: Optional[Union[PretrainNemoArgs, PretrainDatasetsArgs, DPOPretrainDatasetsArgs]]


@dataclass
class PPOArgs:
    """Arguments related to PPO training"""

    # KL Controller
    init_kl_coef: float
    target: float
    horizon: float

    # PPO Loss
    gamma: float
    lam: float
    cliprange_value: float
    cliprange: float
    vf_coef: float

    # Other
    ppo_epochs: int
    queries_length: int
    num_layers_unfrozen: int  # Number of decoder layers to unfreeze (-1 means all layers are unfrozen)
    n_mini_batches_per_batch: int  # Number of optimization steps in 1 ppo_epoch
    generation_batch_size: Optional[int] = 32  # Micro batch size for generation


@dataclass
class DPOArgs:
    """Arguments related to DPO training"""

    # Dataloader
    max_prompt_length: int  # max length of each sample's prompt
    truncation_mode: str  # The truncation mode to use when truncating the prompt. Can be one of ['keep_end', 'keep_start']

    # DPO Loss
    beta: float  # the temperature parameter for the DPO loss, typically something in the range of `0.1` to `0.5`. We ignore the reference model as `beta` -> 0

    # Other
    num_layers_unfrozen: int  # Number of decoder layers to unfreeze (-1 means all layers are unfrozen)
    label_pad_token_id: Optional[int] = -100  # The token id to use for padding the labels


@dataclass
class RewardArgs:
    """Arguments related to Reward training"""

    # Other
    num_layers_unfrozen: int  # Number of decoder layers to unfreeze (-1 means all layers are unfrozen)


@dataclass
class Config:
    """Main configuration class"""

    general: GeneralArgs
    profile: Optional[ProfileArgs]
    checkpoints: CheckpointsArgs
    parallelism: Union[ParallelismArgs, PPOParallelismArgs]
    model: ModelArgs
    logging: LoggingArgs
    tokens: TokensArgs
    optimizer: OptimizerArgs
    learning_rate_scheduler: LRSchedulerArgs
    data: DataArgs
    ppo: Optional[PPOArgs]  # For PPO training
    dpo: Optional[DPOArgs]  # For DPO training
    reward: Optional[RewardArgs]  # For Reward training


str_to_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def cast_str_to_torch_dtype(str_dtype: str):
    if str_dtype in str_to_dtype:
        return str_to_dtype[str_dtype]
    else:
        raise ValueError(f"dtype should be a string selected in {str_to_dtype.keys()} and not {str_dtype}")


def cast_str_to_pipeline_engine(str_pp_engine: str) -> PipelineEngine:
    if str_pp_engine == "afab":
        return AllForwardAllBackwardPipelineEngine()
    elif str_pp_engine == "1f1b":
        return OneForwardOneBackwardPipelineEngine()
    else:
        raise ValueError(f"pp_engine should be a string selected in ['afab', '1f1b'] and not {str_pp_engine}")


def get_args_from_path(config_file: str) -> Config:
    # Open the file and load the file
    with open(config_file) as f:
        args = yaml.load(f, Loader=SafeLoader)

    # Make a nice dataclass from our yaml
    try:
        config = from_dict(
            data_class=Config,
            data=args,
            config=dacite.Config(
                cast=[xPath],
                type_hooks={
                    torch.dtype: cast_str_to_torch_dtype,
                    PipelineEngine: cast_str_to_pipeline_engine,
                    TensorParallelLinearMode: lambda x: TensorParallelLinearMode[x.upper()],
                    RecomputeGranularity: lambda x: RecomputeGranularity[x.upper()],
                },
                strict_unions_match=True,
                strict=True,
            ),
        )
    except Exception as e:
        raise ValueError(f"Error parsing config file {config_file}: {e}")

    # Some final sanity checks across separate arguments sections:
    if config.profile is not None and config.profile.profiler_export_path is not None:
        assert config.tokens.train_steps < 10

    if config.learning_rate_scheduler.lr_decay_steps is None:
        config.learning_rate_scheduler.lr_decay_steps = (
            config.tokens.train_steps - config.learning_rate_scheduler.lr_warmup_steps
        )

    return config
