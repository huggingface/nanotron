from dataclasses import dataclass, fields
from typing import Optional, Union

import torch
import yaml

from nanotron.config import (
    CheckpointsArgs,
    DataArgs,
    ExistingCheckpointInit,
    GeneralArgs,
    LoggingArgs,
    LRSchedulerArgs,
    PretrainDatasetsArgs,
    NanotronConfigs,
    OptimizerArgs,
    ParallelismArgs,
    ProfilerArgs,
    TokenizerArgs,
    TokensArgs,
    get_config_from_file,
)
from nanotron.config.lighteval_config import LightEvalConfig
from nanotron.config.utils_config import cast_str_to_torch_dtype, serialize


@dataclass
class MambaInit:
    # mamba_ssm.models.mixer_seq_simple._init_weights
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = (True,)
    n_residuals_per_layer: int = (1,)  # Change to 2 if we have MLP


@dataclass
class ModelArgs:
    """Arguments related to model architecture"""

    model_config: NanotronConfigs
    init_method: Union[MambaInit, ExistingCheckpointInit]
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
    data: Optional[DataArgs] = None
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


@dataclass
class MambaConfig:
    """Configuration for a Mamba model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    is_mamba_config: bool = True  # We use this help differentiate models in yaml/python conversion
    d_model: int = 2560
    num_hidden_layers: int = 64
    vocab_size: int = 50277
    ssm_cfg: Optional[dict] = None
    rms_norm: bool = True
    fused_add_norm: bool = True
    residual_in_fp32: bool = True
    pad_vocab_size_multiple: int = 8
    # ==== Custom ======
    dtype: str = "float32"
    rms_norm_eps: float = 1e-5
    pad_token_id: Optional[int] = None
