from dataclasses import dataclass
from typing import Optional, Union

import torch
from nanotron.config import Config, ExistingCheckpointInit, NanotronConfigs
from nanotron.config.utils_config import cast_str_to_torch_dtype


@dataclass
class MambaInit:
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = True
    n_residuals_per_layer: int = 1  # Change to 2 if we have MLP


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


@dataclass(kw_only=True)  # pylint: disable=unexpected-keyword-arg
class MambaConfig(Config):
    """Main configuration class"""

    model: ModelArgs


@dataclass
class MambaModelConfig:
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
