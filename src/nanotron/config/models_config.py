from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class RandomInit:
    std: float


@dataclass
class ExistingCheckpointInit:
    """This is used to initialize from an already existing model (without optimizer, lr_scheduler...)"""

    path: Path


@dataclass
class LlamaConfig:
    """Configuration for a LLAMA model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_llama_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000

    def __post_init__(self):
        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class FalconConfig:
    """Configuration for a Falcon model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    alibi: bool = False
    attention_dropout: float = 0.0
    bias: bool = False
    bos_token_id: int = 11
    eos_token_id: int = 11
    hidden_dropout: float = 0.0
    hidden_size: int = 4544
    initializer_range: float = 0.02
    is_falcon_config: bool = True  # We use this help differentiate models in yaml/python conversion
    layer_norm_epsilon: float = 1e-5
    multi_query: bool = True
    new_decoder_architecture: bool = False
    num_attention_heads: int = 71
    num_hidden_layers: int = 32
    num_kv_heads: Optional[int] = None
    pad_token_id: int = 11
    parallel_attn: bool = True
    use_cache: bool = True
    vocab_size: int = 65024

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary(self):
        return not self.alibi


@dataclass
class GPTBigCodeConfig:
    """Configuration for a GPTBigCode model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    is_gpt_bigcode_config: bool = False  # We use this help differentiate models in yaml/python conversion
    activation_function: str = "gelu_pytorch_tanh"
    attention_softmax_in_fp32: bool = True  # TODO: not used
    attn_pdrop: float = 0.1
    bos_token_id: int = 49152  # TODO: not used
    embd_pdrop: float = 0.1
    eos_token_id: int = 49152
    initializer_range: float = 0.02  # TODO: not used
    layer_norm_epsilon: float = 1e-05
    multi_query: bool = True
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 4096
    resid_pdrop: float = 0.1
    scale_attention_softmax_in_fp32: bool = True
    scale_attn_weights: bool = True
    vocab_size: int = 49280

    @property
    def n_embed(self):
        return self.hidden_size

    @property
    def n_head(self):
        return self.num_attention_heads

    @property
    def n_layer(self):
        return self.num_hidden_layers

    @property
    def n_positions(self):
        return self.max_position_embeddings

    @property
    def n_inner(self):
        return self.intermediate_size

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size


@dataclass
class Starcoder2Config(GPTBigCodeConfig):
    """Configuration for a Starcoder2 model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    is_starcoder2_config: bool = True  # We use this help differentiate models in yaml/python conversion
    sliding_window_size: Optional[int] = None
    use_rotary_embeddings: bool = True
    rope_theta: Optional[int] = 10000
    use_position_embeddings: bool = False  # TODO @nouamane this is not used
    global_attn_layers: List[int] = field(default_factory=list)

    # MQA
    multi_query: bool = False
    # GQA
    grouped_query: bool = False
    num_kv_heads: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        delattr(self, "is_gpt_bigcode_config")
        if self.global_attn_layers is None:
            self.global_attn_layers = []

        if self.grouped_query:
            assert self.num_kv_heads is not None, "num_kv_heads must be specified for grouped query"
            assert self.multi_query is False, "Cannot use both multi_query and grouped_query"

        if not self.multi_query and not self.grouped_query:
            self.multi_query = True


NanotronConfigs = Union[FalconConfig, LlamaConfig, GPTBigCodeConfig, Starcoder2Config]
