from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from nanotron.nn.attention import ALL_ATTENTION_FUNCTIONS, AttentionImplementation

# The default attention implementation to use
DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"


@dataclass
class RandomInit:
    std: float


@dataclass
class SpectralMupInit:
    """This is used to initialize the model with spectral mup. Set it to True to use it."""

    use_mup: bool

    def __post_init__(self):
        assert self.use_mup, "Remove `use_mup` if you don't want to use it"


@dataclass
class ExistingCheckpointInit:
    """This is used to initialize from an already existing model (without optimizer, lr_scheduler...)"""

    path: Path


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layers"""

    num_experts: int = 8  # Total number of experts
    top_k: int = 2  # Number of experts to route each token to
    layers: List[int] = field(
        default_factory=lambda: [-1]
    )  # Indices of layers that use MoE. -1 means all layers. Default is all layers
    enable_shared_expert: bool = False  # Whether to use a shared expert alongside specialized experts
    token_dispatcher_type: str = "alltoall"  # Communication pattern for MoE ("alltoall" or "allgather")
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    output_router_logits: bool = False

    def __post_init__(self):
        # Validate the configuration
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot be greater than num_experts ({self.num_experts})")

        if self.token_dispatcher_type not in ["alltoall", "allgather"]:
            raise ValueError(
                f"token_dispatcher_type must be one of ['alltoall', 'allgather'], got {self.token_dispatcher_type}"
            )


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
    attention_bias: bool = False
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = (
        False  # The default value has been True, but for loading Llama3 checkpoints you have to set it to False
    )
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000
    _attn_implementation: Optional[AttentionImplementation] = DEFAULT_ATTENTION_IMPLEMENTATION

    z_loss_enabled: bool = False  # Z-loss regularization https://www.jmlr.org/papers/volume24/22-1144/22-1144.pdf
    z_loss_coefficient: float = 0.0001  # Default from the paper (10^-4)

    def __post_init__(self):
        # NOTE: user don't set self._init_method, ModelArgs will set it
        # then we only pass LlamaConfig around
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Validate that the attention implementation is valid
        if self._attn_implementation is not None:
            assert (
                self._attn_implementation in ALL_ATTENTION_FUNCTIONS
            ), f"Invalid attention implementation: {self._attn_implementation}. Available options are: {ALL_ATTENTION_FUNCTIONS.keys()}"

    # TODO: assert top_k larger than 1 if moe is enabled
    # TODO: do a base config, so don't duplicate this part
    @property
    def is_using_mup(self) -> bool:
        return self._is_using_mup


@dataclass
class Qwen2Config:
    """Configuration for a QWEN2 model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_qwen2_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000
    _attn_implementation: Optional[AttentionImplementation] = DEFAULT_ATTENTION_IMPLEMENTATION
    flex_attention_mask: Optional[str] = None
    attention_bias: bool = False
    sliding_window_size: Optional[int] = None
    z_loss_enabled: bool = False  # Z-loss regularization https://www.jmlr.org/papers/volume24/22-1144/22-1144.pdf
    z_loss_coefficient: float = 0.0001  # Default from the paper (10^-4)
    _fused_rotary_emb: bool = True
    _fused_rms_norm: bool = True
    _use_qkv_packed: bool = True

    # MoE configuration
    moe_config: Optional[MoEConfig] = None

    def __post_init__(self):
        # NOTE: user don't set self._init_method, ModelArgs will set it
        # then we only pass LlamaConfig around
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # By default i want all layers to be MoE layers
        if self.moe_config and self.moe_config.layers == [-1]:
            self.moe_config.layers = list(range(self.num_hidden_layers))

        # Validate that the attention implementation is valid
        if self._attn_implementation is not None:
            assert (
                self._attn_implementation in ALL_ATTENTION_FUNCTIONS
            ), f"Invalid attention implementation: {self._attn_implementation}. Available options are: {ALL_ATTENTION_FUNCTIONS.keys()}"

        if self.sliding_window_size is not None:
            assert self._attn_implementation in [
                "flex_attention",
                "flash_attention_2",
            ], "Sliding window is only supported for Flex Attention and Flash Attention 2"
        if self.flex_attention_mask is not None:
            assert (
                self._attn_implementation == "flex_attention"
            ), "Flex attention mask is only supported for flex attention"
            assert self.flex_attention_mask in [
                "sliding_window",
                "document",
                "sliding_window_document",
            ], "Flex attention mask must be one of ['sliding_window', 'document', 'sliding_window_document']"

    @property
    def is_using_mup(self) -> bool:
        return self._is_using_mup

    @property
    def is_moe_model(self) -> bool:
        """Returns True if the model uses MoE layers"""
        return self.moe_config is not None and len(self.moe_config.layers) > 0


@dataclass
class Starcoder2Config:
    """Configuration for a Starcoder2 model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    activation_function: str = "gelu_pytorch_tanh"
    attention_softmax_in_fp32: bool = True  # TODO: not used
    attn_pdrop: float = 0.1
    bos_token_id: int = 49152  # TODO: not used
    embd_pdrop: float = 0.1
    eos_token_id: int = 49152
    global_attn_layers: List[int] = field(default_factory=list)
    grouped_query: bool = False  # GQA
    hidden_size: int = 2048
    initializer_range: float = 0.02  # TODO: not used
    intermediate_size: Optional[int] = None
    is_starcoder2_config: bool = True  # We use this help differentiate models in yaml/python conversion
    layer_norm_epsilon: float = 1e-05
    max_position_embeddings: int = 4096
    multi_query: bool = False  # MQA
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    num_kv_heads: Optional[int] = None
    resid_pdrop: float = 0.1
    rope_theta: Optional[int] = 10000
    scale_attention_softmax_in_fp32: bool = True
    scale_attn_weights: bool = True
    sliding_window_size: Optional[int] = None
    use_position_embeddings: bool = False  # TODO @nouamane this is not used
    use_rotary_embeddings: bool = True
    vocab_size: int = 49280
    _attn_implementation: Optional[AttentionImplementation] = DEFAULT_ATTENTION_IMPLEMENTATION

    def __post_init__(self):
        if self.global_attn_layers is None:
            self.global_attn_layers = []

        if self.grouped_query:
            assert self.num_kv_heads is not None, "num_kv_heads must be specified for grouped query"
            assert self.multi_query is False, "Cannot use both multi_query and grouped_query"

        if not self.multi_query and not self.grouped_query:
            self.multi_query = True

        # Validate that the attention implementation is valid
        if self._attn_implementation is not None:
            assert (
                self._attn_implementation in ALL_ATTENTION_FUNCTIONS
            ), f"Invalid attention implementation: {self._attn_implementation}. Available options are: {ALL_ATTENTION_FUNCTIONS.keys()}"

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


@dataclass
class VisionConfig:
    """Configuration for the Llama4 Vision Model component"""

    attention_dropout: float = 0.0
    hidden_act: str = "gelu"
    hidden_size: int = 1408
    image_size: int = 336
    initializer_range: float = 0.02
    intermediate_size: int = 5632
    model_type: str = "llama4_vision_model"
    multi_modal_projector_bias: bool = False
    norm_eps: float = 1e-05
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 34
    patch_size: int = 14
    pixel_shuffle_ratio: float = 0.5
    projector_dropout: float = 0.0
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096
    rope_theta: float = 10000
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "default"
    vision_output_dim: int = 4096


@dataclass
class TextConfig:
    """Configuration for the Llama4 Text Model component"""

    attention_bias: bool = False
    attention_chunk_size: int = 8192
    attention_dropout: float = 0.0
    bos_token_id: int = 200000
    eos_token_id: List[int] = field(default_factory=lambda: [200001, 200002, 200003, 200007, 200008])
    hidden_act: str = "silu"
    hidden_size: int = 5120
    initializer_range: float = 0.02
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    max_position_embeddings: int = 131072
    model_type: str = "llama4_text"
    num_attention_heads: int = 40
    num_hidden_layers: int = 48
    num_key_value_heads: int = 8
    pad_token_id: int = 200008
    rms_norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = field(
        default_factory=lambda: {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    )
    rope_theta: float = 500000.0
    rope_interleaved: bool = False
    tie_word_embeddings: bool = False
    use_cache: bool = True
    use_qk_norm: bool = True
    no_rope_layers: List[int] = field(default_factory=list)
    vocab_size: int = 202048
    _attn_implementation: Optional[AttentionImplementation] = DEFAULT_ATTENTION_IMPLEMENTATION
    head_dim: int = 128
    sliding_window_size: Optional[int] = None

    # MoE specific parameters
    for_llm_compressor: bool = False
    num_local_experts: int = 16
    num_experts_per_tok: int = 1
    interleave_moe_layer_step: int = 1
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    z_loss_enabled: bool = False
    z_loss_coefficient: float = 0.0001


@dataclass
class Llama4Config:
    """Configuration for a LLAMA4 model with multimodal support

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    # Configuration components
    text_config: TextConfig
    vision_config: VisionConfig

    is_llama4_config: bool  # We use this help differentiate models in yaml/python conversion
    model_type: str = "llama4"

    # Multimodal tokens
    boi_token_index: int = 200080  # Beginning of image token
    eoi_token_index: int = 200081  # End of image token
    image_token_index: int = 200092

    # MoE configuration
    # moe_config: Optional[MoEConfig] = None

    def __post_init__(self):
        self._is_using_mup: bool = False

        # Setup MoE configuration if not provided
        # if self.moe_config is None and self.text_config.num_local_experts > 1:
        #     self.moe_config = MoEConfig(
        #         num_experts=self.text_config.num_local_experts,
        #         top_k=self.text_config.num_experts_per_tok,
        #         layers=list(range(0, self.text_config.num_hidden_layers, self.text_config.interleave_moe_layer_step)),
        #         output_router_logits=self.text_config.output_router_logits,
        #         router_aux_loss_coef=self.text_config.router_aux_loss_coef,
        #         router_jitter_noise=self.text_config.router_jitter_noise,
        #     )

        # Validate that the attention implementation is valid
        if self.text_config._attn_implementation is not None:
            assert (
                self.text_config._attn_implementation in ALL_ATTENTION_FUNCTIONS
            ), f"Invalid attention implementation: {self.text_config._attn_implementation}. Available options are: {ALL_ATTENTION_FUNCTIONS.keys()}"

        if self.text_config.sliding_window_size is not None:
            assert self.text_config._attn_implementation in [
                "flex_attention",
                "flash_attention_2",
            ], "Sliding window is only supported for Flex Attention and Flash Attention 2"

    @property
    def is_using_mup(self) -> bool:
        return self._is_using_mup

    @property
    def is_moe_model(self) -> bool:
        """Returns True if the model uses MoE layers"""
        return self.moe_config is not None and len(self.moe_config.layers) > 0

    # Convenience properties to access text config parameters directly
    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        # TODO: assert num_key_value_heads from text encoder and vision encoder are the same?
        return self.text_config.num_key_value_heads

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @vocab_size.setter
    def vocab_size(self, value: int):
        self.text_config.vocab_size = value


NanotronConfigs = Union[LlamaConfig, Starcoder2Config, Qwen2Config, Llama4Config, Any]
