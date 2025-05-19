from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from nanotron.config.utils_config import InitScalingMethod
from nanotron.nn.attention import ALL_ATTENTION_FUNCTIONS, AttentionImplementation

# The default attention implementation to use
DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"


@dataclass
class RandomInit:
    std: float
    scaling_method: InitScalingMethod = InitScalingMethod.NUM_LAYERS


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

    num_experts: int # Total number of experts
    top_k: int  # Number of experts to route each token to
    moe_hidden_size: int  # Hidden size of the MoE layer
    moe_intermediate_size: int  # Intermediate size of the MoE layer
    enable_shared_expert: bool = False  # Whether to use a shared expert alongside specialized experts
    shared_expert_hidden_size: int = 4096  # Hidden size of the shared expert
    shared_expert_intermediate_size: int = 11008  # Intermediate size of the shared expert
    router_aux_loss_coef: float = 0.01  # Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended.
    layers: List[int] = field(
        default_factory=lambda: [-1]
    )  # Indices of layers that use MoE. -1 means all layers. Default is all layers
    token_dispatcher_type: str = "alltoall"  # Communication pattern for MoE ("alltoall" or "allgather")
    use_torch_permute: bool = True  # Whether to use Haojun's permute

    moe_impl: str = "transformer_engine"
    grouped_gemm_imple: Literal["transformer_engine", "megablock_grouped_gemm"] = "transformer_engine"

    # Transformer-Engine specific config
    num_shared_experts: int = None
    rotary_base: int = None
    rotary_scaling_factor: int = None
    max_position_embeddings: int = None

    moe_z_loss_coeff: Optional[float] = None # Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended.
    gradient_accumulation_fusion: bool = False # 
    disable_parameter_transpose_cache: bool = False # When set to true, the parameter transposes are not cached for subsequent iterations.
    bias_activation_fusion: bool = True
    permute_fusion: bool = False
    input_jitter_eps: float = None  # Add noise to the input tensor. https://arxiv.org/abs/2101.03961
    # The load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss
    # used in GShard and SwitchTransformer; "seq_aux_loss" corresponds to the loss used in DeepSeekV2,
    # which computes the loss for each individual sample; "sinkhorn" corresponds to the balancing
    # algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss".
    router_load_balancing_type: str = "aux_loss" 
    moe_expert_capacity_factor: Optional[float] = None
    moe_pad_expert_input_to_capacity: bool = False
    moe_token_drop_policy: str = "probs"
    moe_router_pre_softmax: bool = False
    moe_router_num_groups: Optional[int] = None
    moe_router_group_topk: Optional[int] = None
    # Scaling factor for routing score in top-k selection, only works when moe_router_pre_softmax enabled. Defaults to None, which means no scaling
    moe_router_topk_scaling_factor: Optional[float] = None
    moe_router_score_function: str = "softmax" # Score function for MoE routing. Can be "softmax" or "sigmoid"
    moe_router_expert_bias: Optional[bool] = None
    moe_router_dtype: Optional[str] = None
    # TODO: add docs https://github.com/NVIDIA/Megatron-LM/blob/dab7723821fc326564634b398a809d43740a6c8d/megatron/core/transformer/transformer_config.py

    def __post_init__(self):
        # Validate the configuration
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot be greater than num_experts ({self.num_experts})")

        if self.token_dispatcher_type not in ["alltoall", "allgather"]:
            raise ValueError(
                f"token_dispatcher_type must be one of ['alltoall', 'allgather'], got {self.token_dispatcher_type}"
            )

        assert self.grouped_gemm_imple in [
            "transformer_engine",
            "megablock_grouped_gemm",
        ], f"Invalid grouped gemm implementation: {self.grouped_gemm_imple}. Available options are: ['transformer_engine', 'megablock_grouped_gemm']"

        if self.top_k == 1 and self.moe_router_score_function == 'softmax' and not self.moe_router_pre_softmax and self.router_load_balancing_type != 'sinkhorn':
            # https://github.com/NVIDIA/Megatron-LM/blob/28118fcdc22e42621776a021af568ae39c198418/megatron/core/transformer/transformer_config.py#L805-L813
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")


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
    no_rope_layer: Optional[
        int
    ] = None  # Skip rope every no_rope_layer layers (see https://arxiv.org/abs/2501.18795 https://arxiv.org/abs/2305.19466 and Llama4)
    _fused_rotary_emb: bool = True
    _fused_rms_norm: bool = True
    _use_qkv_packed: bool = True
    _use_doc_masking: bool = False

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
        if self.no_rope_layer is not None:
            assert (
                self.num_hidden_layers % self.no_rope_layer == 0
            ), "no_rope_layer must be a multiple of num_hidden_layers"

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


NanotronConfigs = Union[LlamaConfig, Starcoder2Config, Qwen2Config, Any]
