from torch import nn

from nanotron.parallel.tensor_parallel.nn import TensorParallelEmbedding

MODULE_NAMES_THAT_NOT_FP8 = [
    "token_embedding",
    "input_layernorm",
    "post_attention_layernorm",
    "final_layer_norm",
    "lm_head",
]
MODULES_THAT_IN_FLOAT16 = [TensorParallelEmbedding, nn.LayerNorm]
