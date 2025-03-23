from typing import Literal, Optional, Tuple

import torch
from packaging import version
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from nanotron.nn.ring_attention import ring_flash_attn_varlen_func


# Replace direct import with a function for lazy loading
def get_ring_flash_attn_cuda():
    """Lazily import ring_flash_attn_cuda to avoid early Triton dependency."""
    from nanotron.nn.ring_attention_lucidrain import ring_flash_attn_cuda

    return ring_flash_attn_cuda


def is_torch_flex_attn_available():
    # TODO check if some bugs cause push backs on the exact version
    # NOTE: We require torch>=2.5.0 as it is the first release
    return version.parse(torch.__version__) >= version.parse("2.5.0")


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import flex_attention

# adapted from transformers.integrations.flex_attention.flex_attention_forward
def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    def causal_mod(score, b, h, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            score = score + causal_mask[b][0][q_idx][kv_idx]
        return score

    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        score_mod=causal_mod,
        enable_gqa=True,
        scale=scaling,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=True,
    )
    # lse is returned in float32
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


ALL_ATTENTION_FUNCTIONS = {
    "flash_attention_2": flash_attention_forward,
    "flex_attention": flex_attention_forward,
    "sdpa": sdpa_attention_forward,
    "ring_flash_triton": lambda *args, **kwargs: get_ring_flash_attn_cuda()(*args, **kwargs),
    "ring": ring_flash_attn_varlen_func,
}

AttentionImplementation = Literal[tuple(ALL_ATTENTION_FUNCTIONS.keys())]
