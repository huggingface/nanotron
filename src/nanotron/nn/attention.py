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
    sliding_window: Optional[int] = None,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if softcap is not None:
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]
        def causal_mod(score, b, h, q_idx, kv_idx):
            if softcap is not None:
                score = softcap * torch.tanh(score / softcap)
            if causal_mask is not None:
                score = score + causal_mask[b][0][q_idx][kv_idx]
            return score
        kwargs["score_mod"] = causal_mod
        

    if sliding_window is not None:
        def sliding_window_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= sliding_window 
            return causal_mask & window_mask
        kwargs["mask_mod"] = sliding_window_causal

    
    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        enable_gqa=True,
        score_mod = kwargs.get("score_mod", None),
        mask_mod = kwargs.get("mask_mod", None),
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


# TODO @nouamane: optimize this, and make sure it works with flashattn and flexattn
@torch.jit.script
def get_attention_mask(position_ids, seq_length):
    attention_mask = torch.zeros(seq_length, seq_length, device=position_ids.device)
    start_indices = torch.where(position_ids == 0)[0]
    cu_seqlens = torch.cat(
        [start_indices, torch.tensor([seq_length], dtype=torch.int32, device=start_indices.device)]
    ).to(torch.int32)
    # make trius for each document
    for i in range(len(cu_seqlens) - 1):
        attention_mask[cu_seqlens[i] : cu_seqlens[i + 1], cu_seqlens[i] : cu_seqlens[i + 1]] = torch.tril(
            torch.ones(cu_seqlens[i + 1] - cu_seqlens[i], cu_seqlens[i + 1] - cu_seqlens[i])
        )
    return attention_mask.to(torch.bool), cu_seqlens  # [seq_length, seq_length]