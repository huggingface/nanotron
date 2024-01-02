import pytest
import torch
from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
from nanotron.models.llama import RotaryEmbedding
from nanotron.core.random import set_random_seed

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seqlen_offset", [0, 711])
@pytest.mark.parametrize("rotary_emb_fraction", [0.5, 1.0])
def test_rotary(rotary_emb_fraction, seqlen_offset, dtype):
    device = "cuda"
    rtol, atol = (1e-7, 1e-6)
    set_random_seed(42)
    batch_size = 8
    seqlen_total = 2048
    seqlen = seqlen_total - seqlen_offset
    nheads = 16
    headdim = 128
    rotary_dim = int(headdim * rotary_emb_fraction)
    
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype, requires_grad=True
    )

    qkv_flash = qkv.detach().clone().requires_grad_()  # Our implementation modifies qkv inplace

    rotary = RotaryEmbedding(rotary_dim, device=device)
    flash_rotary = FlashRotaryEmbedding(rotary_dim, device=device)

    out = rotary(qkv, seqlen_offset=seqlen_offset)
    out_flash = flash_rotary(qkv_flash, seqlen_offset=seqlen_offset)

    torch.testing.assert_close(rotary._cos_cached, flash_rotary._cos_cached, rtol=rtol, atol=atol)
    torch.testing.assert_close(rotary._sin_cached, flash_rotary._sin_cached, rtol=rtol, atol=atol)
    torch.testing.assert_close(out, out_flash, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    g_flash = g.clone()

    out.backward(g)
    out_flash.backward(g_flash)

    torch.testing.assert_close(qkv.grad, qkv_flash.grad, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seqlen_offset", [0, 711])
@pytest.mark.parametrize("rotary_emb_fraction", [0.5, 1.0])
def test_rotary_kv_cache(rotary_emb_fraction, seqlen_offset, dtype):
    device = "cuda"
    rtol, atol = (1e-7, 1e-6)
    set_random_seed(42)
    batch_size = 8
    seqlen_total = 2048
    seqlen = seqlen_total - seqlen_offset
    nheads = 16
    headdim = 128
    rotary_dim = int(headdim * rotary_emb_fraction)
    
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype, requires_grad=True
    )

    qkv_flash = qkv.detach().clone().requires_grad_()  # Our implementation modifies qkv inplace

    rotary = RotaryEmbedding(rotary_dim, device=device)
    flash_rotary = FlashRotaryEmbedding(rotary_dim, device=device)

    out_q, out_kv = rotary(qkv=qkv[:, :, 0, ...], kv=qkv[:, :, 1:, ...], seqlen_offset=seqlen_offset)
    out_q_flash, out_kv_flash = flash_rotary(qkv=qkv_flash[:, :, 0, ...], kv=qkv_flash[:, :, 1:, ...], seqlen_offset=seqlen_offset)

    torch.testing.assert_close(rotary._cos_cached, flash_rotary._cos_cached, rtol=rtol, atol=atol)
    torch.testing.assert_close(rotary._sin_cached, flash_rotary._sin_cached, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_q, out_q_flash, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_kv, out_kv_flash, rtol=rtol, atol=atol)

    g_q = torch.randn_like(out_q)
    g_kv = torch.randn_like(out_kv)
    g_q_flash = g_q.clone()
    g_kv_flash = g_kv.clone()

    out_q.backward(g_q)
    out_kv.backward(g_kv)
    out_q_flash.backward(g_q_flash)
    out_kv_flash.backward(g_kv_flash)

    torch.testing.assert_close(qkv.grad, qkv_flash.grad, rtol=rtol, atol=atol)