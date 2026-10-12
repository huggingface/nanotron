#!/usr/bin/env python3
"""Benchmark: LlamaRotaryEmbedding vs TorchembedRotaryEmbedding (fused Triton kernel).

Usage:
    python benchmarks/bench_rope.py

Reports median latency and throughput for apply_rotary_pos_emb across
a sweep of sequence lengths. Tests are run on CUDA with bfloat16.
"""

import sys
import time
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Implementations under test
# ---------------------------------------------------------------------------

class LlamaRotaryEmbedding(torch.nn.Module):
    """Verbatim copy of nanotron's LlamaRotaryEmbedding (non-interleaved)."""

    def __init__(self, dim: int, end: int, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.end = end
        self.theta = theta
        self._init()

    def _init(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq.cuda(), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type="cuda", enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=2):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (self.rotate_half(k) * sin)


class TorchembedRotaryEmbedding(torch.nn.Module):
    """Thin wrapper used by nanotron when use_torchembed_rope=True."""

    def __init__(self, dim: int, end: int, theta: float = 500000.0):
        super().__init__()
        from torchembed.positional import RotaryEmbedding as _Impl
        self._impl = _Impl(dim=dim, max_seq_len=end, base=int(theta), use_fused=True).cuda()
        self.end = end

    @torch.no_grad()
    def forward(self, x, position_ids):
        cos = self._impl.cos_cache.to(device=x.device, dtype=x.dtype)
        sin = self._impl.sin_cache.to(device=x.device, dtype=x.dtype)
        if position_ids is None:
            return cos[:x.shape[1]], sin[:x.shape[1]]
        return cos[position_ids], sin[position_ids]

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=2):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        from torchembed._triton import fused_rope_forward
        return fused_rope_forward(q, k, cos, sin)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

WARMUP = 50
ITERS  = 200
DTYPE  = torch.bfloat16

def bench(name, fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1e3  # ms
    return elapsed


def run(batch=4, n_heads=32, d_qk=128, seq_len=2048):
    device = "cuda"
    # position_ids=None → sequential (the pretraining hot path).
    # LlamaRotaryEmbedding needs explicit position_ids; torchembed uses None for its fused path.
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    q = torch.randn(batch, seq_len, n_heads, d_qk, device=device, dtype=DTYPE)
    k = torch.randn(batch, seq_len, n_heads, d_qk, device=device, dtype=DTYPE)

    llama = LlamaRotaryEmbedding(dim=d_qk, end=seq_len + 1).to(device)
    try:
        te = TorchembedRotaryEmbedding(dim=d_qk, end=seq_len + 1).to(device)
        has_te = True
    except Exception as e:
        print(f"  torchembed unavailable: {e}")
        has_te = False

    # Full apply (forward + apply_rotary) — what nanotron calls during inference.
    # For torchembed, position_ids=None triggers the [seq, dim] cos/sin path → fused kernel.
    def llama_fn():
        cos, sin = llama(q, position_ids)
        return llama.apply_rotary_pos_emb(q, k, cos, sin)

    def te_fn():
        # Pass None so forward() returns [seq, dim] cos/sin → fused Triton kernel fires
        cos, sin = te(q, None)
        return te.apply_rotary_pos_emb(q, k, cos, sin)

    t_llama = bench("llama", llama_fn)
    t_te    = bench("te",    te_fn) if has_te else None

    tokens = batch * seq_len
    tput_llama = tokens / (t_llama / 1e3) / 1e6  # M tokens/s

    row = f"  seq={seq_len:5d}  batch={batch}  | llama: {t_llama:6.3f} ms  ({tput_llama:5.1f} Mtok/s)"
    if t_te is not None:
        tput_te = tokens / (t_te / 1e3) / 1e6
        speedup = t_llama / t_te
        row += f"  | torchembed: {t_te:6.3f} ms  ({tput_te:5.1f} Mtok/s)  [{speedup:.2f}x faster]"
    print(row)
    return t_llama, t_te


if __name__ == "__main__":
    print(f"RoPE benchmark — device: {torch.cuda.get_device_name(0)}, dtype: {DTYPE}")
    print(f"Config: batch=4, n_heads=32, d_qk=128")
    print(f"Warmup={WARMUP}, iters={ITERS}\n")

    results = []
    for seq_len in [512, 1024, 2048, 4096, 8192]:
        r = run(seq_len=seq_len)
        results.append((seq_len, *r))

    if results[0][2] is not None:
        print("\nSummary:")
        speedups = [r[1] / r[2] for r in results if r[2] is not None]
        print(f"  Average speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"  Peak speedup:    {max(speedups):.2f}x")
