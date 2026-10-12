#!/usr/bin/env python3
"""Benchmark: LLaMA forward pass baseline vs torch.compile.

Measures single-device forward-pass throughput (tokens/sec) with and without
torch.compile, across a sweep of sequence lengths. This mirrors the improvement
you would see during pretraining with PP=1 (the supported configuration).

Usage:
    python benchmarks/bench_compile.py

Requirements:
    pip install nanotron flash-attn
    (runs on a single GPU, no distributed setup needed)
"""

import time

import torch
import torch.nn as nn

WARMUP = 10
ITERS  = 50
DTYPE  = torch.bfloat16
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Minimal LLaMA-style FFN + Attention block for isolated benchmarking
# (avoids nanotron's distributed machinery while being representative)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.up   = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = dim // n_heads
        self.qkv     = nn.Linear(dim, 3 * dim, bias=False)
        self.o       = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, S, D))


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1  = RMSNorm(dim)
        self.attn   = AttentionBlock(dim, n_heads)
        self.norm2  = RMSNorm(dim)
        self.ffn    = SwiGLU(dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLLaMA(nn.Module):
    """Minimal LLaMA-style model for benchmarking (no PP/TP machinery)."""

    def __init__(self, vocab=32000, dim=4096, n_heads=32, n_layers=8, ffn_dim=11008):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([TransformerLayer(dim, n_heads, ffn_dim) for _ in range(n_layers)])
        self.norm   = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def run(batch=2, seq_len=2048, dim=4096, n_heads=32, n_layers=8, ffn_dim=11008):
    input_ids = torch.randint(0, 32000, (batch, seq_len), device=DEVICE)

    model = MiniLLaMA(dim=dim, n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim)
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()

    compiled = torch.compile(model, fullgraph=False)

    def baseline_fn():
        with torch.no_grad():
            return model(input_ids)

    def compiled_fn():
        with torch.no_grad():
            return compiled(input_ids)

    t_base     = bench(baseline_fn)
    t_compiled = bench(compiled_fn)

    tokens     = batch * seq_len
    tput_base  = tokens / (t_base / 1e3) / 1e3      # K tok/s
    tput_comp  = tokens / (t_compiled / 1e3) / 1e3
    speedup    = t_base / t_compiled

    print(
        f"  seq={seq_len:5d}  batch={batch}  layers={n_layers}  "
        f"| baseline: {t_base:7.1f} ms  ({tput_base:5.1f} Ktok/s)  "
        f"| compiled: {t_compiled:7.1f} ms  ({tput_comp:5.1f} Ktok/s)  "
        f"[{speedup:.2f}x faster]"
    )
    return t_base, t_compiled


if __name__ == "__main__":
    print(f"torch.compile benchmark — device: {torch.cuda.get_device_name(0)}, dtype: {DTYPE}")
    print(f"Model: MiniLLaMA  dim=4096  n_heads=32  ffn_dim=11008  (LLaMA-7B layer config)")
    print(f"Warmup={WARMUP}, iters={ITERS}\n")
    print("Note: first 'compiled' call triggers JIT compilation, already excluded by warmup.\n")

    results = []
    for seq_len in [512, 1024, 2048, 4096]:
        r = run(seq_len=seq_len)
        results.append((seq_len, *r))

    speedups = [r[1] / r[2] for r in results]
    print(f"\nSummary:")
    print(f"  Average speedup: {sum(speedups)/len(speedups):.2f}x")
    print(f"  Peak speedup:    {max(speedups):.2f}x")
