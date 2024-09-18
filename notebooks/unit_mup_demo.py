import re
from typing import *

import datasets
import einops
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import torch
from torch import nn, Tensor, tensor
import torch.nn.functional as F

import unit_scaling as uu
import unit_scaling.functional as U

# Config & helpers
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



vocab_size = 128
depth = 14
head_size = 32
mlp_expansion = 4

# Training
n_steps = int(48000)
warmup_steps = int(5000)
batch_size = 64
sequence_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compile = True


dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
data = torch.frombuffer(bytearray("".join(dataset["text"]), encoding="utf8"), dtype=torch.uint8)
def batches() -> Iterable[Tensor]:
    for _ in range(n_steps):
        yield torch.stack([
            data[i:i + sequence_length].to(device=device, dtype=torch.long)
            for i in torch.randint(0, len(data) - sequence_length, size=(batch_size,))
        ])

def show_layer_stats(layer: nn.Module, input_shape: Tuple[int, ...], is_model=False) -> None:
    if is_model is False:
        input = torch.randn(*input_shape, requires_grad=True)
    else:
        input = list(batches())[0]
    
    output = layer(input)
    output.backward(torch.randn_like(output))
    print(f"# {type(layer).__name__}:")
    for k, v in {
        "output": output.std(),
        # "input.grad": input.grad.std(),
        **{f"{name}": param.std() for name, param in layer.named_parameters()},
        # **{f"{name}.grad": param.grad.std() for name, param in layer.named_parameters()},
    }.items():
        print(f"{k:>20}.std = {v.item():.2f}")



class UmupTransformerLayer(nn.Module):
    def __init__(self, width: int, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = uu.LayerNorm(width)
        self.attn_qkv = uu.Linear(width, 3 * width)
        self.attn_out = uu.Linear(width, width)

        self.mlp_norm = uu.LayerNorm(width)
        self.mlp_up = uu.Linear(width, mlp_expansion * width)
        self.mlp_gate = uu.Linear(width, mlp_expansion * width)
        self.mlp_down = uu.Linear(mlp_expansion * width, width)

        tau_rule = uu.transformer_residual_scaling_rule()
        self.attn_tau = tau_rule(2 * layer_idx, 2 * depth)
        self.mlp_tau = tau_rule(2 * layer_idx + 1, 2 * depth)

    def forward(self, input: Tensor) -> Tensor:
        residual, skip = U.residual_split(input, self.attn_tau)
        residual = self.attn_norm(residual)
        q, k, v = einops.rearrange(self.attn_qkv(residual), "b s (z h d) -> z b h s d", d=head_size, z=3)
        qkv = U.scaled_dot_product_attention(q, k, v, is_causal=True)
        residual = self.attn_out(einops.rearrange(qkv, "b h s d -> b s (h d)"))
        input = U.residual_add(residual, skip, self.attn_tau)

        residual, skip = U.residual_split(input, self.mlp_tau)
        residual = self.mlp_norm(residual)
        residual = self.mlp_down(U.silu_glu(self.mlp_up(residual), self.mlp_gate(residual)))
        return U.residual_add(residual, skip, self.mlp_tau)

class UmupTransformerDecoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.embedding = uu.Embedding(vocab_size, width)
        self.layers = uu.DepthSequential(*(UmupTransformerLayer(width, i) for i in range(depth)))
        self.final_norm = uu.LayerNorm(width)
        self.projection = uu.LinearReadout(width, vocab_size)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        input = self.embedding(input_ids)
        input = self.layers(input)
        input = self.final_norm(input)
        return self.projection(input)

    def loss(self, input_ids: Tensor) -> Tensor:
        logits = self(input_ids).float()
        return U.cross_entropy(
            logits[..., :-1, :].flatten(end_dim=-2), input_ids[..., 1:].flatten()
        )

if __name__ == "__main__":
    # model = UmupTransformerLayer(128, 0)
    model = UmupTransformerDecoder(512)
    model = model.to("cuda")

    for n, p in model.named_parameters():
        print(f"name: {n}, shape: {p.shape}, weight_mup_type: {p.mup_type}, mup_scaling_depth: {p.mup_scaling_depth}")

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            print(f"name: {n}, constraint: {m.constraint}, weight_mup_type: {m.weight_mup_type}")

    # print(f"nane: projection, constraint: {model.projection.constraint} \n")
    # assert 1 ==1 

    show_layer_stats(model, (batch_size, sequence_length, 512), is_model=True)
