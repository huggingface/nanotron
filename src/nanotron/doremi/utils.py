from typing import List

import torch
from torch.utils.data import Dataset


@torch.jit.script
def masked_mean(loss: torch.Tensor, label_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


def compute_domain_weights_based_on_token_count(datasets: List[Dataset]) -> torch.Tensor:
    weights = []
    for d in datasets:
        weights.append(len(d))

    total_samples = sum([len(d) for d in datasets])
    weights = torch.tensor([x / total_samples for x in weights])
    return weights
