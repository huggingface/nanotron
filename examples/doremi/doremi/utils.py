from typing import List

import torch
from torch.utils.data import Dataset


@torch.jit.script
def masked_mean(loss: torch.Tensor, label_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


def compute_domain_weights_based_on_token_count(datasets: List[Dataset]) -> torch.Tensor:
    num_samples_per_domain = [len(d) for d in datasets]
    total_samples = sum(num_samples_per_domain)
    weights = torch.tensor([num_sample / total_samples for num_sample in num_samples_per_domain])
    return weights
