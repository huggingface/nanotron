from dataclasses import dataclass
from typing import List

import torch


@dataclass
class DoReMiContext:
    domain_weights: torch.Tensor
    domain_keys: List[str]
    is_proxy: bool
    step_size: float = 1
    smoothing_param: float = 1e-3

    @property
    def num_domains(self) -> int:
        return self.domain_weights.shape[0]

    def get_domain_name(self, domain_idx: int) -> str:
        return self.domain_keys[domain_idx]

    def __post_init__(self):
        assert self.domain_weights.dim() == 1, "The domain_weights tensor must be 1-dimensional"
        assert torch.allclose(
            self.domain_weights.sum(dim=-1), torch.tensor(1.0), rtol=0.1
        ), "Domain weights must sum up to 1."
        assert (
            self.domain_weights.shape[0] == self.num_domains
        ), "The length of domain_weights must be equal to the number of domains"
