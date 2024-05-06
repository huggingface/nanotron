from dataclasses import dataclass, field
from typing import List, TypedDict

import torch


class WeightHistory(TypedDict):
    step: int
    weight: torch.Tensor


@dataclass
class DoReMiContext:
    # NOTE: this is the current domain weights
    domain_keys: List[str]
    is_proxy: bool
    step_size: float = 1
    smoothing_param: float = 1e-3

    domain_weight_history: WeightHistory = field(default_factory=list)

    @property
    def num_domains(self) -> int:
        return len(self.domain_keys)

    def get_domain_name(self, domain_idx: int) -> str:
        return self.domain_keys[domain_idx]

    def __post_init__(self):
        # NOTE: by default, we do uniform sampling for DoReMi
        self.domain_weights = torch.ones(self.num_domains) / self.num_domains
        assert torch.allclose(
            self.domain_weights.sum(dim=-1), torch.tensor(1.0), rtol=0.001
        ), "Domain weights must sum up to 1."
        assert (
            self.domain_weights.shape[0] == self.num_domains
        ), "The length of domain_weights must be equal to the number of domains"
        self.add_weight_with_history(self.domain_weights, 0)

    def add_weight_with_history(self, domain_weights: torch.Tensor, step: int):
        assert step >= 0, "Step must be a positive integer"
        self.domain_weight_history.append(WeightHistory(step=step, weight=domain_weights.cpu()))
