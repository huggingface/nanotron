from dataclasses import dataclass, field
from typing import List, TypedDict

import torch


class WeightHistory(TypedDict):
    step: int
    weight: torch.Tensor


@dataclass
class DoReMiContext:
    # NOTE: this is the current domain weights
    domain_weights: torch.Tensor
    domain_keys: List[str]
    is_proxy: bool
    step_size: float = 1
    smoothing_param: float = 1e-3

    domain_weight_history: WeightHistory = field(default_factory=list)

    @property
    def num_domains(self) -> int:
        return self.domain_weights.shape[0]

    def get_domain_name(self, domain_idx: int) -> str:
        return self.domain_keys[domain_idx]

    def __post_init__(self):
        assert isinstance(self.domain_weights, list)
        assert isinstance(self.domain_keys, list)
        assert len(self.domain_weights) == len(
            self.domain_keys
        ), "The length of domain_weights and domain_keys must match."

        domain_weights = torch.tensor(self.domain_weights)
        assert torch.allclose(domain_weights.sum(), torch.tensor(1.0), rtol=0.1), "Domain weights must sum up to 1."

        self.domain_weights = domain_weights

        # assert (
        #     domain_weights.shape[0] == self.num_domains
        # ), "The length of domain_weights must be equal to the number of domains"
        self.add_weight_with_history(self.domain_weights, 0)

    def add_weight_with_history(self, domain_weights: torch.Tensor, step: int):
        assert step >= 0, "Step must be a positive integer"
        self.domain_weight_history.append(WeightHistory(step=step, weight=domain_weights.cpu()))
        self.domain_weights = domain_weights
