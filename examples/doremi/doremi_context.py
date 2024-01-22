from dataclasses import dataclass
from typing import List

import torch


@dataclass
class DoReMiContext:
    domain_weights: torch.Tensor
    domain_keys: List[str]
    step_size: float = 0.1
    smoothing_param: float = 1e-3

    @property
    def num_domains(self) -> int:
        return self.domain_weights.shape[0]

    def get_domain_name(self, domain_idx: int) -> str:
        return self.domain_keys[domain_idx]
