from dataclasses import dataclass

import torch


@dataclass
class DoReMiContext:
    domain_weights: torch.Tensor
    step_size: float = 0.1
    smoothing_param: float = 1e-3

    @property
    def num_domains(self) -> int:
        return self.domain_weights.shape[0]
