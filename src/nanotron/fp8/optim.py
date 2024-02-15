from typing import List, Tuple

from torch import nn
from torch.optim import Optimizer


class FP8Adam(Optimizer):
    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 0.01,
        betas: Tuple[float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
    ):
        pass

    def step(self):
        # TODO(xrsrke): update weights in a separate cuda streams
        pass

    def zero_grad(self):
        pass
