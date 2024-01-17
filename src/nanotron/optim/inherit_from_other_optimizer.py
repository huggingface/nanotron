from functools import cache
from typing import Callable, Dict, Optional, Set

import torch

from nanotron.optim.base import BaseOptimizer, Optimizer


class InheritFromOtherOptimizer(BaseOptimizer):
    def __init__(self, optimizer: Optimizer, id_to_name: Dict[int, str]):
        self.optimizer: Optimizer = optimizer
        self.id_to_name = id_to_name

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        return self.optimizer.__setstate__(state)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.optimizer.__repr__()})"

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @cache
    def state_dict_additional_keys(self) -> Set[str]:
        if isinstance(self.optimizer, BaseOptimizer):
            return self.optimizer.state_dict_additional_keys()
        else:
            return set()

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        return self.optimizer.load_state_dict(state_dict)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self.optimizer.step(closure=closure)

    def get_base_optimizer(self):
        if isinstance(self.optimizer, torch.optim.Optimizer):
            return self.optimizer
        else:
            return self.optimizer.get_base_optimizer()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def inherit_from(self, cls):
        if isinstance(self, cls):
            return True
        if isinstance(self.optimizer, InheritFromOtherOptimizer):
            return self.optimizer.inherit_from(cls)
        return False
