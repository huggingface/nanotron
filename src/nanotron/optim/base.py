from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

import torch


class BaseOptimizer(ABC):
    id_to_name: Dict[int, str]
    param_groups: List[Dict[str, Any]]

    @abstractmethod
    def __getstate__(self):
        ...

    @abstractmethod
    def __setstate__(self, state):
        ...

    @abstractmethod
    def __repr__(self):
        ...

    @abstractmethod
    def zero_grad(self):
        ...

    @abstractmethod
    def state_dict_additional_keys(self) -> Set[str]:
        """Additional states we store in `state_dict`. It has to be a dictionary using parameter name as key, and a tensor as value."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        ...

    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        ...

    def inherit_from(self, cls) -> bool:
        ...


Optimizer = TypeVar("Optimizer", BaseOptimizer, torch.optim.Optimizer)
