from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch
from typing_extensions import TypeAlias

Args: TypeAlias = Tuple[Any, ...]
Kwargs: TypeAlias = Dict[str, Any]
StateDict: TypeAlias = Dict[str, Any]


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
    def load_state_dict(self, state_dict: dict, map_location: Optional[Union[str, torch.device]] = None) -> None:
        ...

    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        ...

    def inherit_from(self, cls) -> bool:
        ...


Optimizer = TypeVar("Optimizer", BaseOptimizer, torch.optim.Optimizer)


# Modified from torch.optim.Optimizer._process_value_according_to_param_policy
@staticmethod
def _process_value_according_to_param_policy(
    param: torch.Tensor,
    value: torch.Tensor,
    param_id: int,
    param_groups: List[Dict[Any, Any]],
    map_location: Optional[Union[str, torch.device]],
    key: Hashable = None,
) -> torch.Tensor:
    # If map_location is specified, use it instead of param.device
    target_device = map_location if map_location is not None else param.device

    fused = False
    capturable = False
    assert param_groups is not None
    for pg in param_groups:
        if param_id in pg["params"]:
            fused = pg["fused"] if "fused" in pg else False
            capturable = pg["capturable"] if "capturable" in pg else False
            break

    if key == "step":
        if capturable or fused:
            return value.to(dtype=torch.float32, device=target_device)
        else:
            return value
    else:
        if param.is_floating_point():
            return value.to(dtype=param.dtype, device=target_device)
        else:
            return value.to(device=target_device)


# Modified from torch.optim.Optimizer.load_state_dict
@torch._disable_dynamo
def custom_load_state_dict(self, state_dict: StateDict, map_location: Union[str, torch.device]) -> None:
    r"""Loads the optimizer state.

    Args:
        state_dict (dict): optimizer state. Should be an object returned
            from a call to :meth:`state_dict`.
        map_location (str or torch.device, optional): Device where to load the optimizer states.
            If None, states will be loaded to the same device as their corresponding parameters.
            Default: None
    """

    # shallow copy, to be consistent with module API
    state_dict = state_dict.copy()

    for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
        hook_result = pre_hook(self, state_dict)
        if hook_result is not None:
            state_dict = hook_result

    # Validate the state_dict
    groups = self.param_groups

    # Deepcopy as we write into saved_groups later to update state
    saved_groups = deepcopy(state_dict["param_groups"])

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of " "parameter groups")
    param_lens = (len(g["params"]) for g in groups)
    saved_lens = (len(g["params"]) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError(
            "loaded state dict contains a parameter group " "that doesn't match the size of optimizer's group"
        )

    # Update the state
    id_map = dict(
        zip(chain.from_iterable(g["params"] for g in saved_groups), chain.from_iterable(g["params"] for g in groups))
    )

    def _cast(param, value, param_id=None, param_groups=None, key=None):
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            return _process_value_according_to_param_policy(param, value, param_id, param_groups, map_location, key)
        elif isinstance(value, dict):
            return {k: _cast(param, v, param_id=param_id, param_groups=param_groups, key=k) for k, v in value.items()}
        elif isinstance(value, Iterable):
            return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
    for k, v in state_dict["state"].items():
        if k in id_map:
            param = id_map[k]
            state[param] = _cast(param, v, param_id=k, param_groups=state_dict["param_groups"])
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group: Dict[str, Any], new_group: Dict[str, Any]) -> Dict[str, Any]:
        new_group["params"] = group["params"]
        return new_group

    param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    self.__setstate__({"state": state, "param_groups": param_groups})

    for post_hook in self._optimizer_load_state_dict_post_hooks.values():
        post_hook(self)
