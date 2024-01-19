import contextlib
import random
from dataclasses import dataclass
from typing import MutableMapping, Optional, Tuple

import numpy as np
import torch

from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup


@dataclass
class RandomState:
    random: Tuple[int, Tuple[int, ...], None]
    numpy: Tuple[str, np.ndarray, int, int, float]
    torch_cpu: torch.Tensor
    torch_cuda: Optional[torch.Tensor]

    def __eq__(self, other):
        return (
            isinstance(other, RandomState)
            and all(v1 == v2 for v1, v2 in zip(self.random, other.random))
            and all(
                np.array_equal(v1, v2) if isinstance(v1, np.ndarray) else v1 == v2
                for v1, v2 in zip(self.numpy, other.numpy)
            )
            and torch.equal(self.torch_cpu, other.torch_cpu)
            and (
                other.torch_cuda is None if self.torch_cuda is None else torch.equal(self.torch_cuda, other.torch_cuda)
            )
        )


class RandomStates(MutableMapping[str, RandomState]):
    def __init__(self, dict: dict):
        for key, value in dict.items():
            self.check_type(key, value)
        # TODO @thomasw21: We make a copy for safety measure.
        self._dict = dict.copy()

    @staticmethod
    def check_type(key, value):
        if not isinstance(key, str):
            raise ValueError(f"Expected key to be of type str. Got {type(key)}")
        if not isinstance(value, RandomState):
            raise ValueError(f"Expected value to be of type `nanotron.dataclass.RandomState`. Got {type(value)}")

    def __getitem__(self, item):
        return self._dict[item]

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)

    def __delitem__(self, key):
        raise ValueError("Can't delete a random states key")

    def __setitem__(self, key, value):
        if key not in self._dict:
            raise ValueError("Can't add a new random states after initialisation")
        self.check_type(key, value)
        return self._dict.__setitem__(key, value)

    def __eq__(self, other):
        if not isinstance(other, RandomStates):
            return False

        return self._dict == other._dict


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_random_state(random_state: RandomState):
    random.setstate(random_state.random)
    np.random.set_state(random_state.numpy)
    torch.set_rng_state(random_state.torch_cpu)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(random_state.torch_cuda, "cuda")
    else:
        assert random_state.torch_cuda is None


def get_current_random_state():
    """Returns a snapshot of current random state"""
    return RandomState(
        random=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state("cuda") if torch.cuda.is_available() else None,
    )


@contextlib.contextmanager
def branch_random_state(random_states: RandomStates, key: str, enabled: bool):
    """
    Context manager handling random state:
     - upon entering: Stores current random state and set new random state defined by key.
     - upon exiting: updates key in `random_states` to the new current random state, and set back the old one.
    """
    if not enabled:
        yield
        return

    old_random_state = get_current_random_state()

    # Get the new state associated to the key
    new_random_state = random_states[key]
    set_random_state(new_random_state)

    try:
        yield
    finally:
        # Update state from parallel_context with the newest state
        new_random_state = get_current_random_state()
        random_states[key] = new_random_state

        # Set the old state back
        set_random_state(old_random_state)


def get_synced_random_state(
    random_state: RandomState,
    pg: ProcessGroup,
):
    # We use rank 0 as a reference and broadcast random states from that rank to all the other ranks within a group in order to sync them
    reference_rank = 0
    if dist.get_rank(pg) == reference_rank:
        random_states = [random_state]
    else:
        random_states = [None]

    # TODO @thomasw21: broadcast tensor using `broadcast` in order not to use pickle
    dist.broadcast_object_list(
        random_states, src=dist.get_global_rank(pg, reference_rank), group=pg, device=torch.device("cuda")
    )

    new_random_state = random_states[0]
    assert new_random_state is not None
    return new_random_state
