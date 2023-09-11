import contextlib
import random

import numpy as np
import torch

from brrr.core import distributed as dist
from brrr.core.dataclass import RandomState, RandomStates
from brrr.core.distributed import ProcessGroup


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
        # Update state from dpg with the newest state
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
