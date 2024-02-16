import pytest
import torch
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.random import (
    RandomStates,
    branch_random_state,
    get_current_random_state,
    get_synced_random_state,
)


@pytest.mark.skipif(available_gpus() < 2, reason="Testing test_random_state_sync requires at least 2 gpus")
@pytest.mark.parametrize("tp,dp,pp", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
@rerun_if_address_is_in_use()
def test_random_state_sync(tp: int, dp: int, pp: int):
    # TODO @nouamane: Make a test with 4 gpus (2 in one pg, 2 in other pg)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_random_state_sync)()


def _test_random_state_sync(parallel_context: ParallelContext):
    current_random_state = get_current_random_state()
    reference_rank = 0
    pg = next(
        (pg for pg in [parallel_context.tp_pg, parallel_context.dp_pg, parallel_context.pp_pg] if pg.size() == 2)
    )

    # Check that they are not equal across process group
    if dist.get_rank(pg) == reference_rank:
        random_states = [current_random_state]
    else:
        random_states = [None]
    dist.broadcast_object_list(random_states, src=reference_rank, group=pg)
    if dist.get_rank(pg) != reference_rank:
        assert current_random_state != random_states[0]

    # Sync random state
    synced_random_state = get_synced_random_state(current_random_state, pg=pg)

    # Check that they are equal across process group
    random_states = [synced_random_state]
    dist.broadcast_object_list(random_states, src=reference_rank, group=pg)
    if dist.get_rank(pg) != reference_rank:
        assert current_random_state != random_states[0]

    parallel_context.destroy()


def test_random_state_fork_random_operation_in_global_context():
    key = "my_random_state"
    random_state = get_current_random_state()
    random_states = RandomStates({key: random_state})
    assert random_states[key] == random_state

    # Random operation that updates the random state
    torch.randn(1)

    new_random_state = get_current_random_state()

    # Check that random states changed
    assert new_random_state != random_state
    assert random_states[key] == random_state

    # Check that within the context manager the random state matches the one we stored in `random_states`
    with branch_random_state(random_states=random_states, key=key, enabled=True):
        assert random_states[key] == random_state
        assert get_current_random_state() == random_states[key]

    # Check that random states if back to global one
    assert get_current_random_state() == new_random_state


def test_random_state_fork_random_operation_in_local_context():
    key = "my_random_state"
    random_state = get_current_random_state()
    random_states = RandomStates({key: random_state})

    # Check that within the context manager the random state matches the one we stored in `random_states`
    with branch_random_state(random_states=random_states, key=key, enabled=True):
        old_random_state = get_current_random_state()
        assert old_random_state == random_states[key]

        # Random operation that updates the random state
        torch.randn(1)

        # Check that random states changed
        new_random_state = get_current_random_state()

    # Check that global random_state hasn't changed
    assert get_current_random_state() == random_state

    # Check that local random_state has changed and is equal to `new_random_state`
    assert old_random_state != random_states[key]
    assert new_random_state == random_states[key]
