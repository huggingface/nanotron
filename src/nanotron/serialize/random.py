from pathlib import Path

import torch

from nanotron.core import distributed as dist
from nanotron.core.process_groups import DistributedProcessGroups
from nanotron.core.random import RandomStates
from nanotron.distributed import ParallelContext, ParallelMode


def save_random_states(random_states: RandomStates, root_folder: Path, parallel_context: ParallelContext):
    """All processes save their own random state"""
    tp_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
    pp_rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)

    tp_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    dp_world_size = parallel_context.get_world_size(ParallelMode.DATA)
    pp_world_size = parallel_context.get_world_size(ParallelMode.PIPELINE)

    filename = (
        root_folder
        / "random"
        / f"tp-{tp_rank}-of-{tp_world_size}_dp-{dp_rank}-of-{dp_world_size}_pp-{pp_rank}-of-{pp_world_size}.pt"
    )
    filename.parent.mkdir(exist_ok=True, parents=True)

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    torch.save(random_states, filename)


def load_random_states(dpg: DistributedProcessGroups, root_folder: Path):
    # TODO @thomasw21: This basically assumes that we have exactly the same topology as the one we used when saving.
    filename = (
        root_folder
        / "random"
        / f"tp-{dist.get_rank(dpg.tp_pg)}-of-{dpg.tp_pg.size()}_dp-{dist.get_rank(dpg.dp_pg)}-of-{dpg.dp_pg.size()}_pp-{dist.get_rank(dpg.pp_pg)}-of-{dpg.pp_pg.size()}.pt"
    )

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    state = torch.load(filename)

    return state
