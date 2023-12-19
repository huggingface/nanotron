from pathlib import Path

from nanotron.nn import distributed as dist
from nanotron.nn.process_groups import DistributedProcessGroups
from nanotron.nn.random import RandomStates
from nanotron.serialize.xpath import is_local_path
from nanotron.serialize.serialize import torch_load, torch_save


def save_random_states(
    random_states: RandomStates,
    dpg: DistributedProcessGroups,
    root_folder: Path,
):
    """All processes save their own random state"""
    filename = (
        root_folder
        / "random"
        / f"tp-{dist.get_rank(dpg.tp_pg)}-of-{dpg.tp_pg.size()}_dp-{dist.get_rank(dpg.dp_pg)}-of-{dpg.dp_pg.size()}_pp-{dist.get_rank(dpg.pp_pg)}-of-{dpg.pp_pg.size()}.pt"
    )
    if is_local_path(filename):
        filename.parent.mkdir(exist_ok=True, parents=True)

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    torch_save(random_states, filename)


def load_random_states(dpg: DistributedProcessGroups, root_folder: Path):
    # TODO @thomasw21: This basically assumes that we have exactly the same topology as the one we used when saving.
    filename = (
        root_folder
        / "random"
        / f"tp-{dist.get_rank(dpg.tp_pg)}-of-{dpg.tp_pg.size()}_dp-{dist.get_rank(dpg.dp_pg)}-of-{dpg.dp_pg.size()}_pp-{dist.get_rank(dpg.pp_pg)}-of-{dpg.pp_pg.size()}.pt"
    )

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    state = torch_load(filename)

    return state
