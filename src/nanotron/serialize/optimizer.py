import json
from pathlib import Path
from typing import Optional

from nanotron.nn import distributed as dist
from nanotron.nn import optim as optim
from nanotron.nn.process_groups import DistributedProcessGroups
from nanotron.serialize.main import ObjectType
from nanotron.serialize.serialize import torch_load, torch_save
from nanotron.serialize.xpath import is_local_path, fs_open


def optimizer_filename(dpg: DistributedProcessGroups, is_zero: bool):
    if is_zero is True:
        return f"{ObjectType.OPTIMIZER.value}_pp-{dist.get_rank(dpg.pp_pg)}-of-{dpg.pp_pg.size()}_dp-{dist.get_rank(dpg.dp_pg)}-of-{dpg.dp_pg.size()}_tp-{dist.get_rank(dpg.tp_pg)}-of-{dpg.tp_pg.size()}.pt"
    else:
        return f"{ObjectType.OPTIMIZER.value}_pp-{dist.get_rank(dpg.pp_pg)}-of-{dpg.pp_pg.size()}_tp-{dist.get_rank(dpg.tp_pg)}-of-{dpg.tp_pg.size()}.pt"


def lr_scheduler_filename():
    """The lr_scheduler is the same for all processes."""
    return f"{ObjectType.LR_SCHEDULER.value}.pt"


def save_optimizer(
    optimizer: optim.BaseOptimizer,
    dpg: DistributedProcessGroups,
    root_folder: Path,
):
    """Saves optimizer states
    - If Zero-0 is used, optimizer states are replicated across all DPs. Only DP-0 saves the states
    - If Zero-1 is used, optimizer states are sharded across all DPs. Each DP saves its own states
    """
    # TODO @thomasw21: Figure out if I need to save param groups. Right now I'm assuming no as we only store what's trainable
    # TODO @thomasw21: We can probably "rotate" so that every process stores something (maybe doesn't matter if we're I/O bound)
    root_folder = root_folder / "optimizer"
    if is_local_path(path=root_folder):
        root_folder.mkdir(exist_ok=True, parents=True)

    if dist.get_rank(dpg.world_pg) == 0:
        with fs_open(root_folder / "optimizer_config.json", "w") as fo:
            json.dump({"type": optimizer.__class__.__name__}, fo)

    if (not optimizer.inherit_from(optim.ZeroDistributedOptimizer)) and dist.get_rank(dpg.dp_pg) > 0:
        # this is Zero-0, so only DP-0 saves the optimizer states
        return

    # We dump the optimizer state using `torch.save`
    torch_save(
        optimizer.state_dict(),
        root_folder / optimizer_filename(dpg, is_zero=optimizer.inherit_from(optim.ZeroDistributedOptimizer)),
    )


def save_lr_scheduler(
    lr_scheduler,
    dpg: DistributedProcessGroups,
    root_folder: Path,
):
    """Saves lr scheduler states"""
    if dist.get_rank(dpg.world_pg) > 0:
        # Only WORLD-RANK 0 saves the lr scheduler state
        return

    root_folder = root_folder / "lr_scheduler"
    if is_local_path(path=root_folder):
        root_folder.mkdir(exist_ok=True, parents=True)

    # We dump the optimizer state using `torch.save`
    torch_save(
        lr_scheduler.state_dict(),
        root_folder / lr_scheduler_filename(),
    )


def load_optimizer(
    optimizer: optim.BaseOptimizer,
    dpg: DistributedProcessGroups,
    root_folder: Path,
    map_location: Optional[str] = None,
):
    root_folder = root_folder / "optimizer"
    # `load_state_dict` copies the state dict which can be very large in case of Zero-0 so we load to cpu and then move to the right device
    map_location = "cpu" if not optimizer.inherit_from(optim.ZeroDistributedOptimizer) else map_location

    # TODO @thomasw21: Load optimizer type and check that it's compatible otherwise we might be be loading something else completely
    state_dict = torch_load(
        root_folder / optimizer_filename(dpg, is_zero=optimizer.inherit_from(optim.ZeroDistributedOptimizer)),
        map_location=map_location,
    )
    optimizer.load_state_dict(state_dict)


def load_lr_scheduler(
    lr_scheduler,
    root_folder: Path,
):
    root_folder = root_folder / "lr_scheduler"

    state_dict = torch_load(root_folder / lr_scheduler_filename())
    lr_scheduler.load_state_dict(state_dict)
