from torch import nn

from brrr.core import distributed as dist
from brrr.core.dataclass import DistributedProcessGroups
from brrr.core.parallelism.tied_parameters import get_tied_id_to_param


def initial_sync(model: nn.Module, dpg: DistributedProcessGroups):
    # Synchronize across dp: basic assumption
    sorted_name_params = sorted(model.named_parameters(), key=lambda x: x[0])
    for name, param in sorted_name_params:
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=dpg.dp_pg)

    # Synchronize across tied weights: basic assumption
    for (_, group_ranks), param in sorted(
        get_tied_id_to_param(parameters=model.parameters(), root_module=model).items(), key=lambda x: x[0]
    ):
        group = dpg.world_ranks_to_pg[group_ranks]
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
