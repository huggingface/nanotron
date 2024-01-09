from torch import nn

from nanotron.core import distributed as dist
from nanotron.core.parallel.tied_parameters import get_tied_id_to_param
from nanotron.distributed import ParallelContext, ParallelMode


def initial_sync(model: nn.Module, parallel_context: ParallelContext):
    # Synchronize across dp: basic assumption
    sorted_name_params = sorted(model.named_parameters(), key=lambda x: x[0])
    dp_group = parallel_context.get_group(ParallelMode.DATA)
    for name, param in sorted_name_params:
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=dp_group)

    # Synchronize across tied weights: basic assumption
    for (_, group_ranks), param in sorted(
        get_tied_id_to_param(parameters=model.parameters(), root_module=model).items(), key=lambda x: x[0]
    ):
        group = parallel_context.world_ranks_to_pg[group_ranks]
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
