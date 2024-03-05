import functools
import os

from torch import nn

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.tied_parameters import get_tied_id_to_param


def assert_cuda_max_connections_set_to_1(func):
    flag_is_set_to_1 = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal flag_is_set_to_1
        if flag_is_set_to_1 is None:
            assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1"
            flag_is_set_to_1 = True
        return func(*args, **kwargs)

    return wrapper


def initial_sync(model: nn.Module, parallel_context: ParallelContext):
    # Synchronize across dp: basic assumption
    sorted_name_params = sorted(model.named_parameters(), key=lambda x: x[0])
    for name, param in sorted_name_params:
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=parallel_context.dp_pg)

    # Synchronize across tied weights: basic assumption
    for (_, group_ranks), param in sorted(
        get_tied_id_to_param(parameters=model.parameters(), root_module=model).items(), key=lambda x: x[0]
    ):
        group = parallel_context.world_ranks_to_pg[group_ranks]
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
