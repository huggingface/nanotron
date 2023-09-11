import os

import numpy as np
import torch
from torch import distributed as torch_dist

import brrr.core.distributed as dist
from brrr.core.dataclass import DistributedProcessGroups


def get_process_groups(
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> DistributedProcessGroups:
    """
    Generate all the process groups necessary for training, and returning current ranks process groups.

    :param data_parallel_size: int
    :param tensor_parallel_size: int
    :param pipeline_parallel_size: int
    :return: DistributedProcessGroups
    """
    if not dist.is_available():
        raise ValueError("`torch.distributed is not available as a package, please install it.")

    if not dist.is_initialized():
        initialize_torch_distributed()

    world_pg = torch_dist.distributed_c10d._get_default_group()
    world_size = world_pg.size()
    world_rank = dist.get_rank(world_pg)
    assert (
        world_size == data_parallel_size * tensor_parallel_size * pipeline_parallel_size
    ), f"{world_size} != {data_parallel_size * tensor_parallel_size * pipeline_parallel_size}"

    # In the current implementation in DeepSpeed, tp then dp then pp
    #   https://cs.github.com/microsoft/DeepSpeed/blob/591744eba33f2ece04c15c73c02edaf384dca226/deepspeed/runtime/pipe/topology.py#L243

    ranks = np.arange(0, world_size).reshape((pipeline_parallel_size, data_parallel_size, tensor_parallel_size))
    world_ranks_to_pg = {}

    tp_pg: dist.ProcessGroup
    ranks_with_tp_last = ranks.reshape((pipeline_parallel_size * data_parallel_size, tensor_parallel_size))
    for tp_ranks in ranks_with_tp_last:
        sorted_ranks = tuple(sorted(tp_ranks))
        if sorted_ranks not in world_ranks_to_pg:
            new_group = dist.new_group(ranks=tp_ranks)
            world_ranks_to_pg[sorted_ranks] = new_group
        else:
            new_group = world_ranks_to_pg[sorted_ranks]
        if world_rank in tp_ranks:
            tp_pg = new_group

    dp_pg: dist.ProcessGroup
    ranks_with_dp_last = ranks.transpose((0, 2, 1)).reshape(
        (pipeline_parallel_size * tensor_parallel_size, data_parallel_size)
    )
    for dp_ranks in ranks_with_dp_last:
        sorted_ranks = tuple(sorted(dp_ranks))
        if sorted_ranks not in world_ranks_to_pg:
            new_group = dist.new_group(ranks=dp_ranks)
            world_ranks_to_pg[sorted_ranks] = new_group
        else:
            new_group = world_ranks_to_pg[sorted_ranks]
        if world_rank in dp_ranks:
            dp_pg = new_group

    pp_pg: dist.ProcessGroup
    ranks_with_pp_last = ranks.transpose((2, 1, 0)).reshape(
        (tensor_parallel_size * data_parallel_size, pipeline_parallel_size)
    )
    for pp_ranks in ranks_with_pp_last:
        sorted_ranks = tuple(sorted(pp_ranks))
        if sorted_ranks not in world_ranks_to_pg:
            new_group = dist.new_group(ranks=pp_ranks)
            world_ranks_to_pg[sorted_ranks] = new_group
        else:
            new_group = world_ranks_to_pg[sorted_ranks]
        if world_rank in pp_ranks:
            pp_pg = new_group

    # We build model parallel group (combination of both tensor parallel and pipeline parallel)
    for dp_rank in range(data_parallel_size):
        pp_and_tp_ranks = ranks[:, dp_rank, :].reshape(-1)
        sorted_ranks = tuple(sorted(pp_and_tp_ranks))
        if sorted_ranks not in world_ranks_to_pg:
            new_group = dist.new_group(ranks=pp_and_tp_ranks)
            world_ranks_to_pg[sorted_ranks] = new_group

    return DistributedProcessGroups(
        world_pg=world_pg,
        world_rank_matrix=ranks,
        dp_pg=dp_pg,
        tp_pg=tp_pg,
        pp_pg=pp_pg,
        world_ranks_to_pg=world_ranks_to_pg,
    )


def initialize_torch_distributed():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        # Set the device id.
        # `torch.cuda.device_count` should return the number of device on a single node.
        # We assume the nodes to be homogeneous (same number of gpus per node)
        device_id = local_rank
        torch.cuda.set_device(torch.cuda.device(device_id))
        backend = "nccl"
    else:
        # TODO @thomasw21: Maybe figure out a way to do distributed `cpu` training at some point
        raise NotImplementedError("Dunno if this works.")
        backend = "gloo"

    # Call the init process.
    torch_dist.init_process_group(backend=backend, world_size=world_size, rank=rank, timeout=dist.default_pg_timeout)
    return True
