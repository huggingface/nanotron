import os
from typing import Literal, Tuple

import numpy as np
import torch

import nanotron.distributed as dist

DistributedBackend = Literal["gloo", "mpi", "nccl"]


class ParallelContext:
    def __init__(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        backend: DistributedBackend = "nccl",
    ):
        """Initialize parallel context."""
        num_gpus_per_model = tensor_parallel_size * pipeline_parallel_size
        world_size = int(os.environ["WORLD_SIZE"])

        assert (
            world_size % data_parallel_size == 0
        ), "The total number of processes must be divisible by the data parallel size."
        assert world_size % num_gpus_per_model == 0, (
            "The total number of processes must be divisible by"
            "the number of GPUs per model (tensor_parallel_size * pipeline_parallel_size)."
        )
        if num_gpus_per_model * data_parallel_size != world_size:
            raise ValueError(
                f"The number of process requires to run all replicas ({num_gpus_per_model * data_parallel_size})",
                f"must be equal to the world size ({world_size}).",
            )

        if not dist.is_available():
            raise ValueError("`torch.distributed is not available as a package, please install it.")

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size

        self._groups = {}

        self.set_device()

        assert backend == "nccl", "Only nccl backend is supported for now."

        if not dist.is_initialized():
            dist.initialize_torch_distributed()

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        ranks = list(range(world_size))
        process_group = dist.new_group(
            ranks=ranks,
            backend=dist.get_backend(),
        )
        self.world_pg = process_group

        self._init_parallel_groups()

    def _init_parallel_groups(self):
        """Initialize 3D parallelism's all process groups."""
        # NOTE: ensure all processes have joined the global group
        # before creating other groups
        dist.barrier(group=self.world_pg)

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        ranks = np.arange(0, world_size).reshape(
            (self.pipeline_parallel_size, self.data_parallel_size, self.tensor_parallel_size)
        )
        world_ranks_to_pg = {}

        tp_pg: dist.ProcessGroup
        ranks_with_tp_last = ranks.reshape(
            (self.pipeline_parallel_size * self.data_parallel_size, self.tensor_parallel_size)
        )
        for tp_ranks in ranks_with_tp_last:
            sorted_ranks = tuple(sorted(tp_ranks))
            if sorted_ranks not in world_ranks_to_pg:
                new_group = dist.new_group(ranks=tp_ranks)
                world_ranks_to_pg[sorted_ranks] = new_group
            else:
                new_group = world_ranks_to_pg[sorted_ranks]
            if rank in tp_ranks:
                tp_pg = new_group

        dp_pg: dist.ProcessGroup
        ranks_with_dp_last = ranks.transpose((0, 2, 1)).reshape(
            (self.pipeline_parallel_size * self.tensor_parallel_size, self.data_parallel_size)
        )
        for dp_ranks in ranks_with_dp_last:
            sorted_ranks = tuple(sorted(dp_ranks))
            if sorted_ranks not in world_ranks_to_pg:
                new_group = dist.new_group(ranks=dp_ranks)
                world_ranks_to_pg[sorted_ranks] = new_group
            else:
                new_group = world_ranks_to_pg[sorted_ranks]
            if rank in dp_ranks:
                dp_pg = new_group

        pp_pg: dist.ProcessGroup
        ranks_with_pp_last = ranks.transpose((2, 1, 0)).reshape(
            (self.tensor_parallel_size * self.data_parallel_size, self.pipeline_parallel_size)
        )
        for pp_ranks in ranks_with_pp_last:
            sorted_ranks = tuple(sorted(pp_ranks))
            if sorted_ranks not in world_ranks_to_pg:
                new_group = dist.new_group(ranks=pp_ranks)
                world_ranks_to_pg[sorted_ranks] = new_group
            else:
                new_group = world_ranks_to_pg[sorted_ranks]
            if rank in pp_ranks:
                pp_pg = new_group

        # TODO(xrsrke): this looks unnecessary, remove it if possible
        # We build model parallel group (combination of both tensor parallel and pipeline parallel)
        for dp_rank in range(self.data_parallel_size):
            pp_and_tp_ranks = ranks[:, dp_rank, :].reshape(-1)
            sorted_ranks = tuple(sorted(pp_and_tp_ranks))
            if sorted_ranks not in world_ranks_to_pg:
                new_group = dist.new_group(ranks=pp_and_tp_ranks)
                world_ranks_to_pg[sorted_ranks] = new_group

        self.tp_pg = tp_pg
        self.dp_pg = dp_pg
        self.pp_pg = pp_pg

        self.world_rank_matrix = ranks
        self.world_ranks_to_pg = world_ranks_to_pg

        dist.barrier()

    def set_device(self):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # NOTE: Set the device id.
        # `torch.cuda.device_count` should return the number of device on a single node.
        # We assume the nodes to be homogeneous (same number of gpus per node)
        device_id = local_rank
        torch.cuda.set_device(torch.cuda.device(device_id))

    def get_3d_ranks(self, world_rank: int) -> Tuple[int, int, int]:
        pp_rank = (world_rank // (self.tp_pg.size() * self.dp_pg.size())) % self.pp_pg.size()
        dp_rank = (world_rank // self.tp_pg.size()) % self.dp_pg.size()
        tp_rank = world_rank % self.tp_pg.size()
        return (pp_rank, dp_rank, tp_rank)
