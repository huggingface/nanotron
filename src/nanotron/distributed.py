import datetime
from functools import cache, lru_cache
from typing import List, Optional, Literal, Tuple
import numpy as np
import torch
from packaging import version
from torch import distributed as dist
from torch.distributed import *  # noqa
from torch.distributed.distributed_c10d import ProcessGroup

torch_version_above_1_13 = version.parse(torch.__version__) >= version.parse("1.13.0")
Work = dist.Work if torch_version_above_1_13 else dist._Work
default_pg_timeout = datetime.timedelta(minutes=10)

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
        assert num_gpus_per_model * data_parallel_size == world_size, (
            "The number of process requires to train all replicas",
            "must be equal to the world size.",
        )

        if not dist.is_available():
            raise ValueError("`torch.distributed is not available as a package, please install it.")

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size

        self._groups = {}

        self.set_device()

        if not dist.is_initialized():
            rank = int(os.environ["RANK"])
            host = os.environ["MASTER_ADDR"]
            # TODO(xrsrke): make it auto search for ports?
            port = int(os.environ["MASTER_PORT"])
            self.init_global_dist(rank, world_size, backend, host, port)

        self._init_parallel_groups()

    def init_global_dist(self, rank: int, world_size: int, backend: DistributedBackend, host: str, port: int):
        """Initialize the global distributed group.

        Args:
            rank (int): global rank
            world_size (int): global world size
            backend (DistributedBackend): distributed backend
            host (str): communication host
            port (int): communication port
        """
        assert backend == "nccl", "Only nccl backend is supported for now."

        init_method = f"tcp://{host}:{port}"
        dist.init_process_group(
            rank=rank, world_size=world_size, backend=backend, init_method=init_method, timeout=dist.default_pg_timeout
        )
        ranks = list(range(world_size))
        process_group = dist.new_group(
            ranks=ranks,
            backend=dist.get_backend(),
        )
        self.world_pg = process_group

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


def new_group(  # pylint: disable=function-redefined
    ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None
) -> ProcessGroup:
    if len(ranks) == 0:
        raise ValueError("Cannot create a group with not ranks inside it")

    return dist.new_group(ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options)


def reduce_scatter_tensor(  # pylint: disable=function-redefined
    output: torch.Tensor,
    input: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[Work]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    assert (
        group.size() > 1
    ), "You should probably not call `reduce_scatter_tensor` with a single rank, as it copies data over"

    if torch_version_above_1_13:
        return dist.reduce_scatter_tensor(output=output, input=input, group=group, op=op, async_op=async_op)
    else:
        # Support pytorch 1.12
        return dist._reduce_scatter_base(output=output, input=input, group=group, op=op, async_op=async_op)


def all_gather_into_tensor(  # pylint: disable=function-redefined
    output_tensor, input_tensor, group: Optional[ProcessGroup] = None, async_op: bool = False
) -> Optional[Work]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    assert (
        group.size() > 1
    ), "You should probably not call `all_gather_into_tensor` with a single rank, as it copies data over"

    if torch_version_above_1_13:
        return dist.all_gather_into_tensor(
            output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
        )
    else:
        # Support Pytorch 1.12
        return dist.distributed_c10d._all_gather_base(
            output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
        )


def reduce_scatter_coalesced(
    output_tensor_list: List[torch.Tensor],
    input_tensor_lists: List[List[torch.Tensor]],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output_tensor_list (list[Tensor]): Output tensor.
        input_tensor_lists (list[list[Tensor]]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert len(output_tensor_list) > 0
    assert len(input_tensor_lists) == len(output_tensor_list)
    device = output_tensor_list[0].device
    dtype = output_tensor_list[0].dtype
    group_size = len(input_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `reduce_scatter_coalesced` with a single rank, as it copies data over"

    for output_tensor in output_tensor_list:
        assert device == output_tensor.device
        assert dtype == output_tensor.dtype

    for input_tensor_list in input_tensor_lists:
        assert len(input_tensor_list) == group_size, f"Expected {len(input_tensor_list)} == {group_size}"
        for input_tensor in input_tensor_list:
            assert device == input_tensor.device
            assert dtype == input_tensor.dtype

    output_tensor_buffer = torch._utils._flatten_dense_tensors(output_tensor_list)
    input_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(
            [input_tensor_list[group_rank] for input_tensor_list in input_tensor_lists]
        )
        for group_rank in range(group_size)
    ]

    work = dist.reduce_scatter(output_tensor_buffer, input_tensor_buffer_list, op=op, group=group, async_op=async_op)

    def update_output():
        for original_buffer, reduced_buffer in zip(
            output_tensor_list, torch._utils._unflatten_dense_tensors(output_tensor_buffer, output_tensor_list)
        ):
            original_buffer.copy_(reduced_buffer)

    if async_op is True:
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()


def all_reduce_coalesced(  # pylint: disable=function-redefined
    tensors: List[torch.Tensor],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    if group.size() == 1:
        return

    return dist.all_reduce_coalesced(tensors, op=op, group=group, async_op=async_op)


def all_gather_coalesced(  # pylint: disable=function-redefined
    output_tensor_lists: List[List[torch.Tensor]],
    input_tensor_list: List[torch.Tensor],
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """
    `torch` has a deprecated version of this method that doesn't work over NCCL.
    All gathers a list of tensors to all processes in a group.

    Args:
        output_tensor_lists (list[list[Tensor]]): Output tensor.
        input_tensor_list (list[Tensor]): List of tensors to all_gather from.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert len(output_tensor_lists) > 0
    assert len(input_tensor_list) == len(output_tensor_lists)
    device = input_tensor_list[0].device
    dtype = input_tensor_list[0].dtype
    group_size = len(output_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `all_gather_coalesced` with a single rank, as it copies data over"

    for input_tensor in input_tensor_list:
        assert device == input_tensor.device
        assert dtype == input_tensor.dtype

    for output_tensor_list in output_tensor_lists:
        assert len(output_tensor_list) == group_size
        for output_tensor in output_tensor_list:
            assert device == output_tensor.device
            assert dtype == output_tensor.dtype

    # Invert from `[param_idx][group_rank]` to `[group_rank][param_idx]`
    output_tensor_lists = [
        [output_tensor_list[group_rank] for output_tensor_list in output_tensor_lists]
        for group_rank in range(group_size)
    ]

    input_tensor_buffer = torch._utils._flatten_dense_tensors(input_tensor_list)
    output_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(output_tensor_list) for output_tensor_list in output_tensor_lists
    ]

    work = dist.all_gather(output_tensor_buffer_list, input_tensor_buffer, group=group, async_op=async_op)

    def update_output():
        for original_buffer_list, gathered_buffer_tensor in zip(output_tensor_lists, output_tensor_buffer_list):
            for original_buffer, gathered_buffer in zip(
                original_buffer_list,
                torch._utils._unflatten_dense_tensors(gathered_buffer_tensor, original_buffer_list),
            ):
                original_buffer.copy_(gathered_buffer)

    if async_op is True:
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()


# This cache has a speedup of 4 tflops on a 7b model
@cache
def get_global_rank(group: ProcessGroup, group_rank: int) -> int:  # pylint: disable=function-redefined
    if torch_version_above_1_13:
        return dist.get_global_rank(group, group_rank=group_rank)
    else:
        # Support pytorch 1.12
        return dist.distributed_c10d._get_global_rank(group=group, rank=group_rank)


# We cache for dp, pp, tp process groups, world group, and tied process group for tied params
@lru_cache
def get_rank(group: Optional[ProcessGroup] = None) -> int:  # pylint: disable=function-redefined
    """Similar to `get_rank` except we raise an exception instead of return -1 when current rank is not part of the group"""
    result = dist.get_rank(group)
    if result == -1:
        raise RuntimeError("Can not call `get_rank` on a group in which current process is not a part of")
    return result
