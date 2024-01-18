from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import torch
from nanotron import distributed as dist
from nanotron.parallel.pipeline_parallel.functional import (
    recv_from_pipeline_state_buffer,
    send_to_pipeline_state_buffer,
)
from nanotron.parallel.pipeline_parallel.p2p import P2P, BatchTensorSendRecvState
from nanotron.parallel.pipeline_parallel.state import PipelineBatchState, PipelineTrainBatchState
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from torch import nn


class PipelineBlock(nn.Module):
    """Most granular pipeline block, ie within this module, everything will be part of a single rank, ie the entire computation within this block will happen on a specific device.

    Current limitations:
     - PipelineBlocks have to wrap a method/function/module that outputs a Dict[str, torch.Tensor]

    Some considerations:
     - In the litterature, authors often refer to pipeline stages as a granularity block. Our notion is more granular. A pipeline stage is list of contiguous (in the forward sense) of pipeline blocks.
    All PipelineBlock definition exist in each rank, they are just instantiated/built on a single rank per pipeline parallel process group.
    """

    def __init__(
        self,
        p2p: P2P,
        module_builder: Callable[..., Callable[..., Union[torch.Tensor, Dict[str, torch.Tensor]]]],
        module_kwargs: Dict[str, Any],
        module_input_keys: Set[str],
        module_output_keys: Set[str],
    ):
        super().__init__()
        # Module follows a restrictive API: module.forward return a `Dict[str, torch.Tensor]`
        self.p2p = p2p
        # None signifies that we don't use specific pipeline engine and just run typical torch forward/backward pass
        self.pipeline_state: Optional[PipelineBatchState] = None

        self.module_builder = module_builder
        self.module_kwargs = module_kwargs
        self.module_input_keys = set(module_input_keys)
        self.module_output_keys = set(module_output_keys)

    def build_and_set_rank(self, pp_rank: int):
        """This method is used to define on which rank computation is going to happen"""
        assert pp_rank < self.p2p.pg.size()
        self.rank = pp_rank
        if pp_rank == dist.get_rank(self.p2p.pg):
            # Instantiate the module
            self.pp_block = self.module_builder(**self.module_kwargs)

    def extra_repr(self) -> str:
        return f"pp_rank={self.rank}" if hasattr(self, "rank") is not None else ""

    def set_pipeline_state(self, pipeline_state: Optional[PipelineBatchState]):
        self.pipeline_state = pipeline_state

    def forward(self, **kwargs):
        """Forward pass

        We use a mechanism using TensorPointers to pass Tensors around
        All non Tensor object or TensorPointers are considered pass-through, they are never meant to be communicated cross process

        :param kwargs: Dict[str, Union[TensorPointer, torch.Tensor, Any]]
        :return: Dict[str, Union[TensorPointer, torch.Tensor, Any]
        """
        assert self.module_input_keys == set(
            kwargs.keys()
        ), f"Expected {self.module_input_keys}, got {set(kwargs.keys())}"

        sorted_kwargs = sorted(kwargs.items(), key=get_sort_key(dist.get_rank(self.p2p.pg)))

        # Is the current rank is not the one running the compute
        if dist.get_rank(self.p2p.pg) != self.rank:
            # TODO(kunhao): A better design is to pop this up for both if else branches.
            batch_send_recv = BatchTensorSendRecvState(self.p2p)
            # Send activations from other devices to local rank
            for name, tensor in sorted_kwargs:
                if isinstance(tensor, TensorPointer):
                    # Current rank is neither the rank holding the data nor the rank responsible for computing block
                    continue
                else:
                    assert isinstance(tensor, torch.Tensor)
                    # We need to send the tensor to the rank that actually runs the compute
                    if self.pipeline_state is not None:
                        send_to_pipeline_state_buffer(
                            tensor,
                            to_rank=self.rank,
                            p2p=self.p2p,
                            pipeline_state=self.pipeline_state,
                        )
                        continue

                    if tensor.requires_grad is True:
                        raise ValueError(
                            f"Pipeline engine is None and tensor requires grad. Tried sending a tensor to {self.rank}. Usually that means that your model is pipeline sharded and you haven't chosen a specific pipeline engine."
                        )

                    batch_send_recv.add_send(tensor=tensor, to_rank=self.rank)

            batch_send_recv.flush()
            # Return that the outputs are all in the rank responsible for computing block
            # TODO @thomasw21: Figure out a way to build dummy_input in a generic sense, and remove the necessity to have Dict[str, torch.Tensor] as output
            return {k: TensorPointer(group_rank=self.rank) for k in self.module_output_keys}

        # Recv activations from other devices to local rank
        new_kwargs: Dict[str, torch.Tensor] = {}
        name_to_recv_id = {}
        batch_send_recv = BatchTensorSendRecvState(self.p2p)
        for name, tensor in sorted_kwargs:
            if isinstance(tensor, TensorPointer):
                # Current rank is the one running the compute, we need to query the tensor
                # new_kwargs[name] = recv_tensor(from_rank=tensor.group_rank, p2p=self.p2p)
                # This assumes that prior communication was already done

                # In case of interleaved 1f1b, if this is the second model chunk, then we need to send the previous activations before receiving the current activations
                if isinstance(self.pipeline_state, PipelineTrainBatchState):
                    for _ in range(len(self.pipeline_state.microbatches_activations_to_send)):
                        send_activation = self.pipeline_state.microbatches_activations_to_send.popleft()
                        # Execute
                        send_activation()

                if self.pipeline_state is not None:
                    new_kwargs[name] = recv_from_pipeline_state_buffer(
                        from_rank=tensor.group_rank,
                        p2p=self.p2p,
                        pipeline_state=self.pipeline_state,
                    )
                    continue

                # We don't store result in a buffer
                recv_id = batch_send_recv.add_recv(from_rank=tensor.group_rank)
                name_to_recv_id[name] = recv_id
            else:
                new_kwargs[name] = tensor

        # Run receiving communications
        recv_tensors = batch_send_recv.flush()
        assert len(recv_tensors) == len(name_to_recv_id)
        for name, recv_id in name_to_recv_id.items():
            assert name not in new_kwargs
            new_tensor = recv_tensors[recv_id]
            if new_tensor.requires_grad is True:
                raise ValueError(
                    f"Pipeline engine is None and tensor requires grad. Tried receiving a tensor to {self.rank}. Usually that means that your model is pipeline sharded and you haven't chosen a specific pipeline engine."
                )
            new_kwargs[name] = new_tensor

        output = self.pp_block(**new_kwargs)

        # Helper for functions that return tensors
        if isinstance(output, torch.Tensor):
            assert len(self.module_output_keys) == 1
            output = {next(iter(self.module_output_keys)): output}

        assert isinstance(output, dict), "Modules within a Pipeline Block have to return a Dict[str, torch.Tensor]"
        assert self.module_output_keys == set(
            output.keys()
        ), f"Expected {self.module_output_keys}, got {set(output.keys())}"

        return output


def get_min_max_rank(module: torch.nn.Module) -> Tuple[int, int]:
    """Finds min and max PP ranks of the underlying PipelineBlocks"""
    ranks = [module.rank for module in module.modules() if isinstance(module, PipelineBlock)]
    return min(ranks), max(ranks)


def get_sort_key(current_rank: int):
    """The idea is to free earlier ranks earlier."""

    def sort_key(elt: Tuple[str, Union[torch.Tensor, TensorPointer]]):
        name, tensor = elt
        rank: int
        if isinstance(tensor, TensorPointer):
            rank = tensor.group_rank
        else:
            rank = current_rank
        return rank, name

    return sort_key
