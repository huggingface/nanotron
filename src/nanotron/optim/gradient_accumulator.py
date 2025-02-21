import dataclasses
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
from torch.distributed import GradBucket

import nanotron.distributed as dist
from nanotron import logging
from nanotron.parallel.parameters import NanotronParameter
from nanotron.utils import get_untyped_storage, tensor_from_untyped_storage

logger = logging.get_logger(__name__)


class GradientAccumulator(ABC):
    fp32_grads_allreduce_handle: Optional[torch.futures.Future]

    @abstractmethod
    def __init__(self, named_parameters: Iterator[Tuple[str, NanotronParameter]]):
        ...

    @abstractmethod
    def backward(self, loss: torch.Tensor):
        ...

    @abstractmethod
    def step(self):
        ...

    @abstractmethod
    def sync_gradients_across_dp(self, dp_pg: dist.ProcessGroup, reduce_op: dist.ReduceOp, reduce_scatter: bool):
        ...

    @abstractmethod
    def zero_grad(self):
        ...

    @abstractmethod
    def get_parameter_for_optimizer(self, name: str) -> NanotronParameter:
        ...

    @abstractmethod
    def get_grad_buffer(self, name: str) -> torch.Tensor:
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: torch.Tensor):
        ...


class FP32GradientAccumulator(GradientAccumulator):
    def __init__(
        self,
        named_parameters: Iterator[Tuple[str, NanotronParameter]],
        grad_buckets_named_params: Optional[Iterator[Tuple[str, NanotronParameter]]] = None,
    ):
        """Create a gradient accumulator that will accumulate gradients in fp32.

        Args:
            named_parameters: The parameters that will be updated by the optimizer. In case of Zero 1, this is the parameters that will be updated in this DP rank.
            grad_buckets_named_params: The parameters to accumulate gradients for. If None it defaults to `named_parameters`. In case of Zero 1, this should be all the parameters in the model.

        Note: We use `grad_buckets_named_params` to keep grad buffers for all parameters even when Zero 1 is used. This is because we need to accumulate gradients for all parameters without having to reduce in every accumulation step.
        Note: We make a fp32 copy of parameters during initialization. Therefore parameters need to be initialized or loaded from a checkpoint before constructing this gradient accumulator
        """
        if grad_buckets_named_params is None:
            named_parameters = list(named_parameters)
            grad_buckets_named_params = named_parameters

        # Initialize grad bucket
        self.fp32_grad_buffers, self._contiguous_fp32_grad_buffer = self.build_grad_buffers(
            named_parameters=grad_buckets_named_params
        )

        # Assign big buffer for weights + grad in fp32
        segment_index = {}
        length = 0
        for name, param in named_parameters:
            if not param.requires_grad:
                continue

            start = length
            end_weight = start + param.numel()
            assert name not in segment_index
            segment_index[name] = (start, end_weight, param)
            length = end_weight

        big_flat_buffer = torch.empty(length, dtype=torch.float, device="cuda")
        self.parameters = {
            name: {
                "fp32": big_flat_buffer[start_weight:end_weight].view_as(param),
                "half": param,
            }
            for name, (start_weight, end_weight, param) in segment_index.items()
        }

        with torch.inference_mode():
            for _, elt in self.parameters.items():
                fp32_param = elt["fp32"]
                half_param = elt["half"]

                # Check that fp32 weights have the same memory representation as half precision weights
                assert fp32_param.stride() == half_param.stride()

                # Copy weights from half precision to full precision
                fp32_param.copy_(half_param)

                # Set requires_grad=True
                fp32_param.requires_grad = True

        self._is_accumulation_sync_step = False
        # We need the last allreduce handle to make sure it finishes before the optimizer step
        self.fp32_grads_allreduce_handle: Optional[torch.futures.Future] = None

    def assign_param_offsets(self, param_name_to_offsets: Dict[str, Dict[int, Tuple[int, int]]], dp_rank: int):
        """To use only when you use with ZeRODistributedOptimizer"""
        self.param_name_to_offsets = {
            name: elt[dp_rank] for name, elt in param_name_to_offsets.items() if dp_rank in elt
        }

    def sync_gradients_across_dp(self, dp_pg: dist.ProcessGroup, reduce_op: dist.ReduceOp, reduce_scatter: bool):
        if dp_pg.size() == 1:
            # They are already synced
            return

        if reduce_scatter:
            # Usually you need to run `all_reduce` in order for all gradients to be synced.
            # However when the optimizer state are sharded, you really just need to scatter to ranks that are going to run the optimizer state.
            # Effectively you replace a `all_reduce` with a `reduce_scatter` which should save an `all_gather` when using RING algorithm.
            assert hasattr(self, "param_name_to_offsets")
            named_offsets = sorted(self.param_name_to_offsets.items(), key=lambda x: x[0])
            flat_grad_buffers = [self.fp32_grad_buffers[name]["fp32_grad"].view(-1) for name, _ in named_offsets]
            dist.reduce_scatter_coalesced(
                output_tensor_list=[
                    flat_grad_buffer[start_offset:end_offset]
                    for (_, (start_offset, end_offset)), flat_grad_buffer in zip(named_offsets, flat_grad_buffers)
                ],
                input_tensor_lists=[
                    torch.split(
                        flat_grad_buffer,
                        split_size_or_sections=len(self.fp32_grad_buffers[name]["fp32_grad"].view(-1)) // dp_pg.size(),
                    )
                    for (name, _), flat_grad_buffer in zip(named_offsets, flat_grad_buffers)
                ],
                group=dp_pg,
            )
        else:
            dist.all_reduce(self._contiguous_fp32_grad_buffer, op=reduce_op, group=dp_pg)

    @staticmethod
    def build_grad_buffers(
        named_parameters: Iterator[Tuple[str, NanotronParameter]],
    ) -> Tuple[Dict[str, Dict], torch.Tensor]:
        """Builds grad buffers for all model's parameters, independently of ZeRO sharding

        Args:
            named_parameters: Parameters to build buckets for. In case of Zero1, this should be all parameters.

        Note:
            In ZeRO-1, we need to accumulate grads for all parameters, because we need to allreduce all parameters' grads across DP at each sync step.
        """
        named_parameters = [(name, param) for name, param in named_parameters if param.requires_grad]

        needed_buffer_size = sum(param.numel() for _, param in named_parameters)
        # important to have grads zeroed initially (see `self._accumulate_grad`)
        contiguous_buffer_f32_gradients = torch.zeros(needed_buffer_size, dtype=torch.float, device="cuda")
        untyped_storage = get_untyped_storage(contiguous_buffer_f32_gradients)
        element_size = contiguous_buffer_f32_gradients.element_size()

        # NOTE: Although `bias` can only exist on TP=0. It shouldn't be a problem here, because we only sync across DP
        fp32_grad_buffers = OrderedDict()  # keeps order of insertion
        offset = 0
        for name, param in named_parameters:
            if not param.requires_grad:
                continue

            assert param.dtype != torch.float, f"Expected {name} not to be float"
            assert param.is_contiguous(), f"Expected {name} to be contiguous"

            next_offset = offset + param.numel() * element_size

            fp32_grad_buffer = tensor_from_untyped_storage(
                untyped_storage=untyped_storage[offset:next_offset], dtype=torch.float
            )

            fp32_grad_buffers[name] = {
                "half": param,
                # We create sliced tensors by also slicing storage.
                # We need to specify "cuda" in order to share the same data storage, otherwise it build the tensor in "cpu" and copies over the data
                "fp32_grad": fp32_grad_buffer.view_as(param),
            }

            offset = next_offset

        return fp32_grad_buffers, contiguous_buffer_f32_gradients

    def backward(self, loss: torch.Tensor):
        result = loss.backward()

        for name, elt in self.fp32_grad_buffers.items():
            self._accumulate_grad(name=name, half_param=elt["half"])

        return result

    def _accumulate_grad(self, name: str, half_param: NanotronParameter) -> None:
        """Accumulate grad in fp32 and set the fp32 grad to the fp32 grad buffer, so that optimizer can update fp32 weights afterwards"""
        assert half_param.grad is not None, f"Expected param {name} to have gradient."
        fp32_grad = self.get_grad_buffer(name=name)

        if self._is_accumulation_sync_step is False:
            # WARNING: We assume fp32_grad_bucket is already zeroed
            fp32_grad.add_(half_param.grad)
            # In case _is_accumulation_sync_step = True: no need to add half gradients, because it's done in the allreduce hook

        # TODO @thomasw21: Is it better to set to zero instead?
        half_param.grad = None

        # In the case an optimizer decides to set it to None, we need to re-assign previous buffer
        if name in self.parameters:
            fp32_param = self.parameters[name]["fp32"]
            if hasattr(self, "param_name_to_offsets"):
                if name not in self.param_name_to_offsets:
                    # When `name` isn't in `param_name_to_offsets` it means the slice is empty.
                    return
                start_offset, end_offset = self.param_name_to_offsets[name]
                grad = fp32_grad.view(-1)[start_offset:end_offset]
            else:
                grad = fp32_grad
            fp32_param.grad = grad

    @contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronizations across
        data-parallel ranks.

        Note: if we use `no_sync` once, that means we're in DDP mode, and we switch the default of self._is_accumulation_sync_step to True.
        """
        old_is_accumulation_sync_step = self._is_accumulation_sync_step
        self._is_accumulation_sync_step = False
        try:
            yield
        finally:
            self._is_accumulation_sync_step = old_is_accumulation_sync_step

    @torch.inference_mode()
    def step(self):
        """Updates fp32 weights from fp32 grads.
        In case where OptimizerFromGradientAccumulator and gradient_accumulator_builder are using different parameters (e.g ZeRO).
        We need to update only the parameters that were updated by the optimizer.
        """
        for name in self.parameters.keys():
            fp32_param = self.parameters[name]["fp32"]
            half_param = self.parameters[name]["half"]
            # TODO @nouamane: should we use a fused kernel to copy?
            # Copy weights from full precision to half precision
            half_param.copy_(fp32_param)

    def zero_grad(self):
        # Full precision gradients are reset to zero/none after the underlying `optimiser.step`, so no need to reset.
        for elt in self.fp32_grad_buffers.values():
            half_param = elt["half"]

            if half_param.grad is None:
                continue

            half_param.grad = None

        # in case where self.parameters and self.fp32_grad_buffers are not the same (e.g we want to accumulate all DPs grads, and only sync at sync step)
        self._contiguous_fp32_grad_buffer.zero_()

    def get_parameter_for_optimizer(self, name: str) -> NanotronParameter:
        return self.parameters[name]["fp32"]

    def get_grad_buffer(self, name: str) -> torch.Tensor:
        """Returns the gradient of the parameter from the appropriate grad bucket."""
        return self.fp32_grad_buffers[name]["fp32_grad"]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # We consider `fp32` parameters as a state of the gradient accumulator
        return {name: elt["fp32"] for name, elt in self.parameters.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        assert set(state_dict.keys()) == set(self.parameters.keys())

        with torch.inference_mode():
            for name, elt in self.parameters.items():
                elt["fp32"].copy_(state_dict[name])


@dataclasses.dataclass
class FP32GradBucketManager:
    """Manages the fp32 gradient buckets.

    Attributes:
        dp_pg: The process group to allreduce gradients across.
        accumulator: The gradient accumulator which keeps the gradient buffers.
        bucket_id_to_fp32_grad_buckets_and_dependencies: A dictionary mapping bucket ids to:
            - fp32 grad bucket (torch.Tensor)
            - set of param ids that are in the bucket -> used to know when to delete the buffer
        param_id_to_bucket_id: A dictionary mapping param ids to bucket ids."""

    dp_pg: dist.ProcessGroup
    accumulator: FP32GradientAccumulator
    param_id_to_name: Dict[int, str]

    def __post_init__(self):
        self.accumulator._is_accumulation_sync_step = True


def get_fp32_accum_hook(
    reduce_scatter: bool,
    reduce_op: dist.ReduceOp = dist.ReduceOp.AVG,
) -> Callable:
    """Returns a DDP communication hook that performs gradient accumulation in fp32.

    Args:
        reduce_op: The reduction operation to perform.
    """
    # s = torch.cuda.Stream()

    def fp32_accum_hook(state: FP32GradBucketManager, bucket: GradBucket) -> torch.futures.Future[torch.Tensor]:
        # nonlocal s
        # DDP groups grads in GradBuckets. This hook is called throughout the bwd pass, once each bucket is ready to overlap communication with computation.
        # See https://pytorch.org/docs/stable/ddp_comm_hooks.html#what-does-a-communication-hook-operate-on for more details.
        dp_pg = state.dp_pg
        accumulator = state.accumulator
        param_id_to_name = state.param_id_to_name

        # Add new incoming gradient
        # with torch.cuda.stream(s):
        for param, grad in zip(bucket.parameters(), bucket.gradients()):
            name = param_id_to_name[id(param)]
            fp32_grad_buffer = accumulator.get_grad_buffer(name)
            fp32_grad_buffer.add_(grad.view_as(fp32_grad_buffer))

        # sync across dp
        if dp_pg.size() == 1:
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        if reduce_scatter:
            assert hasattr(accumulator, "param_name_to_offsets")
            grad_buffer_tensor_list = [
                accumulator.get_grad_buffer(param_id_to_name[id(param)]).view(-1) for param in bucket.parameters()
            ]
            device = grad_buffer_tensor_list[0].device
            dtype = grad_buffer_tensor_list[0].dtype
            output_tensor_list = [
                grad_buffer[slice(*accumulator.param_name_to_offsets[param_id_to_name[id(param)]])]
                if param_id_to_name[id(param)] in accumulator.param_name_to_offsets
                else torch.empty(0, dtype=dtype, device=device)
                for grad_buffer, param in zip(grad_buffer_tensor_list, bucket.parameters())
            ]
            input_tensor_lists = [
                torch.split(grad_buffer, split_size_or_sections=len(grad_buffer) // dp_pg.size())
                for grad_buffer in grad_buffer_tensor_list
            ]
            dist.reduce_scatter_coalesced(
                output_tensor_list=output_tensor_list,
                input_tensor_lists=input_tensor_lists,
                op=reduce_op,
                group=dp_pg,
                async_op=True,
            )
        else:
            grad_buffer_tensor_list = [
                accumulator.get_grad_buffer(param_id_to_name[id(param)]).view(-1) for param in bucket.parameters()
            ]
            accumulator.fp32_grads_allreduce_handle = dist.all_reduce_coalesced(
                grad_buffer_tensor_list, group=dp_pg, async_op=True, op=reduce_op
            )
            # we shouldn't wait for this future for the rest of the backward

        # with torch.cuda.stream(s):
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        half_grad_bucket = bucket.buffer()
        fut.set_result(half_grad_bucket)
        return fut  # We don't care about the new half grad values, so we return the old ones

    return fp32_accum_hook
