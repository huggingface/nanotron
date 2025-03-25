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
    def get_global_grad_buffer(self, name: str) -> torch.Tensor:
        ...

    @abstractmethod
    def get_local_grad_buffer(self, name: str) -> torch.Tensor:
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
        local_named_params_f16: Iterator[Tuple[str, NanotronParameter]],
        global_named_params_f16: Optional[Iterator[Tuple[str, NanotronParameter]]] = None,
        param_name_to_dp_rank_offsets: Optional[Dict[str, Dict[int, Tuple[int, int]]]] = None,
        dp_pg: Optional[dist.ProcessGroup] = None,
    ):
        """Create a gradient accumulator that will accumulate gradients in fp32.

        Args:
            local_named_params_f16: The parameters that will be updated by the optimizer. In case of Zero 1, this is the parameters that will be updated in this DP rank.
            global_named_params_f16: (Global in DP sense) The parameters to accumulate gradients for. If None it defaults to `local_named_params_f16`. In case of Zero 1, this should be all the parameters in the model.
            param_name_to_dp_rank_offsets: Mapping from parameter name to the DP rank offsets. Only used when using ZeRO-1 or higher.
            dp_pg: Data parallel process group. Required when using ZeRO-1 or higher for proper gradient buffer padding.

        Note: We use `global_named_params_f16` to keep grad buffers for all parameters even when Zero 1 is used. This is because we need to accumulate gradients for all parameters without having to reduce in every accumulation step.
        Note: We make a fp32 copy of parameters during initialization. Therefore parameters need to be initialized or loaded from a checkpoint before constructing this gradient accumulator
        """
        if global_named_params_f16 is None:
            local_named_params_f16 = list(local_named_params_f16)
            global_named_params_f16 = local_named_params_f16

        # Keep only parameters that require grad
        global_named_params_f16 = [(name, param) for name, param in global_named_params_f16 if param.requires_grad]
        local_named_params_f16 = [(name, param) for name, param in local_named_params_f16 if param.requires_grad]

        if dp_pg is not None:
            dp_size, dp_rank = dp_pg.size(), dist.get_rank(dp_pg)
        else:
            dp_size, dp_rank = 1, 0

        # Initialize grad bucket with proper dp padding if using ZeRO (numel = total_global_numel + dp_padding)
        self.global_fp32_grad_buffers, self._contiguous_fp32_grad_buffer = self.build_grad_buffers(
            global_named_params_f16=global_named_params_f16,
            dp_size=dp_size,
        )

        # Calculate total size needed for local parameters
        total_local_numel = sum(param.numel() for _, param in local_named_params_f16)

        # Create single contiguous buffer for all local fp32 parameters (numel = total_local_numel)
        self.local_fp32_params_buffer = torch.empty(total_local_numel, dtype=torch.float, device="cuda")

        # Create views into the buffer for each parameter
        self.local_params = {}
        offset = 0
        for name, param in local_named_params_f16:
            param_numel = param.numel()
            # SANITY CHECK: Get the DP rank offsets for this parameter
            if param_name_to_dp_rank_offsets is not None:
                start_offset, end_offset = param_name_to_dp_rank_offsets[name][dp_rank]
            else:
                start_offset, end_offset = 0, param.numel()
            assert (
                param_numel == end_offset - start_offset
            ), f"Expected param {name} to have {param_numel} elements on this DP rank, but got {end_offset - start_offset}"

            next_offset = offset + param_numel

            # Create view into the contiguous buffer
            assert next_offset <= self.local_fp32_params_buffer.numel()
            fp32_param = self.local_fp32_params_buffer[offset:next_offset].view_as(param)

            self.local_params[name] = {
                "fp32": fp32_param,  # View into contiguous fp32 buffer
                "half": param,  # Original half precision parameter
            }
            offset = next_offset

        # Initialize fp32 parameters from half parameters
        with torch.inference_mode():
            # copy stack of half params into fp32 params buffer
            half_params = torch.cat([elt["half"] for elt in self.local_params.values()])
            self.local_fp32_params_buffer[: half_params.numel()].copy_(half_params)  # last dp rank has padding

            for name, elt in self.local_params.items():
                fp32_param = elt["fp32"]
                half_param = elt["half"]
                # Check that fp32 weights have the same memory representation as half precision weights
                assert fp32_param.stride() == half_param.stride()
                torch.testing.assert_close(fp32_param, half_param.float())
                fp32_param.requires_grad = True

        self._is_accumulation_sync_step = False
        # We need the last allreduce handle to make sure it finishes before the optimizer step
        self.fp32_grads_allreduce_handle: Optional[torch.futures.Future] = None
        # Reduce scatter buffer because `_contiguous_fp32_grad_buffer` tries to keep fp32_grads contiguous in memory, while we need [grad0_dp0, grad1_dp0, .., grad0_dp1, grad1_dp1, ..]
        # self.reduce_scatter_buffer: Optional[torch.Tensor] = None

        # Zero>=1 attributes
        self.param_name_to_dp_rank_offsets = param_name_to_dp_rank_offsets
        self.dp_rank = dp_rank
        self.dp_size = dp_size

        # SANITY CHECK: set other ranks' portions to nan or infinity
        # Get current rank's slice of the contiguous buffer
        current_rank = self.dp_rank
        assert (
            len(self._contiguous_fp32_grad_buffer) % self.dp_size == 0
        ), "Contiguous buffer must be divisible by dp_size"
        chunk_size = len(self._contiguous_fp32_grad_buffer) // self.dp_size
        start_idx = current_rank * chunk_size

        # SANITY CHECK: set other ranks' portions to nan or infinity
        if start_idx > 0:  # TODO: use config for this
            self._contiguous_fp32_grad_buffer[:start_idx].fill_(float("nan"))
        if start_idx + chunk_size < len(self._contiguous_fp32_grad_buffer):
            self._contiguous_fp32_grad_buffer[start_idx + chunk_size :].fill_(float("nan"))

        # name ="model.decoder.11.pp_block.mlp.gate_up_proj.weight" # problematic param
        # # assert local grads have no nans
        # assert not self.get_local_grad_buffer(name).isnan().any()

        # # assert (list(grad_accumulator.global_fp32_grad_buffers.items()))[71] == global ranks have nans (because it's mixed)
        # assert self.get_global_grad_buffer(name).isnan().any()

        # loop over local params and check that grads are not nan
        for name, elt in self.local_params.items():
            assert not self.get_local_grad_buffer(name).isnan().any(), f"Local grad for {name} is nan"

        self._contiguous_fp32_grad_buffer.zero_()

    def assign_param_offsets(self, param_name_to_offsets: Dict[str, Dict[int, Tuple[int, int]]], dp_rank: int):
        """To use only when you use with ZeRODistributedOptimizer"""
        # self.param_name_to_offsets = {
        #     name: elt[dp_rank] for name, elt in param_name_to_offsets.items() if dp_rank in elt
        # }

    def sync_gradients_across_dp(self, dp_pg: dist.ProcessGroup, reduce_op: dist.ReduceOp, reduce_scatter: bool):
        if dp_pg.size() == 1:
            # They are already synced
            return

        if reduce_scatter:
            # Usually you need to run `all_reduce` in order for all gradients to be synced.
            # However when the optimizer state are sharded, you really just need to scatter to ranks that are going to run the optimizer state.
            # Effectively you replace a `all_reduce` with a `reduce_scatter` which should save an `all_gather` when using RING algorithm.
            assert (
                self.param_name_to_dp_rank_offsets is not None
            ), "Need param_name_to_dp_rank_offsets for reduce_scatter"

            # Get current rank's slice of the contiguous buffer
            current_rank = self.dp_rank
            assert (
                len(self._contiguous_fp32_grad_buffer) % self.dp_size == 0
            ), "Contiguous buffer must be divisible by dp_size"
            chunk_size = len(self._contiguous_fp32_grad_buffer) // self.dp_size
            start_idx = current_rank * chunk_size

            # Single reduce_scatter operation on the contiguous buffer
            dist.reduce_scatter_tensor(
                output=self._contiguous_fp32_grad_buffer[
                    start_idx : start_idx + chunk_size
                ],  # This rank's slice of the buffer
                input=self._contiguous_fp32_grad_buffer,  # Already in [dp0_grads, dp1_grads, ...] layout
                op=reduce_op,
                group=dp_pg,
            )

            # SANITY CHECK: set other ranks' portions to nan or infinity
            # if start_idx > 0: #TODO: use config for this
            #     self._contiguous_fp32_grad_buffer[:start_idx].fill_(float('nan'))
            # if start_idx + chunk_size < len(self._contiguous_fp32_grad_buffer):
            #     self._contiguous_fp32_grad_buffer[start_idx + chunk_size:].fill_(float('nan'))

        else:
            # If not using reduce_scatter, perform regular all_reduce
            dist.all_reduce(
                self._contiguous_fp32_grad_buffer,
                op=reduce_op,
                group=dp_pg,
            )

    @staticmethod
    def build_grad_buffers(
        global_named_params_f16: Iterator[Tuple[str, NanotronParameter]],
        dp_size: int,
    ) -> Tuple[Dict[str, Dict], torch.Tensor]:
        """Builds grad buffers for all model's parameters, independently of ZeRO sharding.
        Creates a contiguous buffer padded to be multiple of dp_size.

        Args:
            global_named_params_f16: Parameters to build buckets for. In case of Zero1, this should be all parameters.

        Returns:
            Tuple of:
            - Dictionary mapping param names to their half and fp32 gradient buffers
            - The contiguous fp32 buffer containing all gradients
        """
        # Calculate total size needed and padding
        total_numel = sum(param.numel() for _, param in global_named_params_f16)

        # Calculate padding to make total_numel a multiple of dp_size
        padded_numel = (total_numel + dp_size - 1) // dp_size * dp_size

        # Create contiguous buffer (padded to be multiple of dp_size, so that we can reduce scatter across DP). We still need grads for all dp ranks to accumulate f16 grads.
        # [grad0_dp0, grad1_dp0, ..., grad5_dp0, grad5_dp1, grad6_dp1, ...] where last DP rank gets the remainder
        contiguous_buffer_f32_gradients = torch.zeros(padded_numel, dtype=torch.float, device="cuda")
        untyped_storage = get_untyped_storage(contiguous_buffer_f32_gradients)
        element_size = contiguous_buffer_f32_gradients.element_size()

        # Create gradient buffers dictionary
        global_fp32_grad_buffers = OrderedDict()  # keeps order of insertion
        offset = 0

        for name, param in global_named_params_f16:
            assert param.dtype != torch.float, f"Expected {name} not to be float"
            assert param.is_contiguous(), f"Expected {name} to be contiguous"

            param_numel = param.numel()
            next_offset = offset + param_numel * element_size

            # Create view into the contiguous buffer for this parameter
            fp32_grad_buffer = tensor_from_untyped_storage(
                untyped_storage=untyped_storage[offset:next_offset], dtype=torch.float
            )

            global_fp32_grad_buffers[name] = {
                "half": param,  # Original half precision parameter
                "fp32_grad": fp32_grad_buffer.view_as(param),  # View into contiguous_buffer_f32_gradients
            }

            offset = next_offset

        return global_fp32_grad_buffers, contiguous_buffer_f32_gradients

    def backward(self, loss: torch.Tensor):
        result = loss.backward()

        for name, elt in self.global_fp32_grad_buffers.items():
            self._accumulate_grad(name=name, half_param=elt["half"])

        return result

    def _accumulate_grad(self, name: str, half_param: NanotronParameter) -> None:
        """Accumulate grad in fp32 and set the fp32 grad to the fp32 grad buffer, so that optimizer can update fp32 weights afterwards"""
        assert half_param.grad is not None, f"Expected param {name} to have gradient."
        global_fp32_grad = self.get_global_grad_buffer(name=name)

        if self._is_accumulation_sync_step is False:
            # WARNING: We assume fp32_grad_bucket is already zeroed
            global_fp32_grad.add_(half_param.grad)
            # In case _is_accumulation_sync_step = True: no need to add half gradients, because it's done in the allreduce hook

        # TODO @thomasw21: Is it better to set to zero instead?
        half_param.grad = None

        # In the case an optimizer decides to set it to None, we need to re-assign previous buffer
        if name in self.local_params:
            fp32_param = self.local_params[name]["fp32"]
            grad = self.get_local_grad_buffer(name)
            assert (
                fp32_param.shape == grad.shape
            ), f"Expected grad of {name} to have shape {fp32_param.shape} but got {grad.shape}"
            fp32_param.grad = grad  # TODO: is this needed?

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

    # @torch.compile(fullgraph=True) # TODO: UserDefinedObjectVariable(SlicedFlatTensor)
    @torch.inference_mode()
    def step(self):
        """Updates fp32 weights from fp32 grads.
        In case where OptimizerFromGradientAccumulator and gradient_accumulator_builder are using different parameters (e.g ZeRO).
        We need to update only the parameters that were updated by the optimizer.
        """
        for name in self.local_params.keys():
            # Update the local shard
            fp32_param = self.local_params[name]["fp32"]
            half_param = self.local_params[name]["half"]
            # Copy weights from full precision to half precision
            half_param.copy_(fp32_param)
        # half_params = torch.cat([elt["half"] for elt in self.local_params.values()]) # TODO: this does a copy and doesnt work
        # half_params.copy_(self.local_fp32_params_buffer[: half_params.numel()])  # last dp rank has padding
        # check that first half param was updated
        assert torch.allclose(
            self.local_params["model.decoder.11.pp_block.mlp.gate_up_proj.weight"]["half"],
            self.local_params["model.decoder.11.pp_block.mlp.gate_up_proj.weight"]["fp32"].bfloat16(),
        )

    def zero_grad(self):
        # Full precision gradients are reset to zero/none after the underlying `optimiser.step`, so no need to reset.
        for elt in self.global_fp32_grad_buffers.values():
            half_param = elt["half"]
            half_param.grad = None

        # in case where self.local_params and self.global_fp32_grad_buffers are not the same (e.g we want to accumulate all DPs grads, and only sync at sync step)
        self._contiguous_fp32_grad_buffer.zero_()

    def get_parameter_for_optimizer(self, name: str) -> NanotronParameter:
        return self.local_params[name]["fp32"]

    def get_global_grad_buffer(self, name: str) -> torch.Tensor:
        """Returns the gradient of the parameter from the appropriate grad bucket."""
        return self.global_fp32_grad_buffers[name]["fp32_grad"]

    def get_local_grad_buffer(self, name: str) -> torch.Tensor:
        """Returns the gradient of the parameter from the appropriate grad bucket."""
        if self.param_name_to_dp_rank_offsets is None:
            # zero0 case
            return self.global_fp32_grad_buffers[name]["fp32_grad"]
        else:
            # zero1 case
            if name not in self.local_params:
                return None
            start_offset, end_offset = self.param_name_to_dp_rank_offsets[name][self.dp_rank]
            return self.global_fp32_grad_buffers[name]["fp32_grad"].view(-1)[start_offset:end_offset]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # We consider `fp32` parameters as a state of the gradient accumulator
        return {name: elt["fp32"] for name, elt in self.local_params.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        assert set(state_dict.keys()) == set(self.local_params.keys())

        with torch.inference_mode():
            for name, elt in self.local_params.items():
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
            fp32_grad_buffer = accumulator.get_local_grad_buffer(name)
            fp32_grad_buffer.add_(grad.view_as(fp32_grad_buffer))

        # sync across dp
        if dp_pg.size() == 1:
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        if reduce_scatter:
            assert accumulator.param_name_to_dp_rank_offsets is not None
            grad_buffer_tensor_list = [
                accumulator.get_local_grad_buffer(param_id_to_name[id(param)]).view(-1)
                for param in bucket.parameters()
            ]
            device = grad_buffer_tensor_list[0].device
            dtype = grad_buffer_tensor_list[0].dtype
            output_tensor_list = [
                grad_buffer[
                    slice(*accumulator.param_name_to_dp_rank_offsets[param_id_to_name[id(param)]][accumulator.dp_rank])
                ]
                if param_id_to_name[id(param)] in accumulator.param_name_to_dp_rank_offsets
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
                accumulator.get_local_grad_buffer(param_id_to_name[id(param)]).view(-1)
                for param in bucket.parameters()
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
