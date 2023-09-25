from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch.optim
from functorch.dim import tree_map

from nanotron.core import distributed as dist
from nanotron.core import logging
from nanotron.core.distributed import ProcessGroup
from nanotron.core.logging import log_rank, warn_once
from nanotron.core.optimizer.base import BaseOptimizer
from nanotron.core.optimizer.inherit_from_other_optimizer import InheritFromOtherOptimizer
from nanotron.core.parallelism.parameters import NanotronParameter

logger = logging.get_logger(__name__)


class ZeroDistributedOptimizer(InheritFromOtherOptimizer):
    """Optimizer that handles partitioning of optimizer's states across DP ranks. See ZeRO Stage 1 in the paper https://arxiv.org/abs/1910.02054v3 for more details."""

    def __init__(
        self,
        named_params_or_groups: Iterable[Union[Tuple[str, NanotronParameter], Dict[str, Any]]],
        optimizer_builder: Callable[[Iterable[Dict[str, Any]]], BaseOptimizer],
        dp_pg: ProcessGroup,
    ):
        named_params_or_groups = list(named_params_or_groups)
        if len(named_params_or_groups) == 0 or isinstance(named_params_or_groups[0], dict):
            # case where named_params_or_groups is Iterable[Dict[str, Any]]
            for d in named_params_or_groups:
                assert (
                    "named_params" in d
                ), f"param_groups must contain a 'named_params' key, got a dict with keys {d.keys()}"

            # keep only named_params_or_groups that require grads
            named_params_or_groups = [
                {
                    "named_params": [
                        (name, param) for name, param in named_param_group["named_params"] if param.requires_grad
                    ],
                    **{k: v for k, v in named_param_group.items() if k != "named_params"},
                }
                for named_param_group in named_params_or_groups
            ]

            self.zero_named_param_groups = named_params_or_groups
        else:
            # case where named_params_or_groups is Iterable[Tuple[str, NanotronParameter]]
            # keep only named_params_or_groups that require grads
            named_params_or_groups = [(name, param) for name, param in named_params_or_groups if param.requires_grad]
            self.zero_named_param_groups = [{"named_params": named_params_or_groups}]

        self.dp_pg = dp_pg  # DP process group

        # partition model's params across DP ranks.
        # `self.param_name_to_dp_rank_offsets` sets mapping between each param inside self.named_params and its rank
        # NOTE: some param_groups may have no params in the current rank. we still keep them in self.optimizer.param_groups
        self.param_name_to_dp_rank_offsets = self._partition_parameters()

        current_dp_rank = dist.get_rank(self.dp_pg)
        param_groups_in_rank = [
            {
                "named_params": [
                    (
                        name,
                        get_sliced_tensor(
                            param=param,
                            start_offset=self.param_name_to_dp_rank_offsets[name][current_dp_rank][0],
                            end_offset=self.param_name_to_dp_rank_offsets[name][current_dp_rank][1],
                        ),
                    )
                    for name, param in param_group["named_params"]
                    if current_dp_rank in self.param_name_to_dp_rank_offsets[name]
                ],
                **{k: v for k, v in param_group.items() if k != "named_params"},
            }
            for param_group in self.zero_named_param_groups
        ]

        # initialize rank's optimizer which is responsible for updating the rank's parameters
        # NOTE: In case of ZeRO, `self.id_to_name` stores only names of parameters that are going to be updated by this DP rank's optimizer.
        # NOTE: In case of ZeRO, `self.optimizer` will only get the parameters that are going to be updated by this DP's optimizer. Which
        # means that `self.optimizer.param_groups` is only a subset of `self.param_groups`.
        optimizer = optimizer_builder(param_groups_in_rank)
        super().__init__(optimizer=optimizer, id_to_name=optimizer.id_to_name)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step (parameter update)."""
        # TODO: @nouamanetazi: handle syncing param groups attrs (e.g. if we update lr)

        loss = super().step(closure=closure)

        # calculate param_size (model) + param_size (grads) + 2*param_size/DP_if_zero1 (optim_states)
        expected_allocated = sum(
            param.numel() * param.element_size() * 2 + param.numel() * param.element_size() * 2 / self.dp_pg.size()
            for named_param_group in self.zero_named_param_groups
            for _, param in named_param_group["named_params"]
        )

        log_rank(
            f"[After optim states allocation] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB "
            f"(Expected 2*param_size + 2*param_size/DP_if_zero1={expected_allocated / 1024**2:.2f}MB). "
            f"Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
            logger=logger,
            level=logging.DEBUG,
            group=self.dp_pg,
            rank=0,
        )

        # All gather updated params
        self._all_gather_params()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Copied from `torch.optim.optimizer.zero_grad` with the only change of using `self.param_groups` instead of `self.optimizer.param_groups`
        because we want to zero out the gradients of all model params (not just the params in the current rank)"""
        super().zero_grad(set_to_none=set_to_none)

        # TODO @thomasw21: This is a call to torch internal API, we need to fix this
        foreach = False  # self.optimizer.defaults.get("foreach", False)

        # TODO @thomasw21: This is a call to torch internal API, we need to fix this
        # if not hasattr(self.optimizer, "_zero_grad_profile_name"):
        #     self.optimizer._hook_for_profile()

        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))

        # TODO @thomasw21: This is a call to torch internal API, we need to fix this
        # with torch.autograd.profiler.record_function(self.optimizer._zero_grad_profile_name):

        # zero out the gradients of all model params (not just the params in the current rank)
        for named_param_group in self.zero_named_param_groups:
            for _, p in named_param_group["named_params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        if not foreach or p.grad.is_sparse:
                            p.grad.zero_()
                        else:
                            per_device_and_dtype_grads[p.grad.device][  # pylint: disable=used-before-assignment
                                p.grad.dtype
                            ].append(p.grad)
        if foreach:
            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._foreach_zero_(grads)

    def _partition_parameters(self) -> Dict[str, Dict[int, Tuple[int, int]]]:
        named_params = [
            (name, param)
            for named_param_group in self.zero_named_param_groups
            for name, param in named_param_group["named_params"]
            if param.requires_grad
        ]

        # maps each model's param to the optimizer's dp rank that is responsible for updating it

        # We assume that parameters can be sharded across DP, ie we can "split" a parameter in different DP. This does break some optimizers, like Adafactor and such.
        # `param_name_to_dp_rank_offsets[name]` is a `Dict[int, Tuple[int, int]]` keys are dp_rank, and `Tuple[int, int]` are the offsets of the param belonging to this DP
        param_name_to_dp_rank_offsets = {}

        for name, param in named_params:
            # We assume parameter to be contiguous in order to have an easy way of sharding it.
            assert param.is_contiguous(), f"Parameter {name} is not contiguous"

            numel = param.numel()
            padded_numel_per_dp = (numel - 1) // self.dp_pg.size() + 1
            sizes = np.full(shape=(self.dp_pg.size()), fill_value=padded_numel_per_dp)
            remainder = padded_numel_per_dp * self.dp_pg.size() - numel
            # Last `remainder` indices has one less element
            if remainder > 0:
                # It's weird that `size[-0:]` returns the entire list instead of nothing
                sizes[-remainder:] -= 1
            end_offsets = np.cumsum(sizes)
            assert len(end_offsets) == self.dp_pg.size()
            assert end_offsets[-1] == numel, f"Somehow {end_offsets[-1]} != {numel}"
            # We want start indices,
            start_offsets = np.concatenate([[0], end_offsets[:-1]])

            param_name_to_dp_rank_offsets[name] = {
                dp_rank: (start_offsets[dp_rank], end_offsets[dp_rank])
                for dp_rank in range(self.dp_pg.size())
                if start_offsets[dp_rank] < end_offsets[dp_rank]  # Only if the slice is not empty.
            }

        log_rank("[ZeRO sharding] Size of optimizer params per rank:", logger=logger, level=logging.DEBUG, rank=0)
        all_memory = sum(
            param_name_to_dp_rank_offsets[name][dp_rank][1] - param_name_to_dp_rank_offsets[name][dp_rank][0]
            for name, param in named_params
            for dp_rank in range(self.dp_pg.size())
            if dp_rank in param_name_to_dp_rank_offsets[name]
        )
        for dp_rank in range(self.dp_pg.size()):
            acc_memory = sum(
                value[dp_rank][1] - value[dp_rank][0]
                for value in param_name_to_dp_rank_offsets.values()
                if dp_rank in value
            )
            log_rank(
                f"[ZeRO sharding] Rank {dp_rank} has {all_memory / 1024 ** 2:.2f} MB out of {acc_memory / 1024 ** 2:.2f} MB ({0 if all_memory == 0 else acc_memory / all_memory * 100:.2f}%) optimizer params",
                logger=logger,
                level=logging.DEBUG,
                rank=0,
            )

        return param_name_to_dp_rank_offsets

    def _all_gather_params(self):
        """All gather updated params"""
        all_named_tensors_to_gather = [
            (name, param.view(-1))
            for named_param_groups in self.zero_named_param_groups
            for name, param in named_param_groups["named_params"]
        ]

        if len(all_named_tensors_to_gather) == 0:
            # No need to broadcast if there's nothing
            return

        if self.dp_pg.size() == 1:
            # They should already be updated
            return

        current_dp_rank = dist.get_rank(self.dp_pg)
        dist.all_gather_coalesced(
            output_tensor_lists=[
                [
                    tensor[slice(*self.param_name_to_dp_rank_offsets[name][dp_rank])]
                    if dp_rank in self.param_name_to_dp_rank_offsets[name]
                    else torch.empty(0, dtype=tensor.dtype, device=tensor.device)
                    for dp_rank in range(self.dp_pg.size())
                ]
                for name, tensor in all_named_tensors_to_gather
            ],
            input_tensor_list=[
                tensor[slice(*self.param_name_to_dp_rank_offsets[name][current_dp_rank])]
                if current_dp_rank in self.param_name_to_dp_rank_offsets[name]
                else torch.empty(0, dtype=tensor.dtype, device=tensor.device)
                for name, tensor in all_named_tensors_to_gather
            ],
            group=self.dp_pg,
        )


# Helpers


class SlicedFlatTensor(torch.Tensor):
    """Subclass of `torch.Tensor` that unable to define `grad` getter on a slice of a flattened tensor."""

    # Based on torch/testing/_internal/logging_tensor.py
    # https://github.com/pytorch/pytorch/issues/102337#issuecomment-1579673041
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def get_sliced_flat_tensor(data, start_offset, end_offset):
        with torch.no_grad():
            return data.view(-1)[start_offset:end_offset]

    @staticmethod
    def __new__(cls, data, start_offset, end_offset):
        sliced_tensor = cls.get_sliced_flat_tensor(data=data, start_offset=start_offset, end_offset=end_offset)

        result = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            sliced_tensor.size(),
            strides=sliced_tensor.stride(),
            storage_offset=sliced_tensor.storage_offset(),
            # TODO: clone storage aliasing
            dtype=sliced_tensor.dtype,
            layout=sliced_tensor.layout,
            device=sliced_tensor.device,
            requires_grad=sliced_tensor.requires_grad,
        )
        return result

    def __init__(self, data, start_offset, end_offset):
        super().__init__()
        # TODO @thomasw21: Make is so that you can never update this value
        self.sliced_flat_tensor = self.get_sliced_flat_tensor(
            data=data, start_offset=start_offset, end_offset=end_offset
        )
        self.orig_data = data
        self.start_offset = start_offset
        self.end_offset = end_offset

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.sliced_flat_tensor if isinstance(e, cls) else e

        def never_wrap(e):
            # Never re-wrap
            return e

        return tree_map(never_wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

    def _get_grad(self):
        if self.orig_data.grad is None:
            return None
        with torch.no_grad():
            return self.orig_data.grad.view(-1)[self.start_offset : self.end_offset]

    def _set_grad(self, grad):
        if grad is not None:
            orig_grad = self._get_grad()
            if orig_grad is None:
                raise NotImplementedError(
                    "Trying to set gradient on a sliced tensor when the original tensor hasn't allocated the buffer for the gradient"
                )
            orig_grad.copy_(grad)
            return
        # TODO @thomasw21: This is unfortunately necessary since we might pass `SliceTensor` to the optimizer.
        warn_once(
            logger=logger,
            msg="You're setting a `SlicedTensor` gradient to None. We're going to assume you meant to set the original tensor gradient to None.",
            rank=0,
        )
        self.orig_data.grad = None

    def _del_grad(self):
        raise NotImplementedError

    # TODO @thomasw21: Figure out why this function doesn't get inherited. https://github.com/pytorch/pytorch/issues/102337#issuecomment-1634363356
    def data_ptr(self):
        return self.sliced_flat_tensor.data_ptr()

    grad = property(_get_grad, _set_grad, _del_grad)


def get_sliced_tensor(param: NanotronParameter, start_offset: int, end_offset: int):
    # This allows us to create a leaf tensor, despite sharing the underlying storage
    result = SlicedFlatTensor(data=param, start_offset=start_offset, end_offset=end_offset)
    return result
