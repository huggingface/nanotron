from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from nanotron import distributed as dist
from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import log_rank
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock

if TYPE_CHECKING:
    from nanotron.config import NanotronConfigs
    from nanotron.parallel.parameters import NanotronParameter

logger = logging.get_logger(__name__)


class NanotronModel(nn.Module, metaclass=ABCMeta):
    """Abstract class for Nanotron models
    We make the following assumptions:
    - When building PP blocks, we assume that the modules order are in the same order as the forward pass."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parallel_context: "ParallelContext"
        self.config: "NanotronConfigs"
        self.module_id_to_prefix: dict[int, str]

        # Attributes defined when building the model
        self.input_pp_rank: int
        self.output_pp_rank: int

        # Useful mapping to get param names
        self.module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in self.named_modules()}
        self.module_id_to_prefix[id(self)] = ""

    def get_named_params_with_correct_tied(self) -> Iterator[Tuple[str, "NanotronParameter"]]:
        """Return named parameters with correct tied params names.
        For example in the case of tied kv heads in MQA, we need to make sure tied params names are correct."""

        def params_gen():
            for name, param in self.named_parameters():
                if param.is_tied:
                    yield (
                        param.get_tied_info().get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=self.module_id_to_prefix
                        ),
                        param,
                    )
                else:
                    yield name, param

        yield from params_gen()

    @abstractmethod
    def init_model_randomly(self, init_method, scaled_init_method):
        ...

    def tie_custom_params(self) -> None:
        """Tie custom parameters. For example for MQA marks kv heads as tied."""
        pass

    @staticmethod
    def get_embeddings_lm_head_tied_names() -> list[str]:
        """Returns the names of the embeddings and lm_head weights that are tied together. Returns empty list if not tied.

        Example for GPT2 model: ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        """
        return []

    def before_tbi_sanity_checks(self) -> None:
        pass

    def after_tbi_sanity_checks(self) -> None:
        pass

    def before_optim_step_sanity_checks(self) -> None:
        pass

    def after_optim_step_sanity_checks(self) -> None:
        pass

    def log_modules(self, level: int = logging.DEBUG, group: Optional[ProcessGroup] = None, rank: int = 0):
        assert hasattr(self, "parallel_context"), "`NanotronModel` needs to have a `parallel_context` attribute"

        for name, module in self.named_modules():
            if not isinstance(module, PipelineBlock):
                continue
            log_rank(
                f"module_name: {name} | PP: {module.rank}/{self.parallel_context.pp_pg.size()}",
                logger=logger,
                level=level,
                group=group,
                rank=rank,
            )


class DTypeInvariantTensor(torch.Tensor):
    """DTypeInvariantTensor is a subclass of torch.Tensor that disallows modification of its dtype. Note that the data
    and other attributes of the tensor can still be modified."""

    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(cls, *args, **kwargs)
        return tensor

    def detach(self, *args, **kwargs):
        raise RuntimeError("Cannot detach an DTypeInvariantTensor")

    def to(self, *args, **kwargs):
        if "dtype" in kwargs or any(isinstance(arg, torch.dtype) for arg in args):
            raise RuntimeError("Cannot change the type of an DTypeInvariantTensor")
        else:
            return super().to(*args, **kwargs)

    def type(self, *args, **kwargs):
        raise RuntimeError("Cannot change the type of an DTypeInvariantTensor")

    def float(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to float")

    def double(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to double")

    def half(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to half")

    def long(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to long")

    def int(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to int")

    def short(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to short")

    def char(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to char")

    def byte(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to byte")

    def bool(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to bool")

    def bfloat16(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to bfloat16")


def build_model(
    model_builder: Callable[[], NanotronModel],
    parallel_context: ParallelContext,
    dtype: torch.dtype,
    target_pp_ranks: Optional[List[int]] = None,
    device: Optional[torch.device] = torch.device("cuda"),
) -> NanotronModel:
    """Build the model and set the pp ranks for each pipeline block."""
    # TODO: classes dont take same args
    log_rank("Building model..", logger=logger, level=logging.INFO, rank=0, group=parallel_context.world_pg)
    model: NanotronModel = model_builder()

    # If no target pp ranks are specified, we assume that we want to use all pp ranks
    if target_pp_ranks is None:
        pp_size = parallel_context.pp_pg.size()
        target_pp_ranks = list(range(pp_size))
    else:
        pp_size = len(target_pp_ranks)

    # Set rank for each pipeline block
    log_rank("Setting PP block ranks..", logger=logger, level=logging.INFO, rank=0, group=parallel_context.world_pg)
    pipeline_blocks = [module for name, module in model.named_modules() if isinstance(module, PipelineBlock)]
    # "cuda" is already defaulted for each process to it's own cuda device
    with init_on_device_and_dtype(device=device, dtype=dtype):
        # TODO: https://github.com/huggingface/nanotron/issues/65

        # Balance compute across PP blocks
        block_compute_costs = model.get_block_compute_costs()
        block_cumulative_costs = np.cumsum(
            [
                block_compute_costs[module.module_builder] if module.module_builder in block_compute_costs else 0
                for module in pipeline_blocks
            ]
        )

        thresholds = [block_cumulative_costs[-1] * ((rank + 1) / pp_size) for rank in range(pp_size)]
        assert thresholds[-1] >= block_cumulative_costs[-1]
        target_pp_rank_idx = 0
        for block, cumulative_cost in zip(pipeline_blocks, block_cumulative_costs):
            assert target_pp_rank_idx < pp_size
            block.build_and_set_rank(target_pp_ranks[target_pp_rank_idx])

            if cumulative_cost > thresholds[target_pp_rank_idx]:
                target_pp_rank_idx += 1

        model.input_pp_rank = target_pp_ranks[0]
        model.output_pp_rank = target_pp_ranks[target_pp_rank_idx]
    return model


# TODO @thomasw21: Should this option override user defined options? Maybe not ... right now it does.
@contextmanager
def init_on_device_and_dtype(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
):
    """
    A context manager under which models are initialized with all parameters on the specified device.
    Args:
        device (`torch.device` defaults to `cpu`):
            Device to initialize all parameters on.
        dtype (`torch.dtype` defaults to `torch.float`):
            Dtype to initialize all parameters on.
        include_buffers (`bool`, defaults to `False`):
            Whether or not to also default all buffers constructors given previous arguments.
    Example:
    ```python
    import torch.nn as nn
    from accelerate import init_on_device
    with init_on_device_and_dtype(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            if isinstance(param, DTypeInvariantTensor):
                # if param is DTypeInvariantTensor we should avoid updating it
                param.data = param.data.to(device)
            else:
                param.data = param.data.to(device, dtype)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            if isinstance(buffer, DTypeInvariantTensor):
                # if buffer is DTypeInvariantTensor we should avoid updating it
                buffer.data = buffer.data.to(device)
            else:
                module._buffers[name] = module._buffers[name].to(device, dtype)

    # Patch tensor creation
    tensor_constructors_to_patch = {
        torch_function_name: getattr(torch, torch_function_name)
        for torch_function_name in ["empty", "zeros", "ones", "full"]
    }

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            kwargs["dtype"] = dtype
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def check_model_has_grad(model: NanotronModel, parallel_context: "ParallelContext"):
    """Check that there's at least a parameter in current PP rank that has a gradient."""
    for param in model.parameters():
        if param.requires_grad:
            return True
    raise ValueError(
        f"Can't use DDP because model in PP={dist.get_rank(parallel_context.pp_pg)} has no gradient. Consider increasing the number of layers of your model, or put a smaller PP size.\n"
        f"Model: {model}"
    )
