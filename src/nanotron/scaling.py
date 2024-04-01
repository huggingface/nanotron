from math import ceil
from typing import Set

import torch
from torch import nn
from torch.fx import symbolic_trace

from nanotron.models import init_on_device_and_dtype
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    DifferentiableAllGather,
    DifferentiableAllReduceSum,
    DifferentiableIdentity,
    DifferentiableReduceScatterSum,
)


class _ExtendedLeafTracer(torch.fx.Tracer):
    """Tracer with an extended set of leaf nn.Modules."""

    def __init__(self, leaf_modules: Set[torch.nn.Module]):
        """Initializes a new _ExtendedLeafTracer object.

        Args:
            leaf_modules: The set of extra nn.Modules instances which will not be traced
                through but instead considered to be leaves.
        """
        super().__init__()
        self.leaf_modules = leaf_modules
        # self.skip_modules = skip_modules

    def is_leaf_module(self, m: torch.nn.Module, model_qualified_name: str) -> bool:
        # return super().is_leaf_module(m, model_qualified_name)
        return super().is_leaf_module(m, model_qualified_name) or m in self.leaf_modules


def _trace(model: torch.nn.Module, skip_modules) -> torch.fx.GraphModule:
    """Traces the given model and automatically wraps untracable modules into leaves."""
    leaf_modules = set()
    tracer = _ExtendedLeafTracer(leaf_modules)
    for name, module in model.named_modules():
        if isinstance(module, PipelineBlock):
            assert 1 == 1

        if module in skip_modules or isinstance(module, tuple(skip_modules)):
            continue

        if module.__class__ in skip_modules:
            continue

        # TODO(ehotaj): The default is_leaf_module includes everything in torch.nn.
        # This means that some coarse modules like nn.TransformerEncoder are treated
        # as leaves, not traced, and are unable to be sharded. We may want to extend our
        # sharding code to trace through these modules as well.
        if tracer.is_leaf_module(module, ""):
            continue
        try:
            tracer.trace(module)
        except (TypeError, torch.fx.proxy.TraceError):
            leaf_modules.add(module)
            tracer = _ExtendedLeafTracer(leaf_modules)
    graph = tracer.trace(model)
    return torch.fx.GraphModule(model, graph)


def mu_transfer(model: nn.Module, dtype: torch.dtype, parallel_context: ParallelContext) -> nn.Module:
    pipeline_blocks = [module for _, module in model.named_modules() if isinstance(module, PipelineBlock)]

    with init_on_device_and_dtype(device=torch.device("meta"), dtype=dtype):
        contiguous_size = ceil(len(pipeline_blocks) / parallel_context.pp_pg.size())
        for i, block in enumerate(pipeline_blocks):
            rank = i // contiguous_size
            block.build_and_set_rank(rank)

    assert 1 == 1
    symbolic_trace(model)


# def mu_transfer(model: nn.Module) -> nn.Module:
#     assert 1 == 1
#     return model


def _tracing(model):
    skip_modules = [
        DifferentiableIdentity,
        DifferentiableAllReduceSum,
        DifferentiableAllGather,
        DifferentiableReduceScatterSum,
    ]

    traced_graph_module = _trace(model, skip_modules)
    return traced_graph_module
