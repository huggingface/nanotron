import math
from enum import Enum, auto
from typing import List, Tuple

import torch
from torch import nn

from nanotron.config import Config
from nanotron.parallel import ParallelContext

# class _ExtendedLeafTracer(torch.fx.Tracer):
#     """Tracer with an extended set of leaf nn.Modules."""

#     def __init__(self, leaf_modules: Set[torch.nn.Module]):
#         """Initializes a new _ExtendedLeafTracer object.

#         Args:
#             leaf_modules: The set of extra nn.Modules instances which will not be traced
#                 through but instead considered to be leaves.
#         """
#         super().__init__()
#         self.leaf_modules = leaf_modules
#         # self.skip_modules = skip_modules

#     def is_leaf_module(self, m: torch.nn.Module, model_qualified_name: str) -> bool:
#         # return super().is_leaf_module(m, model_qualified_name)
#         return super().is_leaf_module(m, model_qualified_name) or m in self.leaf_modules


# def _trace(model: torch.nn.Module, skip_modules) -> torch.fx.GraphModule:
#     """Traces the given model and automatically wraps untracable modules into leaves."""
#     leaf_modules = set()
#     tracer = _ExtendedLeafTracer(leaf_modules)
#     for name, module in model.named_modules():
#         if isinstance(module, PipelineBlock):
#             assert 1 == 1

#         if module in skip_modules or isinstance(module, tuple(skip_modules)):
#             continue

#         if module.__class__ in skip_modules:
#             continue

#         # TODO(ehotaj): The default is_leaf_module includes everything in torch.nn.
#         # This means that some coarse modules like nn.TransformerEncoder are treated
#         # as leaves, not traced, and are unable to be sharded. We may want to extend our
#         # sharding code to trace through these modules as well.
#         if tracer.is_leaf_module(module, ""):
#             continue
#         try:
#             tracer.trace(module)
#         except (TypeError, torch.fx.proxy.TraceError):
#             leaf_modules.add(module)
#             tracer = _ExtendedLeafTracer(leaf_modules)
#     graph = tracer.trace(model)
#     return torch.fx.GraphModule(model, graph)


# def mu_transfer(model: nn.Module, dtype: torch.dtype, parallel_context: ParallelContext) -> nn.Module:
#     pipeline_blocks = [module for _, module in model.named_modules() if isinstance(module, PipelineBlock)]

#     with init_on_device_and_dtype(device=torch.device("meta"), dtype=dtype):
#         contiguous_size = ceil(len(pipeline_blocks) / parallel_context.pp_pg.size())
#         for i, block in enumerate(pipeline_blocks):
#             rank = i // contiguous_size
#             block.build_and_set_rank(rank)

#     assert 1 == 1
#     symbolic_trace(model)


# def _tracing(model):
#     skip_modules = [
#         DifferentiableIdentity,
#         DifferentiableAllReduceSum,
#         DifferentiableAllGather,
#         DifferentiableReduceScatterSum,
#     ]

#     traced_graph_module = _trace(model, skip_modules)
#     return traced_graph_module


class WeightType(Enum):
    INPUT_WEIGHTS = auto()
    HIDDEN_WEIGHTS = auto()
    OUTPUT_WEIGHTS = auto()


def _get_leaf_modules(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return all the leaf modules (modules without any child modules) in a PyTorch module.
    """
    leaf_modules = []
    for n, m in module.named_modules():
        if not list(m.children()):
            leaf_modules.append((n, m))
    return leaf_modules


def init_weight_for_input_weights(weight, hidden_size):
    var = 1 / hidden_size
    torch.nn.init.normal_(weight, mean=0.0, std=math.sqrt(var))


def init_weight_for_hidden_weights(weight, hidden_size):
    var = 1 / hidden_size
    torch.nn.init.normal_(weight, mean=0.0, std=math.sqrt(var))


def init_weight_for_output_weights(weight, hidden_size):
    torch.nn.init.normal_(weight, mean=0.0, std=1)


def init_weight_using_mu_transfer(weight, name, hidden_size, linear_type):
    if linear_type == WeightType.INPUT_WEIGHTS:
        if name == "weight" or name == "bias":
            init_weight_for_input_weights(weight, hidden_size)
    elif linear_type == WeightType.HIDDEN_WEIGHTS:
        init_weight_for_hidden_weights(module.weight, hidden_size)
    elif linear_type == WeightType.OUTPUT_WEIGHTS:
        init_weight_for_output_weights(module.weight, hidden_size)
    else:
        raise ValueError(f"Unknown linear type: {linear_type}")


def parametrize_using_mu_transfer(model: nn.Module, config: Config, parallel_context: ParallelContext) -> nn.Module:
    list(model.named_parameters())
    # named_modules = list(model.named_modules())
    named_modules = _get_leaf_modules(model)

    [x[0] for x in named_modules if not hasattr(x[1], "linear_type")]
    scale_modules = [x for x in named_modules if hasattr(x[1], "linear_type")]

    # for name, module in named_modules:
    #     if isinstance(module, LINEAR_MODULES):
    #         # NOTE: qkv don't have bias
    #         if module.bias is not None:
    #             assert 1 == 1

    fan_in = config.model.model_config.hidden_size

    def input_weight_hook(module, input, output):
        return output * 1

    def hidden_weight_hook(module, input, output):
        return output * 1

    def output_weight_hook(module, input, output):
        return output * (1 / fan_in)

    for name, module in scale_modules:
        assert isinstance(module, nn.Module)
        if module.linear_type == WeightType.INPUT_WEIGHTS:
            hook_func = input_weight_hook
        elif module.linear_type == WeightType.HIDDEN_WEIGHTS:
            hook_func = hidden_weight_hook
        elif module.linear_type == WeightType.OUTPUT_WEIGHTS:
            hook_func = output_weight_hook
        else:
            raise ValueError(f"Unknown linear type: {module.linear_type}")

        module.register_forward_hook(hook_func)
