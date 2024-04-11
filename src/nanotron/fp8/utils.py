from typing import Dict, List, Tuple

# import pydevd
import torch
import transformer_engine as te  # noqa
from torch import nn

from nanotron.fp8.constants import FP8_GPU_NAMES, FP8LM_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.parameter import FP8Parameter


def is_fp8_available() -> bool:
    """Check if FP8 is available on the current device."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
        return any(gpu_name in device_name for gpu_name in FP8_GPU_NAMES)
    else:
        return False


def get_tensor_fp8_metadata(tensor: torch.Tensor, dtype: DTypes) -> FP8Meta:
    from nanotron.fp8.constants import INITIAL_SCALING_FACTOR
    from nanotron.fp8.tensor import update_scaling_factor

    amax = tensor.abs().max().clone()
    assert amax.dtype == torch.float32

    scale = update_scaling_factor(amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), dtype)
    assert scale.dtype == torch.float32

    fp8_meta = FP8Meta(amax, scale, dtype)
    return fp8_meta


# TODO(xrsrke): shorter name
def is_overflow_underflow_nan(tensor: torch.Tensor) -> bool:
    assert isinstance(tensor, torch.Tensor)
    
    overflow = torch.isinf(tensor).any().item()
    underflow = torch.isneginf(tensor).any().item()
    nan = torch.isnan(tensor).any().item()

    return True if (overflow or underflow or nan) else False


def convert_linear_to_fp8(linear: nn.Linear, accum_qtype: DTypes = FP8LM_RECIPE.linear.accum_dtype) -> FP8Linear:
    in_features = linear.in_features
    out_features = linear.out_features
    is_bias = linear.bias is not None

    fp8_linear = FP8Linear(
        in_features, out_features, bias=is_bias, device=linear.weight.device, accum_qtype=accum_qtype
    )
    fp8_linear.weight = FP8Parameter(linear.weight.data.clone(), FP8LM_RECIPE.linear.weight.dtype)

    if is_bias:
        fp8_linear.bias.orig_data = linear.bias.data.clone()
        fp8_linear.bias.data = linear.bias.data.to(QTYPE_TO_DTYPE[accum_qtype])

    return fp8_linear


def get_leaf_modules(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return all the leaf modules (modules without any child modules) in a PyTorch module.
    """
    leaf_modules = []
    for n, m in module.named_modules():
        if not list(m.children()):
            leaf_modules.append((n, m))
    return leaf_modules


def convert_to_fp8_module(module: nn.Module, accum_qtype: DTypes = FP8LM_RECIPE.linear.accum_dtype) -> nn.Module:
    def set_module(model, name, value):
        parts = name.split(".")
        module = model
        for i, part in enumerate(parts):
            if part.isdigit():
                if i == len(parts) - 1:
                    module[int(part)] = value
                else:
                    module = module[int(part)]
            else:
                if i == len(parts) - 1:
                    setattr(module, part, value)
                else:
                    module = getattr(module, part)
        return model

    for name, child in get_leaf_modules(module):
        if isinstance(child, nn.Linear):
            fp8_linear = convert_linear_to_fp8(child, accum_qtype)
            fp8_linear.name = name
            set_module(module, name, fp8_linear)

    return module


def track_module_statistics(name: str, module: nn.Linear, logging: Dict[str, Dict[str, float]]):
    if name not in logging:
        logging[name] = {}

    def _collect_stats(tensor):
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "var": tensor.var().item(),
            "norm": tensor.norm().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }

    def _save_output_stats(module: nn.Linear, input: torch.Tensor, output: torch.Tensor):        
        if hasattr(module, "weight") and module.weight is not None:
            logging[name]["weight"] = _collect_stats(module.weight)
            # logging[name]["weight"] = _collect_stats(module.weight)

        if hasattr(module, "bias") and module.bias is not None:
            logging[name]["bias"] = _collect_stats(module.bias)

        inputs = input if isinstance(input, tuple) else (input,)
        outputs = output if isinstance(output, tuple) else (output,)
        
        if len(inputs) > 1:
            for i, inp in enumerate(inputs):
                if inp.dtype == torch.long:
                    # NOTE: this is input ids in transformers
                    continue
                logging[name][f"input:{i}"] = _collect_stats(inp)
        else:
            logging[name]["input"] = _collect_stats(inputs[0])
        
        if len(outputs) > 1:
            for i, out in enumerate(outputs):
                logging[name][f"output:{i}"] = _collect_stats(out)
        else:
            logging[name]["output"] = _collect_stats(outputs[0])
    
    def _save_grad_stats(module: nn.Linear, grad_input, grad_output: torch.Tensor):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # logging[name][f"weight_grad"] = _collect_stats(module.weight.grad.orig_data)
        # logging[name][f"bias_grad"] = _collect_stats(module.bias.grad)
        
        if isinstance(grad_output, tuple):
            for i, grad in enumerate(grad_output):
                logging[name][f"grad_output:{i}"] = _collect_stats(grad)
        else:
            logging[name]["grad_output"] = _collect_stats(grad_output)
            
            
        if isinstance(grad_input, tuple):
            for i, grad in enumerate(grad_input):
                if grad is not None:
                    logging[name][f"grad_input:{i}"] = _collect_stats(grad)
        else:
            if grad_input is not None:
                logging[name]["grad_input"] = _collect_stats(grad_input)

    module.register_forward_hook(_save_output_stats)
    # module.register_full_backward_pre_hook(_save_grad_stats)
    module.register_backward_hook(_save_grad_stats)
    # module.register_module_full_backward_hook(_save_grad_stats)


def _log(model: nn.Module):
    LOGGING = {}
    leaf_modules = get_leaf_modules(model)
    for name, module in leaf_modules:
        track_module_statistics(name, module, logging=LOGGING)

    
    return LOGGING


def convert_logs_to_flat_logs(logs, prefix):
    flat_logs = {}
    for module_name, components in logs.items():
        for component_name, stats in components.items():
            for stat_name, value in stats.items():
                flat_logs[f"{prefix}:{module_name}:{component_name}:{stat_name}"] = value
    
    return flat_logs
