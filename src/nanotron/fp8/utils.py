from typing import Dict, List, Optional, Tuple

import torch
import transformer_engine as te  # noqa
from torch import nn

from nanotron.config import Config
from nanotron.config.fp8_config import FP8LayerArgs
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


def calculate_kurtosis(X):
    # Calculate s
    s = torch.sqrt(torch.mean(X**2, dim=0))

    # Calculate m4 and m2
    m4 = torch.mean(s**4)
    m2 = torch.mean(s**2)

    # Calculate kurtosis
    kurtosis = m4 / (m2**2)

    if torch.isnan(kurtosis) and not torch.all(torch.eq(X, 0)).item():
        assert 1 == 1

    return kurtosis


def compute_stas(tensor):
    from nanotron.fp8.tensor import FP8Tensor, FP16Tensor

    def compute_snr(tensor):
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        snr = mean / std
        return snr

    if isinstance(tensor, FP8Tensor) or isinstance(tensor, FP16Tensor):
        return {
            "amax": tensor.fp8_meta.amax,
            "scale": tensor.fp8_meta.scale,
        }
    else:
        # return {
        #     "mean": tensor.mean().item(),
        #     "std": tensor.std().item(),
        #     "var": tensor.var().item(),
        #     "l1_norm": tensor.norm(p=1).item(),
        #     "l2_norm": tensor.norm(p=2).item(),
        #     "min": tensor.min().item(),
        #     "max": tensor.max().item(),
        #     "amax": tensor.abs().max().item(),
        #     "abs_mean": tensor.abs().mean().item(),
        #     "kurtosis": calculate_kurtosis(tensor),
        #     "snr": compute_snr(tensor),
        # }
        return {
            "mean": tensor.mean(),
            "std": tensor.std(),
            "var": tensor.var(),
            "l1_norm": tensor.norm(p=1),
            "l2_norm": tensor.norm(p=2),
            "min": tensor.min(),
            "max": tensor.max(),
            "amax": tensor.abs().max(),
            "abs_mean": tensor.abs().mean(),
            "kurtosis": calculate_kurtosis(tensor),
            "snr": compute_snr(tensor),
        }


def track_module_statistics(name: str, module: nn.Linear, logging: Dict[str, Dict[str, float]]):
    if name not in logging:
        logging[name] = {}

    def _save_output_stats(module: nn.Linear, input: torch.Tensor, output: torch.Tensor):
        if hasattr(module, "weight") and module.weight is not None:
            logging[name]["weight"] = compute_stas(module.weight.data)
            # logging[name]["weight"] = _collect_stats(module.weight)

        if hasattr(module, "bias") and module.bias is not None:
            logging[name]["bias"] = compute_stas(module.bias)

        inputs = input if isinstance(input, tuple) else (input,)
        outputs = output if isinstance(output, tuple) else (output,)

        if len(inputs) > 1:
            for i, inp in enumerate(inputs):
                if inp.dtype == torch.long:
                    # NOTE: this is input ids in transformers
                    continue
                logging[name][f"input:{i}"] = compute_stas(inp)
        else:
            logging[name]["input"] = compute_stas(inputs[0])

        if len(outputs) > 1:
            for i, out in enumerate(outputs):
                logging[name][f"output:{i}"] = compute_stas(out)
        else:
            logging[name]["output"] = compute_stas(outputs[0])

    def _save_grad_stats(module: nn.Linear, grad_input, grad_output: torch.Tensor):
        if isinstance(grad_output, tuple):
            for i, grad in enumerate(grad_output):
                if grad is None:
                    continue

                logging[name][f"grad_output:{i}"] = compute_stas(grad)
        else:
            logging[name]["grad_output"] = compute_stas(grad_output)

        if isinstance(grad_input, tuple):
            for i, grad in enumerate(grad_input):
                if grad is not None:
                    logging[name][f"grad_input:{i}"] = compute_stas(grad)
        else:
            if grad_input is not None:
                logging[name]["grad_input"] = compute_stas(grad_input)

    handles = []
    handles.append(module.register_forward_hook(_save_output_stats))
    handles.append(module.register_backward_hook(_save_grad_stats))
    return handles


def _log(model: nn.Module):
    LOGGING = {}
    leaf_modules = get_leaf_modules(model)
    all_handles = []
    for name, module in leaf_modules:
        all_handles.append(track_module_statistics(name, module, logging=LOGGING))

    return LOGGING, all_handles


def convert_logs_to_flat_logs(logs, prefix):
    flat_logs = {}
    for module_name, components in logs.items():
        for component_name, stats in components.items():
            for stat_name, value in stats.items():
                flat_logs[f"{prefix}:{module_name}:{component_name}:{stat_name}"] = value

    return flat_logs


def find_fp8_config_by_module_name(config: Config, target_module_name: str) -> Optional[FP8LayerArgs]:
    if hasattr(config, "fp8") and hasattr(config.fp8, "model"):
        if config.fp8.model is not None:
            for layer_args in config.fp8.model:
                if layer_args.module_name == target_module_name:
                    return layer_args
        else:
            from nanotron.fp8.constant_recipe import MODULE_NAMES_THAT_NOT_FP8
            from nanotron.fp8.constants import FP8LM_LINEAR_RECIPE

            if any(module_name in target_module_name for module_name in MODULE_NAMES_THAT_NOT_FP8):
                return None
            else:
                # NOTE: return default recipe
                return FP8LM_LINEAR_RECIPE
    return None


def get_modules_not_in_fp16():
    from nanotron import constants
    from nanotron.fp8.constant_recipe import MODULE_NAMES_THAT_NOT_FP8

    if constants.CONFIG is not None and hasattr(constants.CONFIG, "fp8"):
        if constants.CONFIG.fp8.model is None:
            # NOTE: convert all modules to fp8 axcept
            name_of_modules_not_in_fp16 = MODULE_NAMES_THAT_NOT_FP8
        else:
            name_of_modules_not_in_fp16 = [x.module_name for x in constants.CONFIG.fp8.model]
    else:
        name_of_modules_not_in_fp16 = []
    return name_of_modules_not_in_fp16


def is_convert_to_fp16(module) -> bool:
    from nanotron import constants
    from nanotron.fp8.constant_recipe import MODULE_NAMES_THAT_NOT_FP8, MODULES_THAT_IN_FLOAT16

    IS_CONVERT_TO_FLOAT16 = False
    name_of_modules_not_in_fp16 = get_modules_not_in_fp16()

    if hasattr(module, "name") and "lm_head" in module.name:
        assert 1 == 1

    if constants.CONFIG.fp8.model is None:
        if any(isinstance(module, m) for m in MODULES_THAT_IN_FLOAT16):
            IS_CONVERT_TO_FLOAT16 = True
        else:
            if hasattr(module, "name") and any(n in module.name for n in MODULE_NAMES_THAT_NOT_FP8):
                IS_CONVERT_TO_FLOAT16 = True
    else:
        if any(isinstance(module, m) for m in MODULES_THAT_IN_FLOAT16) or (
            hasattr(module, "name") and module.name not in name_of_modules_not_in_fp16
        ):
            IS_CONVERT_TO_FLOAT16 = True

    return IS_CONVERT_TO_FLOAT16
