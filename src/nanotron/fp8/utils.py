from typing import Dict, List, Optional, Tuple

import torch
import transformer_engine as te  # noqa
from torch import nn

from nanotron import logging
from nanotron.config.fp8_config import FP8Args, FP8LayerArgs
from nanotron.fp8.constants import FP8_GPU_NAMES, FP8LM_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.meta import FP8Meta
from nanotron.models.base import NanotronModel

logger = logging.get_logger(__name__)


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

    # NOTE: do .clone() somehow fixes nan grad,
    # check `exp801_fp8_nan_debug` for more details
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
    # TODO(xrsrke): do we need clone?
    fp8_linear._set_and_quantize_weights(linear.weight.data.clone())

    if is_bias:
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
        return {
            "mean": tensor.mean(),
            "std": tensor.std(),
            "var": tensor.var(),
            "l1_norm": tensor.norm(p=1),
            "l2_norm": tensor.norm(p=2),
            "rms": tensor.pow(2).mean().sqrt(),
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


def find_fp8_config_by_module_name(target_module_name: str, config: FP8Args) -> Optional[FP8LayerArgs]:
    # NOTE: either model or is_quant_all_except_first_and_last must be specified, not both
    # assert config.fp8.model is not None or config.fp8.is_quant_all_except_first_and_last is not None

    # TODO(xrsrke): remove config.is_quant_all_except_first_and_last
    from nanotron.fp8.constants import FP8LM_LINEAR_RECIPE

    if hasattr(config, "model") and config.model is not None:
        for layer_args in config.model:
            if layer_args.module_name == target_module_name.replace("pp_block.", "").replace("module.", ""):
                return layer_args
    # elif config.is_quant_all_except_first_and_last:
    else:

        def match_layer_pattern(name, layer_idxs):
            # patterns = [
            #     "model.decoder.{}.pp_block.attn.qkv_proj",
            #     "model.decoder.{}.pp_block.attn.o_proj",
            #     "model.decoder.{}.pp_block.mlp.down_proj",
            #     "model.decoder.{}.pp_block.mlp.gate_up_proj",
            # ]
            patterns = [
                "model.decoder.{}.attn.qkv_proj",
                "model.decoder.{}.attn.o_proj",
                "model.decoder.{}.mlp.down_proj",
                "model.decoder.{}.mlp.gate_up_proj",
            ]

            for idx in layer_idxs:
                for pattern in patterns:
                    if name == pattern.format(idx):
                        return True

            return False

        from nanotron import constants

        num_layers = constants.CONFIG.model.model_config.num_hidden_layers
        assert num_layers > 2, "num_hidden_layers must be greater than 2"
        # assert config.fp8_linear_config_temp is not None

        quant_layer_idxs = list(range(1, num_layers - 1))
        # NOTE: remove ".pp_block" from module name
        if match_layer_pattern(target_module_name.replace(".pp_block", ""), quant_layer_idxs) is True:
            from copy import deepcopy

            # config_temp = deepcopy(config.fp8_linear_config_temp)
            config_temp = deepcopy(FP8LM_LINEAR_RECIPE)
            config_temp.module_name = target_module_name
            return config_temp
    # else:
    #     from nanotron.fp8.constant_recipe import MODULE_NAMES_THAT_NOT_FP8

    #     if any(module_name in target_module_name for module_name in MODULE_NAMES_THAT_NOT_FP8):
    #         return None
    #     else:
    #         # NOTE: return default recipe
    #         # NOTE: based on the global setting smooth_quant to decide whether to do smooth quantization
    #         # or not
    #         recipe = FP8LM_LINEAR_RECIPE
    #         recipe.smooth_quant = config.smooth_quant
    #         log_rank(
    #             f"target_module_name={target_module_name}, smooth_quant={recipe.smooth_quant}",
    #             logger=logger,
    #             level=logging.INFO,
    #             rank=0,
    #         )

    #         return recipe
    # return None


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

    # if hasattr(module, "name") and "lm_head" in module.name:
    #     assert 1 == 1

    if constants.CONFIG is not None and constants.CONFIG.fp8.model is None:
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


def convert_model_to_fp8(model: NanotronModel, config: FP8Args) -> NanotronModel:
    from nanotron.fp8.utils import get_leaf_modules

    assert 1 == 1
    # NOTE: convert to FP8

    # from nanotron import constants
    from nanotron.fp8.utils import find_fp8_config_by_module_name
    from nanotron.parallel.tensor_parallel.nn import (
        FP8TensorParallelColumnLinear,
        FP8TensorParallelRowLinear,
        TensorParallelColumnLinear,
        TensorParallelRowLinear,
    )

    TP_LINEAR_CLS_TO_FP8_LINEAR_CLS = {
        TensorParallelColumnLinear: FP8TensorParallelColumnLinear,
        TensorParallelRowLinear: FP8TensorParallelRowLinear,
    }
    for name, module in get_leaf_modules(model):
        if any(p.numel() > 0 for p in module.parameters()) is False:
            continue

        recipe = find_fp8_config_by_module_name(name, config)

        # if isinstance(module, (TensorParallelColumnLinear, TensorParallelRowLinear)):
        if recipe is not None:
            print(f"Converting {name} to FP8")
            module.__class__ = TP_LINEAR_CLS_TO_FP8_LINEAR_CLS[module.__class__]
            # TODO(xrsrke): retrieve custom recipe
            module._set_and_quantize_weights(module.weight.data)

            # assert isinstance(module.weight, NanotronParameter)
            # assert module.weight.data.__class__ == FP8Tensor
            # assert module.weight.data.dtype in [
            #     torch.uint8,
            #     torch.int8,
            # ], f"got {module.weight.data.dtype}, name: {name}"
        else:
            # NOTE: convert it to the residual stream's dtype
            # for p in module.parameters():
            #     p.data = p.data.to(self.config.model.dtype)
            # for p in module.parameters():
            #     p.data = p.data.to(dtype=config.resid_dtype) if p.data
            # pass
            # assert module.weight.data.__class__ == torch.Tensor
            # module.to(dtype=config.resid_dtype)
            # pass
            # assert module.weight.data.__class__ == torch.Tensor
            # NOTE: this causes param.data == NanotronParameter
            assert config.resid_dtype == torch.float32, "not support datatype conversion, because of error 8"

    return model
