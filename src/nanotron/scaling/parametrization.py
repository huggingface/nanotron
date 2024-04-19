import math
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict

import torch
from nanotron import logging
from nanotron.config import ModelArgs
from nanotron.logging import log_rank
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from torch import nn
from torch.nn import init

logger = logging.get_logger(__name__)

DP_PG = None

# NAME_TO_FAN_MAPPING = {"gate_up_proj": [1024, 4096], "token_embedding": [49152, 1024]}

LAYERNORM_NAMES = ["input_layernorm", "post_attention_layernorm", "final_layer_norm"]


def is_layernorm(name):
    for ln_name in LAYERNORM_NAMES:
        if ln_name in name:
            return True

    return False


NAME_TO_FAN_MAPPING = {
    # NOTE: the original
    # "token_embedding.weight": [49152, 1024],
    # "token_embedding.bias": [49152, 1],
    "token_embedding.weight": [1, 1024],
    "token_embedding.bias": [1, 1],
    # NOTE:
    "qkv_proj.weight": [1024, 2048],
    "qkv_proj.bias": [1024, 1],
    "o_proj.weight": [1024, 1024],
    "o_proj.bias": [1024, 1],
    "gate_up_proj.weight": [1024, 8192],
    "gate_up_proj.bias": [1024, 1],
    "down_proj.weight": [4096, 1024],
    "down_proj.bias": [4096, 1],
    "lm_head.pp_block.weight": [1024, 49152],
    # "token_embedding.bias": [1024, 1],
}


def spectral_sigma(fan_in, fan_out, init_std):
    """Spectral parameterization from the [paper](https://arxiv.org/abs/2310.17813)."""
    return (init_std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))


def spectral_lr(fan_in, fan_out):
    """Spectral parameterization from the [paper](https://arxiv.org/abs/2310.17813)."""
    return fan_out / fan_in


NAME_TO_STD = {
    name: spectral_sigma(fan_in, fan_out, init_std=1.0) for name, (fan_in, fan_out) in NAME_TO_FAN_MAPPING.items()
}
NAME_TO_LR = {name: spectral_lr(fan_in, fan_out) for name, (fan_in, fan_out) in NAME_TO_FAN_MAPPING.items()}


def get_fan_in_fan_out_from_param_name(name):
    for key, value in NAME_TO_FAN_MAPPING.items():
        if key in name:
            return value

    return None


def get_std_from_param_name(name, parallel_context):
    # global DP_PG
    # DP_PG = parallel_context.dp_pg

    # if "token_embedding" in name:
    #     log_rank(
    #         f"param_name={name}, std={1.0}",  # noqa
    #         logger=logger,
    #         level=logging.WARNING,
    #         group=DP_PG,
    #         rank=0
    #     )
    #     return 1.0

    # if is_layernorm(name):
    #     return 1.0

    fan_in, fan_out = get_fan_in_fan_out_from_param_name(name)
    std = spectral_sigma(fan_in=fan_in, fan_out=fan_out, init_std=1.0)

    log_rank(
        f"param_name={name}, fan_in={fan_in}, fan_out={fan_out}, std={std}",  # noqa
        logger=logger,
        level=logging.WARNING,
        group=parallel_context.dp_pg,
        rank=0,
    )

    return std


def get_lr_from_param_name(initial_lr, name):
    # global DP_PG

    # if "token_embedding" in name:
    #     log_rank(
    #         f"param_name={name}, final_lr={initial_lr}",  # noqa
    #         logger=logger,
    #         level=logging.WARNING,
    #         group=DP_PG,
    #         rank=0
    #     )
    #     return initial_lr

    # if is_layernorm(name):
    #     return 1.0

    # for key, value in NAME_TO_LR.items():
    #     if key in name:
    #         return value

    # return None
    fan_in, fan_out = get_fan_in_fan_out_from_param_name(name)
    lr = spectral_lr(fan_in=fan_in, fan_out=fan_out)
    lr_after_64 = lr / 64
    scaled_lr = initial_lr * lr_after_64

    log_rank(
        f"param_name={name}, fan_in={fan_in}, fan_out={fan_out}, lr={lr}, lr_after_64={lr_after_64}, scaled_lr={scaled_lr}",  # noqa
        logger=logger,
        level=logging.WARNING,
        group=DP_PG,
        rank=0,
    )

    return scaled_lr


class ParametrizationMethod(Enum):
    STANDARD = auto()
    SPECTRAL_MUP = auto()


class Parametrizator:
    def __init__(self, config: ModelArgs):
        self.config = config

    def parametrize(self, param_name: str, module: nn.Module):
        if not isinstance(module, tuple(self.MODULE_TO_PARAMETRIZE.keys())):
            raise Exception(f"Parameter {param_name} was not initialized")

        return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, module)


class StandardParametrizator(Parametrizator):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_column_linear,
            TensorParallelRowLinear: self._parametrize_row_linear,
            TritonRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }

        self.std = config.init_method.std
        self.num_layers = config.model_config.num_hidden_layers

    def _parametrize_column_linear(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_row_linear(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            std = self.std / math.sqrt(2 * self.num_layers)
            init.normal_(module.weight, mean=0.0, std=std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            # TODO @thomasw21: Sometimes we actually want 0
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        assert param_name in ["weight"]

        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


# class SpectralMupParametrizator(Parametrizator):
#     """
#     A Spectral Condition for Feature Learning by Greg Yang, et al.
#     https://arxiv.org/abs/2310.17813
#     """

#     def __init__(self, config: ModelArgs):
#         super().__init__(config)
#         self.MODULE_TO_PARAMETRIZE = {
#             TensorParallelColumnLinear: self._parametrize_mup_weight,
#             TensorParallelRowLinear: self._parametrize_mup_weight,
#             TritonRMSNorm: self._parametrize_layer_norm,
#             # TensorParallelEmbedding: self._parametrize_embedding,
#             TensorParallelEmbedding: self._parametrize_mup_weight,
#         }
#         self.std = 1.0
#         # self.std = 0.03125

#     @staticmethod
#     def _compute_spectral_std(std: float, fan_in: int, fan_out: int):
#         """
#         Parametrization 1 (Spectral parametrization)
#         Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

#         σₗ = Θ(1/√nₗ₋₁ min{1, √(nₗ/nₗ₋₁)})
#         """
#         return (std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))

#     def _parametrize_mup_weight(self, param_name: str, module: nn.Module):
#         assert param_name in ["weight", "bias"]

#         data = module.weight if param_name == "weight" else module.bias

#         if param_name == "weight":
#             fan_in = module.weight.shape[-1]
#             fan_out = torch.prod(module.weight.shape[:-1]).item()

#         # fan_in, fan_out = init._calculate_fan_in_and_fan_out(data)
#         # world_size = module.world_size
#         world_size = 2

#         if isinstance(module, TensorParallelColumnLinear):
#             fan_out = fan_out * world_size
#         elif isinstance(module, (TensorParallelRowLinear, TensorParallelEmbedding)):
#             fan_in = fan_in * world_size
#         # elif isinstance(module, (TensorParallelEmbedding)):
#         #     fan_in, fan_out = 49152, 1024
#         else:
#             raise ValueError(f"Unknown module {module}")

#         std = SpectralMupParametrizator._compute_spectral_std(std=self.std, fan_in=fan_in, fan_out=fan_out)

#         log_rank(
#             f"Parameter {param_name} has fan_in={fan_in}, fan_out={fan_out}, std={std}",
#             logger=logger,
#             level=logging.INFO,
#         )

#         init.normal_(data, mean=0.0, std=std)

#     def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
#         assert param_name in ["weight", "bias"]

#         # NOTE: you're free to change the initialization of layer norm
#         # as it's not a part of µTransfer
#         if "weight" == param_name:
#             module.weight.fill_(1)
#         elif "bias" == param_name:
#             module.bias.zero_()

#     def _parametrize_embedding(self, param_name: str, module: nn.Module):
#         assert param_name in ["weight"]

#         # NOTE: you're free to change the initialization of input embedding/lm head
#         if "weight" == param_name:
#             init.normal_(module.weight, mean=0.0, std=self.std)


class SpectralMupParametrizator:
    """
    A Spectral Condition for Feature Learning by Greg Yang, et al.
    https://arxiv.org/abs/2310.17813
    """

    def __init__(self, config: ModelArgs, parallel_context):
        self.config = config
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_mup_weight,
            TensorParallelRowLinear: self._parametrize_mup_weight,
            TritonRMSNorm: self._parametrize_layer_norm,
            # TensorParallelEmbedding: self._parametrize_embedding,
            TensorParallelEmbedding: self._parametrize_mup_weight,
        }
        self.std = 1.0
        self.parallel_context = parallel_context
        # self.std = 0.03125

    @staticmethod
    def _compute_spectral_std(std: float, fan_in: int, fan_out: int):
        """
        Parametrization 1 (Spectral parametrization)
        Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

        σₗ = Θ(1/√nₗ₋₁ min{1, √(nₗ/nₗ₋₁)})
        """
        return (std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))

    def _parametrize_mup_weight(self, param_name: str, module: nn.Module):
        # assert param_name in ["weight", "bias"]

        # data = module.weight if param_name == "weight" else module.bias

        # if param_name == "weight":
        #     fan_in = module.weight.shape[-1]
        #     fan_out = torch.prod(module.weight.shape[:-1]).item()

        # # fan_in, fan_out = init._calculate_fan_in_and_fan_out(data)
        # # world_size = module.world_size
        # world_size = 2

        # if isinstance(module, TensorParallelColumnLinear):
        #     fan_out = fan_out * world_size
        # elif isinstance(module, (TensorParallelRowLinear, TensorParallelEmbedding)):
        #     fan_in = fan_in * world_size
        # # elif isinstance(module, (TensorParallelEmbedding)):
        # #     fan_in, fan_out = 49152, 1024
        # else:
        #     raise ValueError(f"Unknown module {module}")

        # std = SpectralMupParametrizator._compute_spectral_std(std=self.std, fan_in=fan_in, fan_out=fan_out)

        # data = module.weight if param_name == "weight" else module.bias

        assert "weight" in param_name or "bias" in param_name
        data = module.weight if "weight" in param_name else module.bias

        std = get_std_from_param_name(param_name, self.parallel_context)

        # log_rank(
        #     f"Parameter {param_name} has fan_in={fan_in}, fan_out={fan_out}, std={std}",
        #     logger=logger,
        #     level=logging.INFO,
        # )

        if "weight" in param_name:
            module.weight.data = torch.randn_like(data.data, dtype=data.dtype, device=data.device) * std
        elif "bias" in param_name:
            module.bias.data = torch.randn_like(data.data, dtype=data.dtype, device=data.device) * std

        # data.data = torch.nn.Parameter(torch.randn_like(data.data) * std)

        # init.normal_(data, mean=0.0, std=std)

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        # assert param_name in ["weight", "bias"]
        assert "weight" in param_name or "bias" in param_name

        # NOTE: you're free to change the initialization of layer norm
        # as it's not a part of µTransfer
        # if "weight" == param_name:
        #     module.weight.fill_(1)
        # elif "bias" == param_name:
        #     module.bias.zero_()

        if "weight" in param_name:
            module.weight.fill_(1)
        elif "bias" in param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        # assert param_name in ["weight"]
        assert "weight" in param_name

        # NOTE: you're free to change the initialization of input embedding/lm head
        if "weight" in param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)

    def parametrize(self, param_name: str, module: nn.Module):
        if not isinstance(module, tuple(self.MODULE_TO_PARAMETRIZE.keys())):
            raise Exception(f"Parameter {param_name} was not initialized")

        return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, module)


class LearningRateForParametrizator:
    def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
        self.lr = lr
        self.names_to_modules = names_to_modules

    @abstractmethod
    def get_lr(self, param_name: str, module: nn.Module) -> float:
        raise NotImplementedError


class LearningRateForSP(LearningRateForParametrizator):
    """All parameters get the same learning rate."""

    def get_lr(self, param_name: str, param: nn.Module) -> float:
        return self.lr


# class LearningRateForSpectralMup(LearningRateForParametrizator):
#     """
#     A Spectral Condition for Feature Learning by Greg Yang, et al.

#     NOTE: each parameter gets a custom learning rate based on its fan-in and fan-out.
#     """

#     def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
#         super().__init__(lr, names_to_modules)
#         self.MODULE_TO_PARAMETRIZE = {
#             TensorParallelColumnLinear: self._get_mup_lr,
#             TensorParallelRowLinear: self._get_mup_lr,
#             TritonRMSNorm: self._get_global_lr,
#             # TensorParallelEmbedding: self._get_global_lr,
#             TensorParallelEmbedding: self._get_mup_lr,
#         }

#     def _get_mup_lr(self, param_name: str, param: nn.Parameter, module: nn.Module):
#         """
#         Parametrization 1 (Spectral parametrization)
#         Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

#         ηₗ = Θ(nₗ/nₗ₋₁)
#         """
#         fan_in, fan_out = init._calculate_fan_in_and_fan_out(param)
#         world_size = module.world_size
#         world_size = 2

#         if isinstance(module, (TensorParallelColumnLinear)):
#             fan_out = fan_out * world_size
#         elif isinstance(module, (TensorParallelRowLinear)):
#             fan_in = fan_in * world_size
#         # elif isinstance(module, (TensorParallelEmbedding)):
#         #     fan_in, fan_out = 49152, 1024
#         else:
#             raise ValueError(f"Unknown module {module}")

#         scaled_lr = (self.lr * (fan_out / fan_in)) / 64
#         log_rank(
#             f"Parameter {param_name} has fan_in={fan_in}, fan_out={fan_out}, scaled_lr={scaled_lr}",
#             logger=logger,
#             level=logging.INFO,
#         )

#         return scaled_lr

#     def _get_global_lr(self, param_name: str, param: nn.Parameter, module: nn.Module) -> float:
#         return self.lr

#     def get_lr(self, param_name: str, param: nn.Parameter) -> float:
#         """Return the learning rate for the given parameter."""
#         # NOTE: param_name should be like 'model.token_position_embeddings.pp_block.token_embedding.weight'
#         # since names_to_modules map module_name to module
#         # so we remove the .weight and .bias from param_name to get the module_name
#         module_name = param_name.rsplit(".", 1)[0]
#         module = self.names_to_modules[module_name]
#         return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, param, module)


class LearningRateForSpectralMup:

    """
    A Spectral Condition for Feature Learning by Greg Yang, et al.

    NOTE: each parameter gets a custom learning rate based on its fan-in and fan-out.
    """

    def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
        self.lr = lr
        self.names_to_modules = names_to_modules

        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._get_mup_lr,
            TensorParallelRowLinear: self._get_mup_lr,
            TritonRMSNorm: self._get_global_lr,
            # TensorParallelEmbedding: self._get_global_lr,
            TensorParallelEmbedding: self._get_mup_lr,
        }

    def _get_mup_lr(self, param_name: str, param: nn.Parameter, module: nn.Module):
        """
        Parametrization 1 (Spectral parametrization)
        Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

        ηₗ = Θ(nₗ/nₗ₋₁)
        """
        # fan_in, fan_out = init._calculate_fan_in_and_fan_out(param)
        # world_size = module.world_size
        # world_size = 2

        # if isinstance(module, (TensorParallelColumnLinear)):
        #     fan_out = fan_out * world_size
        # elif isinstance(module, (TensorParallelRowLinear)):
        #     fan_in = fan_in * world_size
        # # elif isinstance(module, (TensorParallelEmbedding)):
        # #     fan_in, fan_out = 49152, 1024
        # else:
        #     raise ValueError(f"Unknown module {module}")

        # scaled_lr = (self.lr * (fan_out / fan_in)) / 64

        scaled_lr = get_lr_from_param_name(self.lr, param_name)
        # scaled_lr = self.lr * scaling_factor

        # log_rank(
        #     f"Parameter {param_name} has fan_in={fan_in}, fan_out={fan_out}, scaled_lr={scaled_lr}",
        #     logger=logger,
        #     level=logging.INFO,
        # )

        return scaled_lr

    def _get_global_lr(self, param_name: str, param: nn.Parameter, module: nn.Module) -> float:
        return self.lr

    def get_lr(self, param_name: str, param: nn.Parameter) -> float:
        """Return the learning rate for the given parameter."""
        # NOTE: param_name should be like 'model.token_position_embeddings.pp_block.token_embedding.weight'
        # since names_to_modules map module_name to module
        # so we remove the .weight and .bias from param_name to get the module_name
        module_name = param_name.rsplit(".", 1)[0]
        module = self.names_to_modules[module_name]
        return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, param, module)
