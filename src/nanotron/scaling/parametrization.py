import math
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict

from nanotron.config import ModelArgs
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from torch import nn
from torch.nn import init


NAME_TO_FAN_MAPPING = {
    # NOTE: the original
    # "token_embedding.weight": [49152, 1024],
    # "token_embedding.bias": [49152, 1],
    "token_embedding.weight": [49152, 1024],
    # "token_embedding.bias": [1, 1],
    # NOTE:
    "qkv_proj.weight": [1024, 2048],
    # "qkv_proj.bias": [1024, 1],
    "o_proj.weight": [1024, 1024],
    # "o_proj.bias": [1024, 1],
    "gate_up_proj.weight": [1024, 8192],
    # "gate_up_proj.bias": [1024, 1],
    "down_proj.weight": [4096, 1024],
    # "down_proj.bias": [4096, 1],
    "lm_head.pp_block.weight": [1024, 49152],
    # "token_embedding.bias": [1024, 1],
}

N_LAYERS = 14
WIDTH_BASE = 1024
TARGET_WIDTH = 1024

STD_BASE = 0.08
EMBED_MULTIPLIER = 10.0
BASE_LR = 0.001
WIDTH_MULTIPLIER = WIDTH_BASE/TARGET_WIDTH

NAME_TO_STD_MAPPING = {
    "token_embedding.weight": STD_BASE**2,
    "qkv_proj.weight": (STD_BASE**2)/WIDTH_MULTIPLIER,
    "o_proj.weight": (STD_BASE**2)/(2*WIDTH_MULTIPLIER*N_LAYERS),
    "gate_up_proj.weight": (STD_BASE**2)/WIDTH_MULTIPLIER,
    "down_proj.weight": (STD_BASE**2)/(2*WIDTH_MULTIPLIER*N_LAYERS),
    "lm_head.pp_block.weight": STD_BASE**2,
}


NAME_TO_LR_MAPPING = {
    "token_embedding.weight": BASE_LR,
    "qkv_proj.weight": BASE_LR/WIDTH_MULTIPLIER,
    "o_proj.weight": BASE_LR/WIDTH_MULTIPLIER,
    "gate_up_proj.weight": BASE_LR/WIDTH_MULTIPLIER,
    "down_proj.weight": BASE_LR/WIDTH_MULTIPLIER,
    "lm_head.pp_block.weight": BASE_LR,
    
    
    "input_layernorm.weight": BASE_LR,
    "input_layernorm.bias": BASE_LR,
    "post_attention_layernorm.weight": BASE_LR,
    "post_attention_layernorm.bias": BASE_LR,
    "final_layer_norm.pp_block.weight": BASE_LR,
    "final_layer_norm.pp_block.bias": BASE_LR,
}

NAME_TO_MULTIPLIER_MAPPING = {
    "token_embedding": EMBED_MULTIPLIER,
    "lm_head": 1/EMBED_MULTIPLIER, 
}


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
        # assert param_name in ["weight", "bias"]
        assert "weight" in param_name or "bias" in param_name

        if "weight" in param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)
        elif "bias" in param_name:
            module.bias.zero_()

    def _parametrize_row_linear(self, param_name: str, module: nn.Module):
        # assert param_name in ["weight", "bias"]
        assert "weight" in param_name or "bias" in param_name

        if "weight" in param_name:
            std = self.std / math.sqrt(2 * self.num_layers)
            init.normal_(module.weight, mean=0.0, std=std)
        elif "bias" in param_name:
            module.bias.zero_()

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        # assert param_name in ["weight", "bias"]
        assert "weight" in param_name or "bias" in param_name

        if "weight" in param_name:
            # TODO @thomasw21: Sometimes we actually want 0
            module.weight.fill_(1)
        elif "bias" in param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        # assert param_name in ["weight"]
        assert "weight" in param_name or "bias" in param_name
    
        if "weight" in param_name:
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
#             TensorParallelEmbedding: self._parametrize_embedding,
#         }
#         self.std = 1.0

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
#         fan_in, fan_out = init._calculate_fan_in_and_fan_out(data)
#         world_size = module.world_size

#         if isinstance(module, TensorParallelColumnLinear):
#             fan_out = fan_out * world_size
#         elif isinstance(module, TensorParallelRowLinear):
#             fan_in = fan_in * world_size
#         else:
#             raise ValueError(f"Unknown module {module}")

#         std = SpectralMupParametrizator._compute_spectral_std(std=self.std, fan_in=fan_in, fan_out=fan_out)
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
    def __init__(self, config: ModelArgs):
        self.config = config
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_mup_weight,
            TensorParallelRowLinear: self._parametrize_mup_weight,
            TensorParallelEmbedding: self._parametrize_mup_weight,
            TritonRMSNorm: self._parametrize_layer_norm,
        }
        
    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        assert "weight" in param_name or "bias" in param_name, f"Unknown parameter {param_name}"
        
        if "weight" in param_name:
            module.weight.fill_(1)
        elif "bias" in param_name:
            module.bias.zero_()

    def _parametrize_mup_weight(self, param_name: str, module: nn.Module):
        def find_std(param_name):
            for key in NAME_TO_STD_MAPPING:
                if key in param_name:
                    return NAME_TO_STD_MAPPING[key]

            return None
        
        std = find_std(param_name)
        if std is None:
            raise Exception(f"Parameter {param_name} was not initialized")
    
        data = module.weight if "weight" in param_name else module.bias
        
        init.normal_(data, mean=0.0, std=std)
        
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
#             TensorParallelEmbedding: self._get_global_lr,
#         }

#     def _get_mup_lr(self, param: nn.Parameter, module: nn.Module):
#         """
#         Parametrization 1 (Spectral parametrization)
#         Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

#         ηₗ = Θ(nₗ/nₗ₋₁)
#         """
#         fan_in, fan_out = init._calculate_fan_in_and_fan_out(param)
#         world_size = module.world_size

#         if isinstance(module, TensorParallelColumnLinear):
#             fan_out = fan_out * world_size
#         elif isinstance(module, TensorParallelRowLinear):
#             fan_in = fan_in * world_size
#         else:
#             raise ValueError(f"Unknown module {module}")

#         return self.lr * (fan_out / fan_in)

#     def _get_global_lr(self, param: nn.Parameter, module: nn.Module) -> float:
#         return self.lr

#     def get_lr(self, param_name: str, param: nn.Parameter) -> float:
#         """Return the learning rate for the given parameter."""
#         # NOTE: param_name should be like 'model.token_position_embeddings.pp_block.token_embedding.weight'
#         # since names_to_modules map module_name to module
#         # so we remove the .weight and .bias from param_name to get the module_name
#         module_name = param_name.rsplit(".", 1)[0]
#         module = self.names_to_modules[module_name]
#         return self.MODULE_TO_PARAMETRIZE[type(module)](param, module)



class LearningRateForSpectralMup:
    def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
        self.lr = lr
        self.names_to_modules = names_to_modules

    @abstractmethod
    def get_lr(self, param_name: str, module: nn.Module) -> float:
        def find_lr(param_name):
            for key in NAME_TO_LR_MAPPING:
                if key in param_name:
                    return NAME_TO_LR_MAPPING[key]

            return None

        lr = find_lr(param_name)
        if lr is None:
            raise Exception(f"Parameter {param_name} can't find lr")
        
        return lr
