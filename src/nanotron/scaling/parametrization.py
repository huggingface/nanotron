import math
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict

from nanotron.config import Config, ModelArgs
from nanotron.config.models_config import InitScalingMethod
from nanotron.nn.layer_norm import LlamaRMSNorm, TritonRMSNorm
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from torch import nn
from torch.nn import init


class ParametrizationMethod(Enum):
    STANDARD = auto()
    SPECTRAL_MUP = auto()


class Parametrizator:
    def __init__(self, config: ModelArgs):
        self.config = config

    def parametrize(self, param_name: str, module: nn.Module):
        if not isinstance(module, tuple(self.MODULE_TO_PARAMETRIZE.keys())):
            raise Exception(f"Module {type(module)} with parameter {param_name} is not supported for initialization")

        return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, module)


class StandardParametrizator(Parametrizator):
    def __init__(self, config: Config):
        super().__init__(config)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_column_linear,
            TensorParallelRowLinear: self._parametrize_row_linear,
            TritonRMSNorm: self._parametrize_layer_norm,
            LlamaRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }

        self.std = config.model.init_method.std
        self.num_layers = config.model.model_config.num_hidden_layers
        self.tp = config.parallelism.tp
        self.scaling_method = config.model.init_method.scaling_method
        self.hidden_size = config.model.model_config.hidden_size

    def _parametrize_column_linear(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            # TODO @nouamane: should we use trunc_normal_
            init.normal_(module.weight, mean=0.0, std=self.std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _compute_scaling_factor(self) -> float:
        """Compute initialization scaling based on selected method"""
        if self.scaling_method == InitScalingMethod.NONE:
            return 1.0
        elif self.scaling_method == InitScalingMethod.NUM_LAYERS:
            # Scale based on total network depth
            return math.sqrt(2 * self.num_layers)
        elif self.scaling_method == InitScalingMethod.LAYER_INDEX:
            # Scale based on layer position
            raise NotImplementedError("Layer position scaling not yet implemented")
        else:
            raise ValueError(f"Invalid scaling method: {self.scaling_method}")

    def _parametrize_row_linear(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            scaling = self._compute_scaling_factor()
            adjusted_std = self.std / scaling
            # TODO @nouamane: should we use trunc_normal_
            init.normal_(module.weight, mean=0.0, std=adjusted_std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        assert param_name in ["weight"]

        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


class SpectralMupParametrizator(Parametrizator):
    """
    A Spectral Condition for Feature Learning by Greg Yang, et al.
    https://arxiv.org/abs/2310.17813
    """

    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_mup_weight,
            TensorParallelRowLinear: self._parametrize_mup_weight,
            TritonRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }
        self.std = 1.0

    @staticmethod
    def _compute_spectral_std(std: float, fan_in: int, fan_out: int):
        """
        Parametrization 1 (Spectral parametrization)
        Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

        σₗ = Θ(1/√nₗ₋₁ min{1, √(nₗ/nₗ₋₁)})
        """
        return (std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))

    def _parametrize_mup_weight(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        data = module.weight if param_name == "weight" else module.bias
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(data)
        world_size = module.world_size

        if isinstance(module, TensorParallelColumnLinear):
            fan_out = fan_out * world_size
        elif isinstance(module, TensorParallelRowLinear):
            fan_in = fan_in * world_size
        else:
            raise ValueError(f"Unknown module {module}")

        std = SpectralMupParametrizator._compute_spectral_std(std=self.std, fan_in=fan_in, fan_out=fan_out)
        init.normal_(data, mean=0.0, std=std)

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        # NOTE: you're free to change the initialization of layer norm
        # as it's not a part of µTransfer
        if "weight" == param_name:
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        assert param_name in ["weight"]

        # NOTE: you're free to change the initialization of input embedding/lm head
        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


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


class LearningRateForSpectralMup(LearningRateForParametrizator):
    """
    A Spectral Condition for Feature Learning by Greg Yang, et al.

    NOTE: each parameter gets a custom learning rate based on its fan-in and fan-out.
    """

    def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
        super().__init__(lr, names_to_modules)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._get_mup_lr,
            TensorParallelRowLinear: self._get_mup_lr,
            TritonRMSNorm: self._get_global_lr,
            TensorParallelEmbedding: self._get_global_lr,
        }

    def _get_mup_lr(self, param: nn.Parameter, module: nn.Module):
        """
        Parametrization 1 (Spectral parametrization)
        Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

        ηₗ = Θ(nₗ/nₗ₋₁)
        """
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(param)
        world_size = module.world_size

        if isinstance(module, TensorParallelColumnLinear):
            fan_out = fan_out * world_size
        elif isinstance(module, TensorParallelRowLinear):
            fan_in = fan_in * world_size
        else:
            raise ValueError(f"Unknown module {module}")

        return self.lr * (fan_out / fan_in)

    def _get_global_lr(self, param: nn.Parameter, module: nn.Module) -> float:
        return self.lr

    def get_lr(self, param_name: str, param: nn.Parameter) -> float:
        """Return the learning rate for the given parameter."""
        # NOTE: param_name should be like 'model.token_position_embeddings.pp_block.token_embedding.weight'
        # since names_to_modules map module_name to module
        # so we remove the .weight and .bias from param_name to get the module_name
        module_name = param_name.rsplit(".", 1)[0]
        module = self.names_to_modules[module_name]
        return self.MODULE_TO_PARAMETRIZE[type(module)](param, module)
