import math
from abc import abstractmethod

import torch
from nanotron.config import ModelArgs
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from torch import nn
from torch.nn import init


class Parametrizator:
    def __init__(self, config: ModelArgs):
        self.config = config

    @abstractmethod
    def parametrize(self, param_name: str, module: torch.Tensor):
        pass


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
        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_row_linear(self, param_name: str, module: nn.Module):
        if "weight" == param_name:
            std = self.std / math.sqrt(2 * self.num_layers)
            init.normal_(module.weight, mean=0.0, std=std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        if "weight" == param_name:
            # TODO @thomasw21: Sometimes we actually want 0
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)

    def parametrize(self, param_name: str, module: nn.Module):
        if not isinstance(module, tuple(self.MODULE_TO_PARAMETRIZE.keys())):
            raise Exception(f"Parameter {param_name} was not initialized")

        return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, module)


class SpectralMupParametrizator(Parametrizator):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_mup_weight,
            TensorParallelRowLinear: self._parametrize_mup_weight,
            TritonRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }

        # self.std = config.init_method.std
        # NOTE:
        self.std = 1.0
        self.num_layers = config.model_config.num_hidden_layers

        self.hidden_size = config.model_config.hidden_size
        self.intermediate_size = config.model_config.intermediate_size

    def _compute_spectral_std(self, std: float, fan_in: int, fan_out: int):
        return (std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))

    def _parametrize_mup_weight(self, param_name: str, module: torch.Tensor):
        assert param_name in ["weight", "bias"]

        data = module.weight if param_name == "weight" else module.bias
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(data)
        std = self._compute_spectral_std(init_std=self.std, fan_in=fan_in, fan_out=fan_out)
        init.normal_(data, mean=0.0, std=std)

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        if "weight" == param_name:
            # TODO @thomasw21: Sometimes we actually want 0
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)

    def parametrize(self, name: str, data: torch.Tensor, module: nn.Module):
        if not isinstance(module, tuple(self.MODULE_TO_PARAMETRIZE.keys())):
            raise Exception(f"Parameter {name} was not initialized")

        return self.MODULE_TO_PARAMETRIZE[type(module)](name, data)
