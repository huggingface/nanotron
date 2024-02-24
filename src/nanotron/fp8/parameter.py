from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from nanotron.fp8.constants import FP8_DTYPES, INITIAL_AMAX, INITIAL_SCALING_FACTOR
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor, update_scaling_factor


@dataclass
class FP8GradMeta:
    """FP8 metadata for FP8Linear."""

    input_grad: FP8Meta
    weight_grad: FP8Meta
    output_grad: FP8Meta


class FP8Parameter(nn.Parameter):
    """
    A custom FP8 parameter class that allows gradients
    to flow into FP8 tensors (which are integer tensors).
    """

    def __new__(cls, data: torch.Tensor, dtype: DTypes, requires_grad: bool = True) -> nn.Parameter:
        assert isinstance(data, torch.Tensor), "data must be a tensor"
        assert data.dtype not in FP8_DTYPES, "Currently only support turn a non-fp8 tensor to an fp8 parameter"
        assert data.device != torch.device("cpu"), "FP8Parameter only supports CUDA tensors"

        with torch.no_grad():
            # TODO(xrsrke): support take an FP8 Tensor as data
            # currently we can't only quantize a tensor to FP8 after the parameter is created
            # because it raise "Only Tensors of floating point and complex dtype can require gradients"
            # TODO(xrsrke): delete this fp32 tensor from memory after quantization
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self._data = FP8Tensor(data, dtype=dtype)
            # TODO(xrsrke): don't store fp32 raw data in memory after quantization
            self.orig_data = data.data

            # TODO(xrsrke): don't fixed these, take it from the FP8 recipe
            fp8e4m3_scale = update_scaling_factor(
                amax=torch.tensor(INITIAL_AMAX, dtype=torch.float32),
                scaling_factor=torch.tensor(INITIAL_SCALING_FACTOR),
                dtype=DTypes.FP8E4M3,
            )
            fp8e5m2_scale = update_scaling_factor(
                amax=torch.tensor(INITIAL_AMAX, dtype=torch.float32),
                scaling_factor=torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32),
                dtype=DTypes.FP8E5M2,
            )

            # TODO(xrsrke): add type hints of fp8_grad_meta to FP8Parameter
            self.fp8_grad_meta = FP8GradMeta(
                input_grad=FP8Meta(amax=INITIAL_AMAX, dtype=DTypes.FP8E4M3, scale=fp8e4m3_scale),
                # TODO(xrsrke): change weight_grad to data_grad
                # because this is the gradient of the parameter itself
                weight_grad=FP8Meta(amax=INITIAL_AMAX, dtype=DTypes.FP8E4M3, scale=fp8e4m3_scale),
                # kfloat8_e5m2
                output_grad=FP8Meta(amax=INITIAL_AMAX, dtype=DTypes.FP8E5M2, scale=fp8e5m2_scale),
            )
            self._grad = None

        return self

    @property
    def data(self) -> FP8Tensor:
        return self._data

    @data.setter
    def data(self, data: FP8Tensor):
        self._data = data

    # # NOTE: because pytorch don't allow to assign an int grad to a tensor
    # # so we bypass it by using a property
    @property
    def grad(self) -> Optional[Union[torch.Tensor, FP8Tensor]]:
        return self.data._grad

    @grad.setter
    def grad(self, value: Optional[Union[torch.Tensor, FP8Tensor]]):
        self.data._grad = value

    @property
    def fp8_meta(self) -> FP8Meta:
        return self.data.fp8_meta

    def __repr__(self) -> str:
        return f"FP8Parameter({self.data}, fp8_meta={self.fp8_meta}, requires_grad={self.requires_grad}, fp8_grad_meta={self.fp8_grad_meta})"

    # def __setattr__(self, name: str, value: Any):
    #     if name == "grad":
    #         # NOTE: bypass RuntimeError: assigned grad has data of a different type
    #         # beca(use pytorch don't allow to assign an int grad to a tensor
    #         # TODO(xrsrke): allow get fp8 gradients using param.grad instead of param.fp8_grad
    #         # self.fp8_grad = value
    #         self.data._grad = value
    #     else:
    #         return super().__setattr__(name, value)
