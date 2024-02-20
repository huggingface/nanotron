import torch
from torch import nn

from nanotron.fp8.constants import FP8_DTYPES
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor


class FP8Parameter(nn.Parameter):
    """
    A custom FP8 parameter class that allows gradients
    to flow into FP8 tensors (which are integer tensors).
    """

    def __new__(cls, data: torch.Tensor, dtype: DTypes, requires_grad: bool = True) -> nn.Parameter:
        assert isinstance(data, torch.Tensor), "data must be a tensor"
        assert data.dtype not in FP8_DTYPES, "Currently only support turn a non-fp8 tensor to an fp8 parameter"
        assert data.device != torch.device("cpu"), "FP8Parameter only supports CUDA tensors"
        # TODO(xrsrke): if the tensor is on cpu, then bypass quantization

        with torch.no_grad():
            # TODO(xrsrke): support take an FP8 Tensor as data
            # currently we can't only quantize a tensor to FP8 after the parameter is created
            # because it raise "Only Tensors of floating point and complex dtype can require gradients"
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self._data = FP8Tensor(data, dtype=dtype)
        return self

    @property
    def data(self) -> FP8Tensor:
        return self._data

    @data.setter
    def data(self, data: FP8Tensor):
        self._data = data

    @property
    def fp8_meta(self) -> FP8Meta:
        return self.data.fp8_meta

    def __repr__(self) -> str:
        return f"FP8Parameter({self.data}, fp8_meta={self.fp8_meta}, requires_grad={self.requires_grad}"
