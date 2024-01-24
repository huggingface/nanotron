from typing import Union

import torch
from torch import nn

from nanotron.fp8.constants import DTypes
from nanotron.fp8.tensor import FP8Tensor


class FP8Parameter(nn.Parameter):
    """
    A custom FP8 parameter class that allows gradients
    to flow into FP8 tensors (which are integer tensors).
    """

    def __new__(cls, data: Union[torch.Tensor, FP8Tensor], dtype: DTypes, requires_grad: bool = True):
        with torch.no_grad():
            # TODO(xrsrke): if the tensor is on cpu, then bypass quantization
            assert data.device != torch.device("cpu"), "FP8Parameter only supports CUDA tensors"
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self._data = FP8Tensor(data, dtype=dtype) if isinstance(data, torch.Tensor) else data
        return self

    @property
    def data(self) -> FP8Tensor:
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def fp8_meta(self):
        return self.data.fp8_meta

    # def __repr__(self) -> str:
    #     return f"FP8Parameter({self.data}, fp8_meta={self.fp8_meta}, requires_grad={self.requires_grad}"
