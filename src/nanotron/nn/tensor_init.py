# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math
from typing import Callable

import torch


def init_method_normal(sigma: float) -> Callable[[torch.Tensor], None]:
    """Init method based on N(0, sigma)."""

    def init_(tensor: torch.Tensor):
        torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma: float, num_layers: int) -> Callable[[torch.Tensor], None]:
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor: torch.Tensor):
        torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def tensor_from_untyped_storage(untyped_storage: torch.UntypedStorage, dtype: torch.dtype):
    # TODO @thomasw21: Figure out what's the best Pytorch way of building a tensor from a storage.
    device = untyped_storage.device
    tensor = torch.empty([], dtype=dtype, device=device)
    tensor.set_(source=untyped_storage)
    return tensor
