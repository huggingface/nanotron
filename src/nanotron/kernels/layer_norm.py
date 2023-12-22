# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by nanotron team.

import importlib
import numbers
from typing import List, Union

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter


def _is_fast_layer_norm_available() -> bool:
    try:
        fast_layer_norm_module = importlib.util.find_spec("apex.contrib.layer_norm.layer_norm")
        return fast_layer_norm_module is not None
    except:
        return False


def _is_fused_layer_norm_available() -> bool:
    try:
        fused_layer_norm_module = importlib.util.find_spec("apex.normalization.fused_layer_norm")
        return fused_layer_norm_module is not None
    except:
        return False


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty(
        (1,),
        dtype=inp.dtype,
        device=inp.device,
        requires_grad=requires_grad,
    )
    out.data = inp.data
    return out


class _MakeViewlessTensor(torch.autograd.Function):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return _MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


class FusedLayerNorm(nn.Module):
    """This is MixedFusedLayerNorm ported from Megatron-LM.
    https://github.com/NVIDIA/Megatron-LM/blob/fab0bd693ec5be55b32c4f12a1ea44766ec63448/megatron/model/fused_layer_norm.py#L30
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Tensor],
        eps: float = 1e-5,
        no_persist_layer_norm: bool = True,
        apply_layernorm_1p: bool = False,
    ):
        super().__init__()
        # List of hiddens sizes supported in the persistentlayer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        if normalized_shape not in persist_ln_hidden_sizes or not _is_fast_layer_norm_available():
            assert (
                _is_fused_layer_norm_available() is True
            ), "FusedLayerNormAffineFunction is not available, please install apex from https://github.com/NVIDIA/apex"
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.apply_layernorm_1p = apply_layernorm_1p
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.no_persist_layer_norm = no_persist_layer_norm
        self.reset_parameters()

    def reset_parameters(self):
        if self.apply_layernorm_1p:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

        if self.no_persist_layer_norm:
            from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
            return FusedLayerNormAffineFunction.apply(
                input, weight, self.bias, self.normalized_shape, self.eps
            )
        else:
            from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

            output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = _make_viewless_tensor(inp=output, requires_grad=input.requires_grad, keep_graph=True)
            return output
