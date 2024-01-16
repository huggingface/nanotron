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
from typing import List, Optional, Union

import torch
import torch.nn as nn


def _is_fast_layer_norm_available() -> bool:
    try:
        fast_layer_norm_module = importlib.util.find_spec("apex.contrib.layer_norm.layer_norm")
        return fast_layer_norm_module is not None
    except ImportError:
        return False


def _is_fused_layer_norm_available() -> bool:
    try:
        fused_layer_norm_module = importlib.util.find_spec("apex.normalization.fused_layer_norm")
        return fused_layer_norm_module is not None
    except ImportError:
        return False


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
        device: torch.device = torch.device("cuda:0"),
        dtype: Optional[torch.dtype] = torch.float16,
    ):
        super().__init__()
        # NOTE: List of hiddens sizes supported in the persistentlayer norm kernel
        # If the hidden size is not supported, fall back to
        # the non-persistent kernel.
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
        self.weight = nn.Parameter(torch.ones(*normalized_shape, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(*normalized_shape, device=device, dtype=dtype))
        self.no_persist_layer_norm = no_persist_layer_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

        if self.no_persist_layer_norm:
            from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

            return FusedLayerNormAffineFunction.apply(input, weight, self.bias, self.normalized_shape, self.eps)
        else:
            from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

            return FastLayerNormFN.apply(input, weight, self.bias, self.eps)
