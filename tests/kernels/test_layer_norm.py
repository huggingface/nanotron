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

import pytest
import torch
from nanotron.nn.layer_norm import TritonLayerNorm

# from helpers.utils import available_gpus
from torch.nn import LayerNorm


# @pytest.mark.skipif(available_gpus() < 1, reason="Testing test_fused_layer_norm requires at least 1 gpus")
@pytest.mark.parametrize(
    "hidden_size",
    [1024, 1025],  # fused layer norm supports 1024 as hidden size but not 1025
)
@pytest.mark.parametrize("no_persist_layer_norm", [True, False])
def test_fused_layer_norm(hidden_size, no_persist_layer_norm):
    BATCH_SIZE = 5
    SEQ_LEN = 128
    DEVICE, DTYPE = torch.device("cuda:0"), torch.float16
    inputs = torch.rand(BATCH_SIZE, SEQ_LEN, hidden_size, device=DEVICE, dtype=DTYPE)

    layer_norm = LayerNorm(normalized_shape=inputs.size(-1), device=DEVICE, dtype=DTYPE)
    ref_outputs = layer_norm(inputs)

    fused_layer_norm = TritonLayerNorm(
        normalized_shape=inputs.size(-1),
        no_persist_layer_norm=no_persist_layer_norm,
        device=DEVICE,
        dtype=DTYPE,
    )
    outputs = fused_layer_norm(inputs)

    assert torch.allclose(outputs, ref_outputs, rtol=1e-3, atol=1e-3)

    outputs.sum().backward()
    ref_outputs.sum().backward()

    assert torch.allclose(fused_layer_norm.weight.grad, layer_norm.weight.grad, rtol=1e-3)
    assert torch.allclose(fused_layer_norm.bias.grad, layer_norm.bias.grad, rtol=1e-3)
