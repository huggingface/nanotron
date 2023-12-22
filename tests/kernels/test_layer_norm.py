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
# from helpers.utils import available_gpus
from nanotron.kernels.layer_norm import FusedLayerNorm, _is_fast_layer_norm_available, _is_fused_layer_norm_available
from torch.nn import LayerNorm
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel


def test_is_fused_layer_norm_available():
    assert _is_fused_layer_norm_available() is True

    
def test_is_fast_layer_norm_available():
    assert _is_fast_layer_norm_available() is True
    

# @pytest.mark.skipif(available_gpus() < 1, reason="Testing test_fused_layer_norm requires at least 1 gpus")
@pytest.mark.parametrize("hidden_size", [
    1024, # fused layer norm supports this hidden size
    1025  # fused layer norm does not support this hidden size
])
@pytest.mark.parametrize("no_persist_layer_norm", [True, False])
def test_fused_layer_norm(hidden_size, no_persist_layer_norm):
    BATCH_SIZE = 5
    SEQ_LEN = 128
    embedding_outputs = torch.rand(BATCH_SIZE, SEQ_LEN, hidden_size).cuda().half()

    layer_norm = LayerNorm(normalized_shape=embedding_outputs.size(-1)).cuda().half()
    ref_outputs = layer_norm(embedding_outputs)

    fused_layer_norm = FusedLayerNorm(
        normalized_shape=embedding_outputs.size(-1), no_persist_layer_norm=no_persist_layer_norm
    )
    fused_layer_norm = fused_layer_norm.cuda().half()
    outputs = fused_layer_norm(embedding_outputs)

    assert torch.allclose(outputs, ref_outputs, rtol=1e-3, atol=1e-3)
    
    loss = outputs.sum()
    ref_loss = ref_outputs.sum()
    
    loss.backward()
    ref_loss.backward()
    
    assert torch.allclose(fused_layer_norm.weight.grad, layer_norm.weight.grad, rtol=1e-3)
    assert torch.allclose(fused_layer_norm.bias.grad, layer_norm.bias.grad, rtol=1e-3)
