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
from nanotron.kernels.layer_norm import FusedLayerNorm
from torch.nn import LayerNorm
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel


@pytest.mark.parametrize("no_persist_layer_norm", [True, False])
def test_fused_layer_norm(no_persist_layer_norm):
    bert = BertModel.from_pretrained("bert-base-cased").cuda().half()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    test_text = ["Persistence is all you need.", "Hello world from nanotron."]

    tokens = tokenizer([test_text] * 4, return_tensors="pt")

    # NOTE: [bsz, seq_len, d_model]
    embedding_outputs = bert.embeddings(
        input_ids=tokens["input_ids"].cuda(),
        position_ids=None,
        token_type_ids=tokens["token_type_ids"].cuda(),
        inputs_embeds=None,
        past_key_values_length=0,
    )
    embedding_outputs = embedding_outputs.cuda().half()

    layer_norm = LayerNorm(normalized_shape=embedding_outputs.size(-1)).cuda().half()
    ref_outputs = layer_norm(embedding_outputs)

    fused_layer_norm = FusedLayerNorm(
        normalized_shape=embedding_outputs.size(-1), no_persist_layer_norm=no_persist_layer_norm
    )
    fused_layer_norm = fused_layer_norm.cuda().half()
    outputs = fused_layer_norm(embedding_outputs)

    assert torch.allclose(outputs, ref_outputs, rtol=1e-3)
