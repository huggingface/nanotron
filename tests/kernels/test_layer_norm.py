import importlib

from nanotron.kernels.layer_norm import MixedFusedLayerNorm
from torch.nn import LayerNorm


def test_load_fused_kernel():
    try:
        importlib.import_module("fused_layer_norm_cuda")
    except ImportError as e:
        raise e


def test_layer_norm():
    from transformers import BertTokenizer
    from transformers.models.bert.modeling_bert import BertModel

    bert = BertModel.from_pretrained("bert-base-cased").cuda().half()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    test_text = "Persistence is all you need." "Hello world from nanotron."  # 32

    tokens = tokenizer(
        [test_text] * 4,
        return_tensors="pt",
    )

    # [bsz, seq_len, d_model]
    embedding_output = (
        bert.embeddings(
            input_ids=tokens["input_ids"].cuda(),
            position_ids=None,
            token_type_ids=tokens["token_type_ids"].cuda(),
            inputs_embeds=None,
            past_key_values_length=0,
        )
        .cuda()
        .half()
    )

    fused_layernorm_layer = MixedFusedLayerNorm(normalized_shape=embedding_output.size(-1)).cuda().half()

    torch_layernorm_layer = LayerNorm(normalized_shape=embedding_output.size(-1)).cuda().half()

    fused_output = fused_layernorm_layer(embedding_output)
    torch_output = torch_layernorm_layer(embedding_output)
    test_result = (fused_output - torch_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_layer_norm"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_output[-1][-1][:5].tolist()}"
            f"\n > torch_values={torch_output[-1][-1][:5].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_layer_norm"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_output[-1][-1][:5].tolist()}, "
            f"\n > torch_values={torch_output[-1][-1][:5].tolist()}"
        )
