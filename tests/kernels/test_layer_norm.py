import pytest
import torch
from torch.nn import LayerNorm

from nanotron.nn.layer_norm import TritonLayerNorm


@pytest.mark.fa2
@pytest.mark.parametrize(
    "hidden_size",
    [1024, 1025],  # fused layer norm supports 1024 as hidden size but not 1025
)
def test_fused_layer_norm(hidden_size):
    BATCH_SIZE = 5
    SEQ_LEN = 128
    DEVICE, DTYPE = torch.device("cuda:0"), torch.float16
    inputs = torch.rand(BATCH_SIZE, SEQ_LEN, hidden_size, device=DEVICE, dtype=DTYPE)

    layer_norm = LayerNorm(normalized_shape=inputs.size(-1), device=DEVICE, dtype=DTYPE)
    ref_outputs = layer_norm(inputs)

    fused_layer_norm = TritonLayerNorm(
        normalized_shape=inputs.size(-1),
        device=DEVICE,
        dtype=DTYPE,
    )
    outputs = fused_layer_norm(inputs)

    # NOTE: with torch.float16, FA2's use a atol of 1e-2
    # https://github.com/Dao-AILab/flash-attention/blob/87a1277653fc55cd615f5341255e00c69d5c00a1/tests/ops/triton/test_layer_norm.py#L63-L64
    torch.testing.assert_close(outputs, ref_outputs, rtol=1e-3, atol=1e-2)

    outputs.sum().backward()
    ref_outputs.sum().backward()

    # NOTE: same as above
    torch.testing.assert_close(fused_layer_norm.weight.grad, layer_norm.weight.grad, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(fused_layer_norm.bias.grad, layer_norm.bias.grad, rtol=1e-3, atol=1e-2)
