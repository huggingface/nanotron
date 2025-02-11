import pytest
from nanotron.parallel.tensor_parallel.domino import is_async_comm


@pytest.mark.parametrize(
    "op_name, expected",
    [
        ("fwd.layer_attn_1_batch_0", True),
        ("fwd.layer_attn_1_batch_1", True),
        ("fwd.layer_mlp_1_batch_0", True),
        ("fwd.layer_mlp_1_batch_1", False),
        ("bwd.layer_mlp_1_batch_1", True),
        ("bwd.layer_mlp_1_batch_0", True),
        ("bwd.layer_attn_1_batch_1", True),
        ("bwd.layer_attn_1_batch_0", False),
    ],
)
def test_is_async_comm(op_name, expected):
    assert is_async_comm(op_name) == expected
