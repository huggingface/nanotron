from copy import deepcopy

import pytest
import torch
from helpers.llama import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import ModelArgs, RandomInit
from nanotron.config.parallelism_config import DominoArgs
from nanotron.models.llama import DominoLlamaDecoderLayer
from nanotron.parallel import ParallelContext
from nanotron.parallel.comm import CudaStreamManager
from nanotron.parallel.tensor_parallel.domino import OperationContext, get_op_name, is_domino_async_comm


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
def test_is_domino_async_comm(op_name, expected):
    assert is_domino_async_comm(op_name) == expected


@pytest.mark.parametrize("tp,dp,pp", [(2, 2, 1)])
@rerun_if_address_is_in_use()
def test_domino_model(tp: int, dp: int, pp: int):
    BATCH_SIZE, SEQ_LEN = 10, 128

    model_config = deepcopy(TINY_LLAMA_CONFIG)
    model_config.num_hidden_layers = 28
    model_args = ModelArgs(init_method=RandomInit(std=1.0), model_config=TINY_LLAMA_CONFIG)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_domino_model)(
        model_args=model_args, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
    )


def _test_domino_model(
    parallel_context: ParallelContext,
    model_args: ModelArgs,
    batch_size: int,
    seq_len: int,
):
    config = get_llama_training_config(model_args, parallel_context)
    config.parallelism.domino = DominoArgs(num_input_batches=2)
    stream_manager = CudaStreamManager()
    stream_manager.init_default_comm_stream()

    llama_model = create_llama_from_config(
        model_config=config.model.model_config,
        parallel_config=config.parallelism,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
        stream_manager=stream_manager,
    )
    llama_model.init_model_randomly(config=config)

    for m in llama_model.model.decoder:
        assert isinstance(m.pp_block, DominoLlamaDecoderLayer)

    input_ids = torch.randint(0, config.model.model_config.vocab_size, size=(batch_size, seq_len), device="cuda")
    input_mask = torch.ones_like(input_ids)
    outputs = llama_model(input_ids, input_mask, input_mask, input_mask)

    assert isinstance(outputs["loss"], torch.Tensor)
    assert stream_manager.comm_bucket.is_all_completed() is True


### OperationContext tests ###


def test_op_name_context_reentry():
    assert get_op_name() is None
    context = OperationContext("reusable_op")

    with context:
        assert get_op_name() == "reusable_op"

    assert get_op_name() is None

    with context:
        assert get_op_name() == "reusable_op"

    assert get_op_name() is None


def test_deeply_nested_contexts():
    with OperationContext("level1"):
        assert get_op_name() == "level1"

        with OperationContext("level2"):
            assert get_op_name() == "level2"

        assert get_op_name() == "level1"


def test_multiple_sequential_contexts():
    assert get_op_name() is None

    with OperationContext("first_op"):
        assert get_op_name() == "first_op"

    with OperationContext("second_op"):
        assert get_op_name() == "second_op"

    assert get_op_name() is None
