from copy import deepcopy

import pytest
import torch
from helpers.llama import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import ModelArgs, RandomInit
from nanotron.config.parallelism_config import DominoArgs
from nanotron.models.llama import DominoLlamaDecoderLayer
from nanotron.parallel import ParallelContext
from nanotron.parallel.comm import AsyncCommBucket
from nanotron.parallel.tensor_parallel.domino import is_domino_async_comm


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

    llama_model = create_llama_from_config(
        model_config=config.model.model_config,
        parallel_config=config.parallelism,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    llama_model.init_model_randomly(config=config)

    for m in llama_model.model.decoder:
        assert isinstance(m.pp_block, DominoLlamaDecoderLayer)

    input_ids = torch.randint(0, config.model.model_config.vocab_size, size=(batch_size, seq_len), device="cuda")
    input_mask = torch.ones_like(input_ids)
    outputs = llama_model(input_ids, input_mask, input_mask, input_mask)

    assert isinstance(outputs["loss"], torch.Tensor)
    assert AsyncCommBucket.is_all_completed() is True
