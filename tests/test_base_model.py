import pytest
import torch
import torch.distributed as dist
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import ModelArgs, RandomInit
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from torch import nn

from tests.helpers.llama_helper import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1), (2, 2, 2)])
@rerun_if_address_is_in_use()
def test_get_named_modules_in_pp_rank(tp: int, dp: int, pp: int):
    model_args = ModelArgs(init_method=RandomInit(std=1.0), model_config=TINY_LLAMA_CONFIG)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_get_named_modules_in_pp_rank)(model_args=model_args)


def _test_get_named_modules_in_pp_rank(
    parallel_context: ParallelContext,
    model_args: ModelArgs,
):
    config = get_llama_training_config(model_args, parallel_context)
    model = create_llama_from_config(
        model_config=config.model.model_config,
        parallel_config=config.parallelism,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    model.init_model_randomly(config=config)

    modules_that_not_in_current_pp_rank = {}
    current_pp_rank = dist.get_rank(group=parallel_context.pp_pg)
    for name, module in model.named_modules():
        if isinstance(module, PipelineBlock) and module.rank != current_pp_rank:
            modules_that_not_in_current_pp_rank[name] = module

    named_modules_in_pp_rank = model.named_modules_in_pp_rank

    for name, module in named_modules_in_pp_rank.items():
        # NOTE: if a module is in the current rank, we expect it to be an initialized module
        # not PipelineBlock
        assert isinstance(module, nn.Module)
        assert name not in modules_that_not_in_current_pp_rank


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1), (2, 2, 1)])
@rerun_if_address_is_in_use()
def test_llama_model(tp: int, dp: int, pp: int):
    BATCH_SIZE, SEQ_LEN = 10, 128
    model_args = ModelArgs(init_method=RandomInit(std=1.0), model_config=TINY_LLAMA_CONFIG)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_llama_model)(
        model_args=model_args, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
    )


def _test_llama_model(
    parallel_context: ParallelContext,
    model_args: ModelArgs,
    batch_size: int,
    seq_len: int,
):
    config = get_llama_training_config(model_args, parallel_context)
    llama_model = create_llama_from_config(
        model_config=config.model.model_config,
        parallel_config=config.parallelism,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    llama_model.init_model_randomly(config=config)

    input_ids = torch.randint(0, config.model.model_config.vocab_size, size=(batch_size, seq_len), device="cuda")
    input_mask = torch.ones_like(input_ids)
    outputs = llama_model(input_ids, input_mask, input_mask, input_mask)

    assert list(outputs.keys()) == ["loss"]
    assert isinstance(outputs["loss"], torch.Tensor)
