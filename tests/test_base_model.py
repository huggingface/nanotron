import pytest
import torch
import torch.distributed as dist
from nanotron.config import Config, ModelArgs, RandomInit
from nanotron.models.base import init_on_device_and_dtype
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.testing.llama import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config
from nanotron.testing.utils import init_distributed, rerun_if_address_is_in_use
from torch import nn


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
def test_override_dtype_and_device_in_module_init(dtype, device):
    class ModuleWithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(2, 2))
            self.weight = nn.Parameter(torch.randn(2, 2))

    with init_on_device_and_dtype(device=device, dtype=dtype):
        linear = ModuleWithBuffer()

    assert all(p.dtype == dtype for p in linear.parameters())
    assert all(p.device == device for p in linear.parameters())

    assert all(b.dtype == dtype for b in linear.buffers())
    assert all(b.device == device for b in linear.buffers())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
@rerun_if_address_is_in_use()
def test_dtype_of_model_initialization(dtype: torch.dtype, device: torch.device):
    init_distributed(tp=1, dp=1, pp=1)(_test_dtype_of_model_initialization)(dtype=dtype, device=device)


def _test_dtype_of_model_initialization(parallel_context: ParallelContext, dtype: torch.dtype, device: torch.device):
    model_args = ModelArgs(init_method=RandomInit(std=1.0), model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)
    llama = create_llama_from_config(
        model_config=TINY_LLAMA_CONFIG, device=device, parallel_context=parallel_context, dtype=dtype
    )
    llama.init_model_randomly(config=config)

    assert all(p.dtype == dtype for p in llama.parameters())
    assert all(p.device == device for p in llama.parameters())

    # assert all(b.dtype == dtype for b in llama.buffers())
    # NOTE: we explicitly cast inv_freq to float32, so skip it
    assert all(b.dtype == dtype for n, b in llama.named_buffers() if "inv_freq" not in n)
    assert all(b.device == device for b in llama.buffers())


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1), (2, 2, 2)])
@pytest.mark.skip
@rerun_if_address_is_in_use()
def test_get_named_modules_in_pp_rank(tp: int, dp: int, pp: int):
    model_args = ModelArgs(init_method=RandomInit(std=1.0), model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_get_named_modules_in_pp_rank)(config=config)


def _test_get_named_modules_in_pp_rank(
    parallel_context: ParallelContext,
    config: Config,
):
    model = create_llama_from_config(
        model_config=config.model.model_config,
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
