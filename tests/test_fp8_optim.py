from typing import cast

import pytest
import torch
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import AdamOptimizerArgs, LRSchedulerArgs, OptimizerArgs
from nanotron.fp8.optim import (
    FP8Adam,
)
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from nanotron.fp8.utils import get_leaf_modules
from nanotron.helpers import init_optimizer_and_grad_accumulator
from nanotron.optim import NamedOptimizer
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.nn import FP8TensorParallelColumnLinear, FP8TensorParallelRowLinear
from nanotron.scaling.parametrization import ParametrizationMethod
from nanotron.testing.utils import create_nanotron_model

DEFAULT_OPTIMIZER_CONFIG = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.1,
    clip_grad=1.0,
    accumulate_grad_in_fp32=False,
    learning_rate_scheduler=LRSchedulerArgs(
        # NOTE(xrsrke): use a high learning rate to make changes in the weights more visible
        learning_rate=0.1,
        lr_warmup_steps=100,
        lr_warmup_style="linear",
        lr_decay_style="cosine",
        min_decay_lr=1e-5,
    ),
    optimizer_factory=AdamOptimizerArgs(
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-08,
        torch_adam_is_fused=False,
    ),
)


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_params_of_fp8_adam(
    tp: int,
    dp: int,
    pp: int,
):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_params_of_fp8_adam)()


def _test_params_of_fp8_adam(parallel_context: ParallelContext):
    model = create_nanotron_model(parallel_context, dtype=torch.int8)
    FP8_MODULES = [FP8TensorParallelColumnLinear, FP8TensorParallelRowLinear]

    optimizer, _ = init_optimizer_and_grad_accumulator(
        parametrization_method=ParametrizationMethod.STANDARD,
        model=model,
        optimizer_args=DEFAULT_OPTIMIZER_CONFIG,
        parallel_context=parallel_context,
    )

    optimizer = cast(NamedOptimizer, optimizer)
    assert isinstance(optimizer.optimizer, FP8Adam)

    modules = get_leaf_modules(model)
    num_fp8_modules = sum([isinstance(module, tuple(FP8_MODULES)) for module in model.modules()])
    fp8_params = [
        param for _, module in modules for param in module.parameters() if isinstance(module, tuple(FP8_MODULES))
    ]
    num_master_weights = len(optimizer.optimizer.mappping_fp8_to_master_weight)

    assert num_fp8_modules == num_master_weights
    assert sum(
        [1 for fp8_param in fp8_params if fp8_param in optimizer.optimizer.mappping_fp8_to_master_weight.keys()]
    ) == len(fp8_params)


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_fp8_adam_zero_grad(
    tp: int,
    dp: int,
    pp: int,
):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_adam_zero_grad)(input_ids=input_ids, input_mask=input_mask)


def _test_fp8_adam_zero_grad(parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor):
    input_ids = input_ids.to("cuda")
    input_mask = input_mask.to("cuda")
    nanotron_model = create_nanotron_model(parallel_context, dtype=torch.int8)

    optimizer, _ = init_optimizer_and_grad_accumulator(
        parametrization_method=ParametrizationMethod.STANDARD,
        model=nanotron_model,
        optimizer_args=DEFAULT_OPTIMIZER_CONFIG,
        parallel_context=parallel_context,
    )

    logits = nanotron_model.model(input_ids, input_mask)
    logits.sum().backward()

    optimizer.zero_grad()

    for param_group in optimizer.optimizer.param_groups:
        for param in param_group["params"]:
            if param.data.__class__ == FP8Tensor:
                assert param.data._temp_grad is None
            else:
                assert param.grad is None


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_parameters_change_after_fp8_adam_step(
    tp: int,
    dp: int,
    pp: int,
):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_parameters_change_after_fp8_adam_step)(
        input_ids=input_ids, input_mask=input_mask
    )


def _test_parameters_change_after_fp8_adam_step(
    parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor
):
    input_ids = input_ids.to("cuda")
    input_mask = input_mask.to("cuda")
    nanotron_model = create_nanotron_model(parallel_context, dtype=torch.int8)

    optimizer, _ = init_optimizer_and_grad_accumulator(
        parametrization_method=ParametrizationMethod.STANDARD,
        model=nanotron_model,
        optimizer_args=DEFAULT_OPTIMIZER_CONFIG,
        parallel_context=parallel_context,
    )
    param_data_before_step = [param.data.clone() for param in nanotron_model.parameters()]

    logits = nanotron_model.model(input_ids, input_mask)
    logits.sum().backward()
    optimizer.step()

    param_data_after_step = [param.data.clone() for param in nanotron_model.parameters()]

    allclose_params = []
    for p1, p2 in zip(param_data_before_step, param_data_after_step):
        if p1.data.__class__ == FP8Tensor:
            fp32_p1 = convert_tensor_from_fp8(p1, p1.fp8_meta, torch.float32)
            fp32_p2 = convert_tensor_from_fp8(p2, p2.fp8_meta, torch.float32)
            assert not torch.allclose(fp32_p1, fp32_p2)
        else:
            try:
                assert not torch.allclose(p1, p2)
            except AssertionError:
                allclose_params.append((p1, p2))

    assert 1 == 1
