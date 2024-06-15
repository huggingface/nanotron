from typing import cast

import pytest
import torch
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.tensor import FP8Tensor, FP16Tensor, convert_tensor_from_fp8, convert_tensor_from_fp16
from nanotron.fp8.utils import get_leaf_modules
from nanotron.helpers import init_optimizer_and_grad_accumulator
from nanotron.optim import NamedOptimizer
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.nn import FP8TensorParallelColumnLinear, FP8TensorParallelRowLinear
from nanotron.scaling.parametrization import ParametrizationMethod
from nanotron.testing.utils import DEFAULT_OPTIMIZER_CONFIG, create_nanotron_model


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_fp8_adam_states(
    tp: int,
    dp: int,
    pp: int,
):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_adam_states)(input_ids=input_ids, input_mask=input_mask)


def _test_fp8_adam_states(parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor):
    input_ids = input_ids.to("cuda")
    input_mask = input_mask.to("cuda")
    nanotron_model = create_nanotron_model(parallel_context, dtype=torch.int8)

    optimizer, _ = init_optimizer_and_grad_accumulator(
        parametrization_method=ParametrizationMethod.STANDARD,
        model=nanotron_model,
        optimizer_args=DEFAULT_OPTIMIZER_CONFIG,
        parallel_context=parallel_context,
    )
    fp8_optimizer = optimizer.get_base_optimizer()

    for i in range(3):
        optimizer.zero_grad()
        nanotron_model.model(input_ids, input_mask).sum().backward()
        optimizer.step()

        assert all(x["exp_avg"].dtype == torch.float32 for x in list(fp8_optimizer.state.values()))
        assert all(x["exp_avg_sq"].dtype == torch.float32 for x in list(fp8_optimizer.state.values()))
        assert all(x["step"] == i + 1 for x in list(fp8_optimizer.state.values()))


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
    assert isinstance(optimizer.get_base_optimizer(), FP8Adam)

    fp8_optim = optimizer.get_base_optimizer()
    modules = get_leaf_modules(model)
    num_fp8_modules = sum([isinstance(module, tuple(FP8_MODULES)) for module in model.modules()])
    fp8_params = [
        param for _, module in modules for param in module.parameters() if isinstance(module, tuple(FP8_MODULES))
    ]
    num_master_weights = len(fp8_optim.mappping_fp8_to_master_weight)

    assert all(p.__class__ == FP16Tensor for p in fp8_optim.mappping_fp8_to_master_weight.values())
    assert num_fp8_modules == num_master_weights
    assert sum([1 for fp8_param in fp8_params if fp8_param in fp8_optim.mappping_fp8_to_master_weight.keys()]) == len(
        fp8_params
    )


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_fp8adam_not_change_memory_address(
    tp: int,
    dp: int,
    pp: int,
):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8adam_not_change_memory_address)(
        input_ids=input_ids, input_mask=input_mask
    )


def _test_fp8adam_not_change_memory_address(
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

    param_mem_address_before_step = [id(p) for param_group in optimizer.param_groups for p in param_group["params"]]
    logits = nanotron_model.model(input_ids, input_mask)
    logits.sum().backward()
    optimizer.step()
    param_mem_address_after_step = [id(p) for param_group in optimizer.param_groups for p in param_group["params"]]

    assert param_mem_address_before_step == param_mem_address_after_step


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

    for param_group in optimizer.get_base_optimizer().param_groups:
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

    for _ in range(3):
        param_data_before_step = [param.data.clone() for param in nanotron_model.parameters()]

        optimizer.zero_grad()
        logits = nanotron_model.model(input_ids, input_mask)
        logits.sum().backward()
        optimizer.step()

        param_data_after_step = [param.data.clone() for param in nanotron_model.parameters()]

        for p1, p2 in zip(param_data_before_step, param_data_after_step):
            if p1.data.__class__ == FP8Tensor:
                p1 = convert_tensor_from_fp8(p1, p1.fp8_meta, torch.float32)
                p2 = convert_tensor_from_fp8(p2, p2.fp8_meta, torch.float32)

            assert not torch.allclose(p1, p2)


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_master_params_are_updated_after_fp8_adam_step(
    tp: int,
    dp: int,
    pp: int,
):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_master_params_are_updated_after_fp8_adam_step)(
        input_ids=input_ids, input_mask=input_mask
    )


def _test_master_params_are_updated_after_fp8_adam_step(
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
    fp8_optimizer = optimizer.get_base_optimizer()

    master_params_before_step = [
        convert_tensor_from_fp16(p, torch.float32).clone()
        for p in list(fp8_optimizer.mappping_fp8_to_master_weight.values())
    ]

    optimizer.zero_grad()
    nanotron_model.model(input_ids, input_mask).sum().backward()
    optimizer.step()

    master_params_after_step = [
        convert_tensor_from_fp16(p, torch.float32).clone()
        for p in list(fp8_optimizer.mappping_fp8_to_master_weight.values())
    ]

    for p1, p2 in zip(master_params_before_step, master_params_after_step):
        assert not torch.allclose(p1, p2)


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_optimizer_states_are_updated_after_fp8_adam_step(
    tp: int,
    dp: int,
    pp: int,
):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_optimizer_states_are_updated_after_fp8_adam_step)(
        input_ids=input_ids, input_mask=input_mask
    )


def _test_optimizer_states_are_updated_after_fp8_adam_step(
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
    fp8_optimizer = optimizer.get_base_optimizer()

    exp_avg_before_step = []
    exp_avg_sq_before_step = []
    exp_avg_after_step = []
    exp_avg_sq_after_step = []

    for i in range(2):
        optimizer.zero_grad()
        nanotron_model.model(input_ids, input_mask).sum().backward()
        optimizer.step()

        if i == 0:
            for state in list(fp8_optimizer.state.values()):
                exp_avg_before_step.append(state["exp_avg"].clone())
                exp_avg_sq_before_step.append(state["exp_avg_sq"].clone())
        elif i == 1:
            for state in list(fp8_optimizer.state.values()):
                exp_avg_after_step.append(state["exp_avg"].clone())
                exp_avg_sq_after_step.append(state["exp_avg_sq"].clone())

    for v1, v2 in zip(exp_avg_before_step, exp_avg_after_step):
        assert not torch.allclose(v1, v2)

    for v1, v2 in zip(exp_avg_sq_before_step, exp_avg_sq_after_step):
        assert not torch.allclose(v1, v2)
