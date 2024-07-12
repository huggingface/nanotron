from typing import cast

import nanotron.fp8.distributed as dist
import pytest
import torch
from nanotron.fp8 import constants
from nanotron.fp8.optim import Adam, FP8Adam
from nanotron.fp8.tensor import FP8Tensor, FP16Tensor, convert_tensor_from_fp8, convert_tensor_from_fp16
from nanotron.fp8.utils import get_leaf_modules
from nanotron.helpers import init_optimizer_and_grad_accumulator

# from torch.optim import Adam
from nanotron.models.base import NanotronModel
from nanotron.optim import NamedOptimizer
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import FP8TensorParallelColumnLinear, FP8TensorParallelRowLinear
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.scaling.parametrization import ParametrizationMethod
from nanotron.testing.parallel import init_distributed, rerun_if_address_is_in_use
from nanotron.testing.utils import DEFAULT_OPTIMIZER_CONFIG, create_nanotron_model
from torch import nn


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_fp8_optim_default_initiation(
    tp: int,
    dp: int,
    pp: int,
):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_optim_default_initiation)()


def _test_fp8_optim_default_initiation(parallel_context: ParallelContext):
    in_features = 32
    out_features_per_tp_rank = 16

    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    class FP8Model(FP8TensorParallelColumnLinear, NanotronModel):
        def init_model_randomly(self):
            pass

    column_linear = FP8Model(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=False,
    )
    ref_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False, device="cuda")

    optim, _ = init_optimizer_and_grad_accumulator(
        parametrization_method=ParametrizationMethod.STANDARD,
        model=column_linear,
        optimizer_args=DEFAULT_OPTIMIZER_CONFIG,
        parallel_context=parallel_context,
    )
    ref_optim = Adam(
        ref_linear.parameters(),
        lr=DEFAULT_OPTIMIZER_CONFIG.learning_rate_scheduler.learning_rate,
        betas=(
            DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_beta1,
            DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_beta2,
        ),
        eps=DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_eps,
        weight_decay=DEFAULT_OPTIMIZER_CONFIG.weight_decay,
    )

    assert all(ref_optim.defaults[attr] == optim.get_base_optimizer().defaults[attr] for attr in OPTIM_ATTRS)

    parallel_context.destroy()


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


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@pytest.mark.parametrize("n_steps", [1, 3])
@rerun_if_address_is_in_use()
def test_fp8_adam_optimizer_states(
    tp: int,
    dp: int,
    pp: int,
    n_steps: int,
):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_adam_optimizer_states)(n_steps=n_steps)


def _test_fp8_adam_optimizer_states(parallel_context: ParallelContext, n_steps: int):
    batch_size = 16
    in_features = 32
    out_features_per_tp_rank = 16

    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    class FP8Model(FP8TensorParallelColumnLinear, NanotronModel):
        def init_model_randomly(self):
            pass

    column_linear = FP8Model(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=False,
    )
    ref_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False, device="cuda")

    # Copy weights/bias from sharded to un-sharded
    with torch.inference_mode():
        dist.all_gather(
            tensor_list=list(ref_linear.weight.split(out_features_per_tp_rank, dim=0)),
            tensor=column_linear.weight.data,
            group=parallel_context.tp_pg,
        )

    optim, _ = init_optimizer_and_grad_accumulator(
        parametrization_method=ParametrizationMethod.STANDARD,
        model=column_linear,
        optimizer_args=DEFAULT_OPTIMIZER_CONFIG,
        parallel_context=parallel_context,
    )
    ref_optim = Adam(
        ref_linear.parameters(),
        lr=DEFAULT_OPTIMIZER_CONFIG.learning_rate_scheduler.learning_rate,
        betas=(
            DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_beta1,
            DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_beta2,
        ),
        eps=DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_eps,
        weight_decay=DEFAULT_OPTIMIZER_CONFIG.weight_decay,
    )

    random_input = torch.randn(batch_size, in_features, device="cuda")
    dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)

    # dist.barrier()
    assert_tensor_synced_across_pg(random_input, pg=parallel_context.tp_pg)

    sharded_random_input = random_input.clone().contiguous()

    start_idx = dist.get_rank(parallel_context.tp_pg) * out_features_per_tp_rank
    end_idx = (dist.get_rank(parallel_context.tp_pg) + 1) * out_features_per_tp_rank
    sharded_portion = (slice(None), slice(start_idx, end_idx))

    for _ in range(n_steps):
        optim.zero_grad()
        ref_optim.zero_grad()

        sharded_output = column_linear(sharded_random_input)
        reference_output = ref_linear(random_input)

        torch.testing.assert_close(
            sharded_output.to(torch.float32),
            reference_output[sharded_portion],
            rtol=constants.FP8_RTOL_THRESHOLD,
            atol=constants.FP8_ATOL_THRESHOLD,
        )

        sharded_output.sum().backward()
        reference_output.sum().backward()

        optim.step()
        ref_optim.step()

    for (_, ref_states), (_, fp8_states) in zip(ref_optim.state.items(), optim.get_base_optimizer().state.items()):
        for (ref_name, ref_state), (fp8_name, fp8_state) in zip(ref_states.items(), fp8_states.items()):
            assert ref_name == fp8_name
            if ref_name == "exp_avg_sq":
                continue

            if ref_name in ["exp_avg", "exp_avg_sq"]:
                # NOTE: i'm not sure this should be the target threshold
                # but i assume that if two tensors are equal, then the difference should be
                # from quantization error, so i take these thresholds from fp8's quantization error's threshold
                torch.testing.assert_close(
                    fp8_state,
                    ref_state[[slice(start_idx, end_idx), slice(None)]],
                    rtol=constants.FP8_1ST_OPTIM_STATE_RTOL_THRESHOLD,
                    atol=constants.FP8_1ST_OPTIM_STATE_ATOL_THRESHOLD,
                )
                # torch.testing.assert_close(
                #     exp_avg_sq,
                #     ref_state["exp_avg_sq"][[slice(start_idx, end_idx), slice(None)]],
                #     rtol=constants.FP8_2ND_OPTIM_STATE_RTOL_THRESHOLD,
                #     atol=constants.FP8_2ND_OPTIM_STATE_ATOL_THRESHOLD
                # )
            else:
                # assert fp8_state == ref_state.item()
                assert torch.allclose(fp8_state, ref_state)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@pytest.mark.parametrize("n_steps", [1, 3])
@rerun_if_address_is_in_use()
def test_fp8_adam_parameters(
    tp: int,
    dp: int,
    pp: int,
    n_steps: int,
):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_adam_parameters)(n_steps=n_steps)


def _test_fp8_adam_parameters(parallel_context: ParallelContext, n_steps: int):
    batch_size = 16
    in_features = 32
    out_features_per_tp_rank = 16

    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    class FP8Model(FP8TensorParallelColumnLinear, NanotronModel):
        def init_model_randomly(self):
            pass

    column_linear = FP8Model(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=False,
    )
    ref_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False, device="cuda")

    # Copy weights/bias from sharded to un-sharded
    with torch.inference_mode():
        dist.all_gather(
            tensor_list=list(ref_linear.weight.split(out_features_per_tp_rank, dim=0)),
            tensor=column_linear.weight.data,
            group=parallel_context.tp_pg,
        )

    optim, _ = init_optimizer_and_grad_accumulator(
        parametrization_method=ParametrizationMethod.STANDARD,
        model=column_linear,
        optimizer_args=DEFAULT_OPTIMIZER_CONFIG,
        parallel_context=parallel_context,
    )
    ref_optim = Adam(
        ref_linear.parameters(),
        lr=DEFAULT_OPTIMIZER_CONFIG.learning_rate_scheduler.learning_rate,
        betas=(
            DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_beta1,
            DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_beta2,
        ),
        eps=DEFAULT_OPTIMIZER_CONFIG.optimizer_factory.adam_eps,
        weight_decay=DEFAULT_OPTIMIZER_CONFIG.weight_decay,
    )

    random_input = torch.randn(batch_size, in_features, device="cuda")
    dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)

    dist.barrier()
    assert_tensor_synced_across_pg(random_input, pg=parallel_context.tp_pg)

    sharded_random_input = random_input.clone().contiguous()

    start_idx = dist.get_rank(parallel_context.tp_pg) * out_features_per_tp_rank
    end_idx = (dist.get_rank(parallel_context.tp_pg) + 1) * out_features_per_tp_rank
    sharded_portion = (slice(None), slice(start_idx, end_idx))

    for _ in range(n_steps):
        optim.zero_grad()
        ref_optim.zero_grad()

        sharded_output = column_linear(sharded_random_input)
        reference_output = ref_linear(random_input)

        torch.testing.assert_close(
            sharded_output.to(torch.float32),
            reference_output[sharded_portion],
            rtol=constants.FP8_RTOL_THRESHOLD,
            atol=constants.FP8_ATOL_THRESHOLD,
        )

        sharded_output.sum().backward()
        reference_output.sum().backward()

        optim.step()
        ref_optim.step()

    for (_, ref_state), (_, fp8_state) in zip(ref_optim.state.items(), optim.get_base_optimizer().state.items()):
        exp_avg = fp8_state["exp_avg"]
        fp8_state["exp_avg_sq"]

        # NOTE: i'm not sure this should be the target threshold
        # but i assume that if two tensors are equal, then the difference should be
        # from quantization error, so i take these thresholds from fp8's quantization error's threshold
        torch.testing.assert_close(
            exp_avg,
            ref_state["exp_avg"][[slice(start_idx, end_idx), slice(None)]],
            rtol=constants.FP8_1ST_OPTIM_STATE_RTOL_THRESHOLD,
            atol=constants.FP8_1ST_OPTIM_STATE_ATOL_THRESHOLD,
        )
        # torch.testing.assert_close(
        #     exp_avg_sq,
        #     ref_state["exp_avg_sq"][[slice(start_idx, end_idx), slice(None)]],
        #     rtol=constants.FP8_2ND_OPTIM_STATE_RTOL_THRESHOLD,
        #     atol=constants.FP8_2ND_OPTIM_STATE_ATOL_THRESHOLD
        # )

    parallel_context.destroy()
