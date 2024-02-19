from copy import deepcopy

import pytest
import torch
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from torch import nn
from torch.optim import Adam
from utils import convert_linear_to_fp8


@pytest.mark.parametrize("learning_rate", [1, 1e-3])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.9, 0.99)])
@pytest.mark.parametrize("eps", [1e-8, 1e-3])
@pytest.mark.parametrize("weight_decay", [0, 0.5])
def test_fp8adam_optimizer_states(learning_rate, betas, eps, weight_decay):
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))
    # fp8_linear = deepcopy(linear)

    optim = Adam(linear.parameters(), learning_rate, betas, eps, weight_decay)
    fp8_optim = FP8Adam(fp8_linear.parameters(), learning_rate, betas, eps, weight_decay)

    for _ in range(4):
        optim.zero_grad()
        fp8_optim.zero_grad()

        linear(input).sum().backward()
        fp8_linear(input).sum().backward()

        optim.step()
        fp8_optim.step()

    for (_, ref_state), (_, fp8_state) in zip(optim.state.items(), fp8_optim.state.items()):
        exp_avg_fp32 = convert_tensor_from_fp8(fp8_state["exp_avg"], fp8_state["exp_avg"].fp8_meta, torch.float32)
        exp_avg_sq_32 = fp8_state["exp_avg_sq"].to(torch.float32)

        torch.testing.assert_allclose(exp_avg_fp32, ref_state["exp_avg"])
        torch.testing.assert_allclose(exp_avg_sq_32, ref_state["exp_avg_sq"])


@pytest.mark.parametrize("fp8_recipe", [FP8LM_RECIPE])
def test_fp8adam_optimizer_state_dtypes(fp8_recipe):
    exp_avg_dtype = FP8LM_RECIPE.optim.exp_avg_dtype
    exp_avg_sq_dtype = FP8LM_RECIPE.optim.exp_avg_sq_dtype

    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))

    fp8_optim = FP8Adam(fp8_linear.parameters())
    fp8_linear(input).sum().backward()
    fp8_optim.step()

    for _, fp8_state in fp8_optim.state.items():
        # TODO(xrsrke): currently testing a fixed fp8 recipe
        # support different fp8 recipes
        assert isinstance(fp8_state["exp_avg"], FP8Tensor)
        assert fp8_state["exp_avg"].fp8_meta.dtype == exp_avg_dtype

        assert not isinstance(fp8_state["exp_avg_sq"], FP8Tensor)
        assert fp8_state["exp_avg_sq"].dtype == exp_avg_sq_dtype


@pytest.mark.parametrize("learning_rate", [1e-3, 1])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.9, 0.99)])
@pytest.mark.parametrize("eps", [1e-8, 1e-3])
@pytest.mark.parametrize("weight_decay", [1e-2, 0])
def test_fp8adam_step(learning_rate, betas, eps, weight_decay):
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))

    optim = Adam(linear.parameters(), learning_rate, betas, eps, weight_decay)
    fp8_optim = FP8Adam(fp8_linear.parameters(), learning_rate, betas, eps, weight_decay)
    input = torch.randn(16, 16, device="cuda")

    for _ in range(5):
        linear(input).sum().backward()
        optim.step()
        optim.zero_grad()

        fp8_linear(input).sum().backward()
        fp8_optim.step()
        fp8_optim.zero_grad()

    weight_fp32 = convert_tensor_from_fp8(fp8_linear.weight.data, fp8_linear.weight.data.fp8_meta, torch.float32)
    # NOTE: this specific threshold is based on the FP8-LM implementation
    # the paper shows that it don't hurt convergence
    torch.testing.assert_allclose(weight_fp32, linear.weight, rtol=0.1, atol=3e-4)
    torch.testing.assert_allclose(fp8_linear.bias, linear.bias, rtol=0, atol=3e-4)


def test_fp8adam_zero_grad():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=1e-3)
    fp8_linear(input).sum().backward()
    fp8_optim.step()

    assert [p.grad is not None for p in fp8_linear.parameters()]

    fp8_optim.zero_grad()

    assert [p.grad is None for p in fp8_linear.parameters()]


def test_fp8adam_load_state_dict():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))
    # fp8_linear = deepcopy(linear)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=1e-3)
    fp8_linear(input).sum().backward()
    fp8_optim.step()

    saved_state_dict = fp8_optim.state_dict()
    new_fp8_optim = FP8Adam(fp8_linear.parameters(), lr=1e-3)
    new_fp8_optim.load_state_dict(saved_state_dict)

    for param_group_new, param_group_saved in zip(new_fp8_optim.param_groups, fp8_optim.param_groups):
        torch.testing.assert_allclose(param_group_new["lr"], param_group_saved["lr"])

    for state_new, state_saved in zip(new_fp8_optim.state.values(), fp8_optim.state.values()):
        for key in state_saved:
            torch.testing.assert_allclose(state_new[key], state_saved[key])


def test_fp8adam_grad_accumulation():
    pass


# TODO(xrsrke): add sanity check all parameters are FP8Parameter
