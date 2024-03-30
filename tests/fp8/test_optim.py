from copy import deepcopy

import pytest
import torch
from nanotron.fp8.optim import Adam
from torch import nn
from torch.optim import Adam


@pytest.mark.parametrize("learning_rate", [1, 1e-3])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.9, 0.99)])
@pytest.mark.parametrize("eps", [1e-8, 1e-3])
@pytest.mark.parametrize("weight_decay", [0, 0.5])
def test_fp8adam_optimizer_states(learning_rate, betas, eps, weight_decay):
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    # fp8_linear = convert_linear_to_fp8(deepcopy(linear))
    ref_linear = deepcopy(linear)

    optim = Adam(linear.parameters(), learning_rate, betas, eps, weight_decay)
    ref_optim = Adam(ref_linear.parameters(), learning_rate, betas, eps, weight_decay)

    for _ in range(4):
        optim.zero_grad()
        ref_optim.zero_grad()

        linear(input).sum().backward()
        ref_linear(input).sum().backward()

        optim.step()
        ref_optim.step()

    for (_, ref_state), (_, fp8_state) in zip(optim.state.items(), ref_optim.state.items()):
        torch.testing.assert_allclose(ref_state["exp_avg"], fp8_state["exp_avg"])
        torch.testing.assert_allclose(ref_state["exp_avg_sq"], fp8_state["exp_avg_sq"])


def test_fp8adam_optimizer_state_dtypes():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    ref_linear = deepcopy(linear)

    ref_optim = Adam(ref_linear.parameters())
    ref_linear(input).sum().backward()
    ref_optim.step()

    for _, fp8_state in ref_optim.state.items():
        # NOTE: assert fp8 dtypes and FP8Tensor
        assert fp8_state["exp_avg"].dtype == torch.float32
        assert fp8_state["exp_avg_sq"].dtype == torch.float32


@pytest.mark.parametrize("learning_rate", [1, 1e-3])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.9, 0.99)])
@pytest.mark.parametrize("eps", [1e-8, 1e-3])
@pytest.mark.parametrize("weight_decay", [0, 0.5])
def test_fp8adam_step(learning_rate, betas, eps, weight_decay):
    linear = nn.Linear(16, 16, device="cuda")
    # fp8_linear = convert_to_fp8_module(linear)
    ref_linear = deepcopy(linear)

    optim = Adam(linear.parameters(), learning_rate, betas, eps, weight_decay)
    ref_optim = Adam(ref_linear.parameters(), learning_rate, betas, eps, weight_decay)
    input = torch.randn(16, 16, device="cuda")

    for _ in range(5):
        # NOTE: we intentionally put .zero_grad() first because using
        # torch.Optimizer's zero_grad, it causes some bugs
        # that makes gradients not flowing
        optim.zero_grad()
        linear(input).sum().backward()
        optim.step()

        ref_optim.zero_grad()
        ref_linear(input).sum().backward()
        ref_optim.step()

    torch.testing.assert_allclose(ref_linear.weight, linear.weight, rtol=0.1, atol=3e-4)
    torch.testing.assert_allclose(ref_linear.bias, linear.bias, rtol=0, atol=3e-4)


def test_fp8adam_zero_grad():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    # fp8_linear = convert_to_fp8_module(linear)
    ref_linear = deepcopy(linear)
    ref_optim = Adam(ref_linear.parameters(), lr=1e-3)
    ref_linear(input).sum().backward()
    ref_optim.step()

    assert [p.grad is not None for p in ref_linear.parameters()]

    ref_optim.zero_grad()

    assert [p.grad is None for p in ref_linear.parameters()]


def test_fp8adam_load_state_dict():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    # fp8_linear = convert_to_fp8_module(linear)
    ref_linear = deepcopy(linear)
    ref_optim = Adam(ref_linear.parameters(), lr=1e-3)
    ref_linear(input).sum().backward()
    ref_optim.step()

    saved_state_dict = ref_optim.state_dict()
    new_fp8_optim = Adam(ref_linear.parameters(), lr=1e-3)
    new_fp8_optim.load_state_dict(saved_state_dict)

    for param_group_new, param_group_saved in zip(new_fp8_optim.param_groups, ref_optim.param_groups):
        torch.testing.assert_allclose(param_group_new["lr"], param_group_saved["lr"])

    for state_new, state_saved in zip(new_fp8_optim.state.values(), ref_optim.state.values()):
        for key in state_saved:
            torch.testing.assert_allclose(state_new[key], state_saved[key])


def test_fp8adam_grad_accumulation():
    pass


# NOTE: some sanity check low-level implementation, once it works, we can remove it

# TODO(xrsrke): add sanity check all parameters are FP8Parameter
