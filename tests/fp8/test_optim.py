import torch
from nanotron.fp8.optim import FP8Adam
from torch import nn
from torch.optim import Adam
from utils import convert_to_fp8_module


def test_fp8adam_optimizer_states():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_to_fp8_module(linear)

    optim = Adam(linear.parameters(), lr=1e-3)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=1e-3)

    for _ in range(4):
        linear(input).sum().backward()
        fp8_linear(input).sum().backward()

    for (_, ref_state), (_, fp8_state) in zip(optim.state.items(), fp8_optim.state.items()):
        # TODO(xrsrke): add checking dtype based on fp8 recipe
        torch.testing.allclose(ref_state["exp_avg"], fp8_state["exp_avg"])
        torch.testing.allclose(ref_state["exp_avg_sq"], fp8_state["exp_avg_sq"])


def test_fp8adam_step():
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_to_fp8_module(linear)

    optim = Adam(linear.parameters(), lr=1e-3)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=1e-3)
    input = torch.randn(16, 16, device="cuda")

    for _ in range(5):
        linear(input).sum().backward()
        optim.step()
        optim.zero_grad()

        fp8_linear(input).sum().backward()
        fp8_optim.step()
        fp8_optim.zero_grad()

    torch.testing.assert_close(fp8_linear.weight, linear.weight, rtol=0.1, atol=3e-4)
    torch.testing.assert_close(fp8_linear.bias, linear.bias, rtol=0, atol=3e-4)


def test_fp8adam_zero_grad():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_to_fp8_module(linear)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=1e-3)
    fp8_linear(input).sum().backward()
    fp8_optim.step()

    assert [p.grad is not None for p in fp8_linear.parameters()]

    fp8_optim.zero_grad()

    assert [p.grad is None for p in fp8_linear.parameters()]


def test_fp8adam_state_dict():
    pass


def test_fp8adam_load_state_dict():
    pass


def test_fp8adam_grad_accumulation():
    pass
