from copy import deepcopy

import pytest
import torch
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.optim import Adam as REFAdam
from nanotron.fp8.optim import (
    FP8Adam,
)

# convert_tensor_from_fp16, convert_tensor_to_fp16
from nanotron.fp8.tensor import FP8Tensor, FP16Tensor, convert_tensor_from_fp8
from torch import nn
from torch.optim import Adam
from utils import convert_linear_to_fp8, convert_to_fp8_module


def test_fp8_optim_default_initiation():
    OPTIM_ATTRS = ["lr", "betas", "eps", "weight_decay", "amsgrad"]
    linear = nn.Linear(16, 16, device="cuda")
    ref_optim = Adam(linear.parameters())

    fp8_linear = convert_linear_to_fp8(deepcopy(linear))
    fp8_optim = FP8Adam(fp8_linear.parameters())

    assert all(ref_optim.defaults[attr] == fp8_optim.defaults[attr] for attr in OPTIM_ATTRS)


def test_fp8_optim_master_weights_fp16_and_fp8():
    ref_model = nn.Sequential(
        *[
            layer
            for _ in range(3)
            for layer in (nn.Linear(16, 64, device="cuda"), nn.ReLU(), nn.Linear(64, 16, device="cuda"), nn.ReLU())
        ]
    )
    fp8_model = convert_to_fp8_module(deepcopy(ref_model))
    fp8_optim = FP8Adam(fp8_model.parameters())

    REF_PARAMS = list(ref_model.parameters())
    REF_TOTAL_PARAMS = sum([p.numel() for p in ref_model.parameters()])

    assert isinstance(fp8_optim.master_weights, list)
    # assert all(isinstance(w, torch.Tensor) for w in fp8_optim.master_weights)
    assert all(isinstance(w, FP16Tensor) for w in fp8_optim.master_weights)
    # TODO(xrsrke): retrieve the dtype from the FP8 recipe
    assert all(w.dtype == torch.float16 for w in fp8_optim.master_weights)
    assert REF_TOTAL_PARAMS == sum([w.numel() for w in fp8_optim.master_weights])
    assert all(w.shape == p.shape for w, p in zip(fp8_optim.master_weights, REF_PARAMS))

    assert isinstance(fp8_optim.fp8_weights, list)
    # NOTE: we keep bias as FP32, and only quantize weight to FP8
    # bias has ndim == 1
    assert all(isinstance(p, FP8Tensor) for p in fp8_optim.fp8_weights if p.ndim != 1)
    assert all(isinstance(p, torch.Tensor) for p in fp8_optim.fp8_weights if p.ndim != 1)
    assert REF_TOTAL_PARAMS == sum([w.numel() for w in fp8_optim.fp8_weights])
    assert all(w.shape == p.shape for w, p in zip(fp8_optim.fp8_weights, REF_PARAMS))


@pytest.mark.parametrize("learning_rate", [1, 1e-3])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.9, 0.99)])
@pytest.mark.parametrize("eps", [1e-8, 1e-3])
@pytest.mark.parametrize("weight_decay", [0, 0.5])
def test_fp8adam_optimizer_states(learning_rate, betas, eps, weight_decay):
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))

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


def test_fp8adam_optimizer_state_dtypes():
    exp_avg_dtype = FP8LM_RECIPE.optim.exp_avg_dtype

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

        assert not isinstance(fp8_state["exp_avg_sq"], FP16Tensor)
        assert fp8_state["exp_avg_sq"].dtype == torch.float16


# @pytest.mark.parametrize("n_steps", [1, 5])
@pytest.mark.parametrize(
    "learning_rate",
    [
        1e-3,
        # 1
    ],
)
@pytest.mark.parametrize(
    "betas",
    [
        (0.9, 0.999),
        # (0.9, 0.99)
    ],
)
@pytest.mark.parametrize(
    "eps",
    [
        1e-8,
        # 1e-3
    ],
)
@pytest.mark.parametrize(
    "weight_decay",
    [
        1e-3,
        # 0
    ],
)
def test_fp8adam_step(learning_rate, betas, eps, weight_decay):
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))

    optim = Adam(linear.parameters(), learning_rate, betas, eps, weight_decay)
    fp8_optim = FP8Adam(fp8_linear.parameters(), learning_rate, betas, eps, weight_decay)
    input = torch.randn(16, 16, device="cuda")

    for _ in range(1):
        linear(input).sum().backward()
        optim.step()
        optim.zero_grad()

        fp8_linear(input).sum().backward()
        fp8_optim.step()
        fp8_optim.zero_grad()

    # NOTE: since optimizer update depends on the gradients
    # and in this test we only want to check whether fp8 optim step is correct
    # so we will set the gradients to the target one, and only check the optim step

    weight_fp32 = convert_tensor_from_fp8(fp8_linear.weight.data, fp8_linear.weight.data.fp8_meta, torch.float32)
    # NOTE: this specific threshold is based on the FP8-LM implementation
    # the paper shows that it don't hurt convergence
    torch.testing.assert_allclose(weight_fp32, linear.weight, rtol=0, atol=3e-4)
    torch.testing.assert_allclose(fp8_linear.bias, linear.bias, rtol=0, atol=3e-4)


def test_fp8adam_zero_grad():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=1e-3)
    fp8_linear(input).sum().backward()

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


# @pytest.mark.parametrize("size", [4, 8, 16, 64])
# def test_quantize_and_dequantize_tensor_in_fp16(size):
#     tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
#     ref_tensor = deepcopy(tensor)

#     fp16_tensor, fp16_meta = convert_tensor_to_fp16(tensor)

#     assert isinstance(fp16_tensor, torch.Tensor)
#     assert fp16_tensor.device == ref_tensor.device
#     assert fp16_tensor.dtype == torch.float16
#     assert fp16_tensor.shape == ref_tensor.shape
#     assert fp16_tensor.numel() == ref_tensor.numel()
#     assert not np.array_equal(fp16_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

#     tensor = convert_tensor_from_fp16(fp16_tensor, fp16_meta, torch.float32)
#     assert isinstance(tensor, torch.Tensor)
#     assert tensor.dtype == torch.float32
#     assert torch.allclose(tensor, ref_tensor, rtol=0, atol=1e-03)


@pytest.mark.parametrize(
    "learning_rate",
    [
        1e-3,
        # 1
    ],
)
@pytest.mark.parametrize(
    "betas",
    [
        (0.9, 0.999),
        # (0.9, 0.99)
    ],
)
@pytest.mark.parametrize(
    "eps",
    [
        1e-8,
        # 1e-3
    ],
)
@pytest.mark.parametrize(
    "weight_decay",
    [
        1e-3,
        # 0
    ],
)
def test_fp8adam_step_fp16(learning_rate, betas, eps, weight_decay):
    linear = nn.Linear(16, 16, device="cuda")
    # fp8_linear = convert_linear_to_fp8(deepcopy(linear))
    fp16_linear = deepcopy(linear)
    fp16_linear_weight, fp16_weight_meta = convert_tensor_to_fp16(fp16_linear.weight)
    fp16_linear_bias, fp16_bias_meta = convert_tensor_to_fp16(fp16_linear.bias)
    fp16_linear.weight = nn.Parameter(fp16_linear_weight)
    fp16_linear.bias = nn.Parameter(fp16_linear_bias)

    optim = Adam(linear.parameters(), learning_rate, betas, eps, weight_decay)
    fp8_optim = FP8Adam(fp16_linear.parameters(), learning_rate, betas, eps, weight_decay)
    input = torch.randn(16, 16, device="cuda")

    # for _ in range(1):
    #     linear(input).sum().backward()
    #     optim.step()
    #     optim.zero_grad()

    #     fp8_linear(input).sum().backward()
    #     fp8_optim.step()
    #     fp8_optim.zero_grad()

    linear(input).sum().backward()
    fp16_linear.weight.grad = deepcopy(linear.weight.grad)
    fp16_linear.bias.grad = deepcopy(linear.bias.grad)

    optim.step()
    fp8_optim.step()

    # NOTE: since optimizer update depends on the gradients
    # and in this test we only want to check whether fp8 optim step is correct
    # so we will set the gradients to the target one, and only check the optim step

    # weight_fp32 = convert_tensor_from_fp8(fp16_linear.weight.data, fp16_linear.weight.data.fp8_meta, torch.float32)
    # NOTE: this specific threshold is based on the FP8-LM implementation
    # the paper shows that it don't hurt convergence
    weight_fp32 = convert_tensor_from_fp16(fp16_linear.weight, fp16_weight_meta, torch.float32)
    torch.testing.assert_allclose(weight_fp32, linear.weight, rtol=0, atol=3e-4)
    # torch.testing.assert_allclose(fp8_linear.bias, linear.bias, rtol=0, atol=3e-4)


def test_fp8adam_grad_accumulation():
    pass


# TODO(xrsrke): add sanity check all parameters are FP8Parameter


# @pytest.mark.parametrize("n_steps", [1, 5])
# @pytest.mark.parametrize(
#     "learning_rate",
#     [
#         1e-3,
#         # 1
#     ],
# )
# @pytest.mark.parametrize(
#     "betas",
#     [
#         (0.9, 0.999),
#         # (0.9, 0.99)
#     ],
# )
# @pytest.mark.parametrize(
#     "eps",
#     [
#         1e-8,
#         # 1e-3
#     ],
# )
# @pytest.mark.parametrize(
#     "weight_decay",
#     [
#         1e-3,
#         # 0
#     ],
# )
def test_fp8adam_step_with_correct_grad():
    LR = 1e-3
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    WEIGHT_DECAY = 1e-3

    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))

    optim = REFAdam(linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)
    fp8_optim = FP8Adam(fp8_linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)

    # for _ in range(1):
    #     linear(input).sum().backward()
    #     optim.step()
    #     optim.zero_grad()

    #     fp8_linear(input).sum().backward()
    #     fp8_optim.step()
    #     fp8_optim.zero_grad()

    linear(input).sum().backward()

    fp8_linear.weight.grad = FP8Tensor(deepcopy(linear.weight.grad), dtype=FP8LM_RECIPE.linear.weight_grad.dtype)
    # fp8_linear.bias.grad = deepcopy(linear.bias.grad).to(torch.float16)
    fp8_linear.bias.grad = FP16Tensor(deepcopy(linear.bias.grad), dtype=DTypes.KFLOAT16)

    optim.step()
    fp8_optim.step()

    # NOTE: since optimizer update depends on the gradients
    # and in this test we only want to check whether fp8 optim step is correct
    # so we will set the gradients to the target one, and only check the optim step

    weight_fp32 = convert_tensor_from_fp8(fp8_linear.weight.data, fp8_linear.weight.data.fp8_meta, torch.float32)
    # NOTE: this specific threshold is based on the FP8-LM implementation
    # the paper shows that it don't hurt convergence
    # reference: https://github.com/Azure/MS-AMP/blob/51f34acdb4a8cf06e0c58185de05198a955ba3db/tests/optim/test_adamw.py#L85
    torch.testing.assert_allclose(weight_fp32, linear.weight, rtol=0, atol=3e-4)
    # torch.testing.assert_allclose(fp8_linear.bias, linear.bias, rtol=0, atol=3e-4)
