from copy import deepcopy
from typing import Optional

import pytest
import torch
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.loss_scaler import LossScaler
from nanotron.fp8.optim import Adam as REFAdam
from nanotron.fp8.optim import (
    FP8Adam,
)
from nanotron.fp8.tensor import (
    FP8Tensor,
    FP16Tensor,
    convert_tensor_from_fp8,
)
from nanotron.fp8.utils import convert_linear_to_fp8, convert_to_fp8_module
from torch import nn
from torch.optim import Adam


def set_fake_fp8_grads(linear: FP8Linear, ref_linear: Optional[nn.Linear] = None) -> FP8Linear:
    weight_grad = (
        torch.randn_like(linear.weight.data, device="cuda", dtype=torch.float32)
        if ref_linear is None
        else ref_linear.weight.grad
    )
    weight_grad = FP8Tensor(deepcopy(weight_grad), dtype=FP8LM_RECIPE.linear.weight_grad.dtype)

    bias_grad = (
        torch.randn_like(linear.bias.data, device="cuda", dtype=torch.float32)
        if ref_linear is None
        else ref_linear.bias.grad
    )
    bias_grad = FP16Tensor(deepcopy(bias_grad), dtype=DTypes.KFLOAT16)

    linear.weight.grad = weight_grad
    linear.bias.grad = bias_grad
    return linear


def test_fp8_optim_default_initiation():
    OPTIM_ATTRS = ["lr", "betas", "eps", "weight_decay", "amsgrad"]
    linear = nn.Linear(16, 16, device="cuda")
    ref_optim = Adam(linear.parameters())

    fp8_linear = convert_linear_to_fp8(deepcopy(linear), accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_linear.parameters())

    assert all(ref_optim.defaults[attr] == fp8_optim.defaults[attr] for attr in OPTIM_ATTRS)


# def test_fp8_optim_param_groups():
#     linear = nn.Linear(16, 16, device="cuda")
#     fp8_linear = convert_linear_to_fp8(deepcopy(linear), accum_qtype=DTypes.KFLOAT16)
#     fp8_optim = FP8Adam(fp8_linear.parameters())

#     assert 1 == 1


@pytest.mark.parametrize("n_layers", [1, 10])
def test_fp8adam_params(n_layers):
    linear = nn.Sequential(
        *[layer for _ in range(n_layers) for layer in (nn.Linear(16, 16, device="cuda"), nn.ReLU())]
    )
    ref_linear = deepcopy(linear)
    fp8_linear = convert_to_fp8_module(deepcopy(linear))

    ref_optim = Adam(ref_linear.parameters())
    fp8_optim = FP8Adam(fp8_linear.parameters())

    for ref_param_group, fp8_param_group in zip(ref_optim.param_groups, fp8_optim.param_groups):
        assert len(ref_param_group["params"]) == len(fp8_param_group["params"])


def test_fp8_optim_master_weights_fp16_and_fp8():
    ref_model = nn.Sequential(
        *[
            layer
            for _ in range(3)
            for layer in (nn.Linear(16, 64, device="cuda"), nn.ReLU(), nn.Linear(64, 16, device="cuda"), nn.ReLU())
        ]
    )
    fp8_model = convert_to_fp8_module(deepcopy(ref_model), accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_model.parameters())

    REF_PARAMS = list(ref_model.parameters())
    REF_TOTAL_PARAMS = sum([p.numel() for p in ref_model.parameters()])

    assert isinstance(fp8_optim.master_weights, list)
    assert all(isinstance(w, FP16Tensor) for w in fp8_optim.master_weights)
    # TODO(xrsrke): retrieve the dtype from the FP8 recipe
    assert all(w.dtype == torch.float16 for w in fp8_optim.master_weights)
    assert REF_TOTAL_PARAMS == sum([w.numel() for w in fp8_optim.master_weights])
    assert all(w.shape == p.shape for w, p in zip(fp8_optim.master_weights, REF_PARAMS))

    assert isinstance(fp8_optim.fp8_weights, list)
    # NOTE: we keep bias as FP32, and only quantize weight
    # to FP8, bias has ndim == 1
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
    fp8_linear = convert_linear_to_fp8(deepcopy(linear), accum_qtype=DTypes.KFLOAT16)

    optim = Adam(linear.parameters(), learning_rate, betas, eps, weight_decay)
    fp8_optim = FP8Adam(fp8_linear.parameters(), learning_rate, betas, eps, weight_decay)

    for _ in range(2):
        optim.zero_grad()
        fp8_optim.zero_grad()

        linear(input).sum().backward()
        fp8_linear(input).sum().backward()

        optim.step()
        fp8_optim.step()

    for (_, ref_state), (_, fp8_state) in zip(optim.state.items(), fp8_optim.state.items()):
        # fp32_exp_avg = convert_tensor_from_fp8(fp8_state["exp_avg"], fp8_state["exp_avg"].fp8_meta, torch.float32)
        # fp32_exp_avg_sq = convert_tensor_from_fp16(fp8_state["exp_avg_sq"], torch.float32)
        fp32_exp_avg = fp8_state["exp_avg"]
        fp32_exp_avg_sq = fp8_state["exp_avg_sq"]

        # NOTE: i'm not sure this should be the target threshold
        # but i assume that if two tensors are equal, then the difference should be
        # from quantization error, so i take these thresholds from fp8's quantization error's threshold
        torch.testing.assert_allclose(fp32_exp_avg, ref_state["exp_avg"], rtol=0.1, atol=0.1)
        # torch.testing.assert_allclose(fp32_exp_avg_sq, ref_state["exp_avg_sq"], rtol=0.1, atol=0.1)
        # torch.testing.assert_allclose(fp32_exp_avg_sq, ref_state["exp_avg_sq"], rtol=0, atol=1e-03) # fp16's quantization error's threshold
        # torch.testing.assert_allclose(fp32_exp_avg_sq, ref_state["exp_avg_sq"], rtol=0, atol=1e-02)
        torch.testing.assert_allclose(fp32_exp_avg_sq, ref_state["exp_avg_sq"], rtol=0.1, atol=0.1)


@pytest.mark.skip(reason="not implemented yet")
def test_fp8adam_optimizer_state_dtypes():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear), accum_qtype=DTypes.KFLOAT16)

    fp8_optim = FP8Adam(fp8_linear.parameters())
    fp8_linear(input).sum().backward()
    fp8_optim.step()

    for _, fp8_state in fp8_optim.state.items():
        # TODO(xrsrke): currently testing a fixed fp8 recipe
        # support different fp8 recipes
        assert isinstance(fp8_state["exp_avg"], FP8Tensor)
        assert fp8_state["exp_avg"].fp8_meta.dtype == FP8LM_RECIPE.optim.exp_avg_dtype

        assert isinstance(fp8_state["exp_avg_sq"], FP16Tensor)
        assert fp8_state["exp_avg_sq"].dtype == torch.float16


@pytest.mark.parametrize("n_steps", [1, 5])
def test_fp8adam_step(n_steps):
    LR, BETAS, EPS, WEIGHT_DECAY = (0.001, (0.9, 0.999), 1e-08, 0)

    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear), accum_qtype=DTypes.KFLOAT16)
    before_amax = fp8_linear.weight.data.fp8_meta.amax

    # optim = Adam(linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)
    # fp8_optim = FP8Adam(fp8_linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)
    optim = Adam(linear.parameters(), weight_decay=WEIGHT_DECAY)
    fp8_optim = FP8Adam(fp8_linear.parameters(), weight_decay=WEIGHT_DECAY)

    for _ in range(n_steps):
        linear(input).sum().backward()
        optim.step()
        optim.zero_grad()

        fp8_linear(input).sum().backward()
        fp8_optim.step()
        fp8_optim.zero_grad()

    # NOTE: since optimizer update depends on the gradients
    # and in this test we only want to check whether fp8 optim step is correct
    # so we will set the gradients to the target one, and only check the optim step

    # NOTE: sanity check if the parameters
    # are updated at all
    after_amax = fp8_linear.weight.data.fp8_meta.amax

    assert not torch.equal(before_amax, after_amax)

    weight_fp32 = convert_tensor_from_fp8(fp8_linear.weight.data, fp8_linear.weight.data.fp8_meta, torch.float32)
    # NOTE: this specific threshold is based on the FP8-LM implementation
    # the paper shows that it don't hurt convergence
    torch.testing.assert_allclose(weight_fp32, linear.weight, rtol=0, atol=3e-4)
    torch.testing.assert_allclose(fp8_linear.bias, linear.bias, rtol=0, atol=3e-4)


def test_fp8adam_zero_grad():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear), accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=1e-3)
    fp8_linear(input).sum().backward()

    assert [p.grad is not None for p in fp8_linear.parameters()]

    fp8_optim.zero_grad()

    assert [p.grad is None for p in fp8_linear.parameters()]


# NOTE: this is just for sanity check, will remove this once fp8 training works
# @pytest.mark.parametrize("n_layers", [1, 5])
@pytest.mark.parametrize("is_bias", [True, False])
@pytest.mark.parametrize("n_steps", [1, 5])
def test_if_gradient_flows_with_multiple_layers(is_bias, n_steps):
    n_layers = 1
    HIDDEN_SIZE = 64

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=True)
    ref_model = nn.Sequential(
        *[
            layer
            for _ in range(n_layers)
            for layer in (nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=is_bias, device="cuda"), nn.ReLU())
        ][:-1]
    )
    ref_optim = Adam(ref_model.parameters(), lr=1e-3)

    fp8_model = convert_to_fp8_module(deepcopy(ref_model))
    fp8_optim = FP8Adam(fp8_model.parameters(), lr=1e-3)

    for _ in range(n_steps):
        # NOTE: we intentionally call .zero_grad() first because if using
        # torch.Optimizer's default .zero_grad() method,
        # it causes gradients not flowing
        ref_optim.zero_grad()
        ref_model(input).sum().backward()
        ref_optim.step()

        fp8_optim.zero_grad()
        fp8_model(input).sum().backward()
        fp8_optim.step()

    assert all(p.grad is not None for p in fp8_model.parameters())
    assert all(isinstance(p.grad, FP8Tensor) for p in fp8_model.parameters() if p.grad.ndim > 1)
    assert all(isinstance(p.grad, torch.Tensor) for p in fp8_model.parameters() if p.grad.ndim == 1)


@pytest.mark.parametrize(
    "scaling_value",
    [
        torch.tensor(2, dtype=torch.float32).pow(1),
        torch.tensor(2, dtype=torch.float32).pow(2),
        torch.tensor(2, dtype=torch.float32).pow(3),
        torch.tensor(2, dtype=torch.float32).pow(4),
        torch.tensor(2, dtype=torch.float32).pow(5),
        torch.tensor(2, dtype=torch.float32).pow(6),
        torch.tensor(2, dtype=torch.float32).pow(7),
        torch.tensor(2, dtype=torch.float32).pow(8),
        torch.tensor(2, dtype=torch.float32).pow(9),
        torch.tensor(2, dtype=torch.float32).pow(10),
        torch.tensor(2, dtype=torch.float32).pow(11),
        torch.tensor(2, dtype=torch.float32).pow(12),
        torch.tensor(2, dtype=torch.float32).pow(13),
        torch.tensor(2, dtype=torch.float32).pow(14),
        torch.tensor(2, dtype=torch.float32).pow(15),
        torch.tensor(32768, dtype=torch.float32),
    ],
)
def test_fp8adam_step_with_loss_scaling(scaling_value):
    LR, BETAS, EPS, WEIGHT_DECAY = (0.001, (0.9, 0.999), 1e-08, 0)

    input = torch.randn(16, 16, device="cuda")
    ref_linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(ref_linear), accum_qtype=DTypes.KFLOAT16)
    loss_scaler = LossScaler(scaling_value)

    ref_optim = Adam(ref_linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)
    fp8_optim = FP8Adam(fp8_linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)

    for _ in range(1):
        ref_loss = ref_linear(input).sum()
        ref_loss.backward()
        ref_optim.step()
        ref_optim.zero_grad()

        loss = fp8_linear(input).sum()
        scaled_loss = loss_scaler.scale(loss)

        # TODO(xrsrke): remove these sanity checks after debugging
        assert not torch.allclose(scaled_loss, ref_loss)

        scaled_loss.backward()
        loss_scaler.step(fp8_optim)
        fp8_optim.zero_grad()

    # NOTE: since optimizer update depends on the gradients
    # and in this test we only want to check whether fp8 optim step is correct
    # so we will set the gradients to the target one, and only check the optim step

    # NOTE: this specific threshold is based on the FP8-LM implementation
    # the paper shows that it don't hurt convergence
    # source: https://github.com/Azure/MS-AMP/blob/0a2cd721fa68ee725e3b2fb132df02ddb8069d62/tests/optim/test_adamw.py#L85
    torch.testing.assert_close(fp8_optim.fp32_p.data, ref_linear.weight, rtol=0, atol=3e-4)
    # assert fp8_optim.updated_p_fp8.fp8_meta == fp8_linear.weight.data.fp8_meta

    # NOTE: sanity check if we quantize the correct unquantized params
    # quantized_fp32_p = FP8Tensor(fp8_optim.fp32_p.data, fp8_linear.weight.data.fp8_meta.dtype)
    # assert torch.equal(quantized_fp32_p, fp8_linear.weight.data)
    # # assert quantized_fp32_p.fp8_meta == fp8_linear.weight.data.fp8_meta
    # unquantized_quantized_fp32_p = convert_tensor_from_fp8(quantized_fp32_p, quantized_fp32_p.fp8_meta, torch.float32)
    weight_fp32 = convert_tensor_from_fp8(fp8_linear.weight.data, fp8_linear.weight.data.fp8_meta, torch.float32)

    # assert torch.equal(unquantized_quantized_fp32_p, weight_fp32)

    torch.testing.assert_allclose(fp8_linear.bias, ref_linear.bias, rtol=0, atol=3e-4)
    torch.testing.assert_allclose(weight_fp32, ref_linear.weight, rtol=0, atol=3e-4)


def test_fp8adam_step_with_correct_grad():
    LR = 1e-3
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    WEIGHT_DECAY = 1e-3

    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear), accum_qtype=DTypes.KFLOAT16)

    optim = REFAdam(linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)
    fp8_optim = FP8Adam(fp8_linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)

    linear(input).sum().backward()

    fp8_linear.weight.grad = FP8Tensor(deepcopy(linear.weight.grad), dtype=FP8LM_RECIPE.linear.weight_grad.dtype)
    # fp8_linear.bias.grad = FP16Tensor(deepcopy(linear.bias.grad), dtype=DTypes.KFLOAT16)
    fp8_linear.bias.grad = deepcopy(linear.bias.grad).to(dtype=torch.float16)

    optim.step()
    fp8_optim.step()

    # NOTE: since optimizer update depends on the gradients
    # and in this test we only want to check whether fp8 optim step is correct
    # so we will set the gradients to the target one, and only check the optim step

    weight_fp32 = convert_tensor_from_fp8(fp8_linear.weight.data, fp8_linear.weight.data.fp8_meta, torch.float32)
    # bias_fp32 = _convert_tensor_from_fp16(fp8_linear.bias.data, fp8_linear.bias.fp8_meta, torch.float32)

    # torch.testing.assert_close(bias_fp32, linear.bias, rtol=0, atol=3e-4)

    # NOTE: this specific threshold is based on the FP8-LM implementation
    # the paper shows that it don't hurt convergence
    # reference: https://github.com/Azure/MS-AMP/blob/51f34acdb4a8cf06e0c58185de05198a955ba3db/tests/optim/test_adamw.py#L85
    torch.testing.assert_close(fp8_optim.fp32_p.data[:], linear.weight[:], rtol=0, atol=3e-4)
    torch.testing.assert_allclose(weight_fp32, linear.weight, rtol=0, atol=3e-4)
    # torch.testing.assert_allclose(fp8_linear.bias, linear.bias, rtol=0, atol=3e-4)


def test_fp8adam_load_state_dict():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear), accum_qtype=DTypes.KFLOAT16)
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


def test_fp8adam_not_change_memory_address():
    input = torch.randn(16, 16, device="cuda")
    linear = nn.Sequential(*[layer for _ in range(5) for layer in (nn.Linear(16, 16, device="cuda"), nn.ReLU())])
    fp8_linear = convert_to_fp8_module(linear)
    fp8_optim = FP8Adam(fp8_linear.parameters())
    # note: the memory address of params in param_grou[s]
    before_param_mem_addres = []
    for param_group in fp8_optim.param_groups:
        for p in param_group["params"]:
            before_param_mem_addres.append(id(p))

    for _ in range(5):
        fp8_optim.zero_grad()
        fp8_linear(input).sum().backward()
        fp8_optim.step()

    after_param_mem_addres = []
    for param_group in fp8_optim.param_groups:
        for p in param_group["params"]:
            after_param_mem_addres.append(id(p))

    assert before_param_mem_addres == after_param_mem_addres


# def test_fp8_parameter_track_amaxs():
#     pass


# def test_fp8_parameter_delay_scaling_factor_update():
#     pass


# TODO(xrsrke): add sanity check all parameters are FP8Parameter

# TODO(xrsrke): add gradient accumulation test
