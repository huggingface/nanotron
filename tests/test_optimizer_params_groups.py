import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron.optim.gradient_accumulator import FP32GradientAccumulator
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.optim.optimizer_from_gradient_accumulator import OptimizerFromGradientAccumulator
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.random import set_random_seed


class DummyModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(10, 20, bias=False).to(dtype=dtype)
        self.fc2 = nn.Linear(20, 2, bias=False).to(dtype=dtype)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def test_optimizer_lr_one_group():
    set_random_seed(42)

    model = DummyModel().to("cuda")

    lr1 = 0.1

    named_params_or_groups = []
    for name, param in model.named_parameters():
        named_params_or_groups.append((name, param))
    named_params_or_groups = [{"named_params": named_params_or_groups, "lr": lr1}]

    optimizer = NamedOptimizer(
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=lambda param_groups: optim.SGD(
            param_groups,
            lr=9999999,  # this is a dummy value that should be overwritten by the lr in the named_params_or_groups
        ),
    )

    input = torch.randn(10, 10).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for _ in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output, target)
        loss.backward()

        fc1_grad = model.fc1.weight.grad.clone()
        fc2_grad = model.fc2.weight.grad.clone()

        # compute gradient manually
        with torch.no_grad():
            expected_fc1_weight = model.fc1.weight - lr1 * fc1_grad
            expected_fc2_weight = model.fc2.weight - lr1 * fc2_grad

        optimizer.step()

        updated_fc1_weight = model.fc1.weight
        updated_fc2_weight = model.fc2.weight

        torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
        torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)


def test_optimizer_lr_multiple_group():
    set_random_seed(42)

    model = DummyModel().to("cuda")

    lr1, lr2 = 0.1, 0.001

    named_params_or_groups = [
        {"named_params": [(name, param) for name, param in model.named_parameters() if "fc1" in name], "lr": lr1},
        {"named_params": [(name, param) for name, param in model.named_parameters() if "fc2" in name], "lr": lr2},
    ]

    optimizer = NamedOptimizer(
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=lambda param_groups: optim.SGD(
            param_groups,
            lr=9999999,  # this is a dummy value that should be overwritten by the lr in the named_params_or_groups
        ),
    )

    input = torch.randn(10, 10).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for _ in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output, target)
        loss.backward()

        fc1_grad = model.fc1.weight.grad.clone()
        fc2_grad = model.fc2.weight.grad.clone()

        with torch.no_grad():
            expected_fc1_weight = model.fc1.weight - lr1 * fc1_grad
            expected_fc2_weight = model.fc2.weight - lr2 * fc2_grad

        optimizer.step()

        updated_fc1_weight = model.fc1.weight
        updated_fc2_weight = model.fc2.weight

        torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
        torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)


def test_optimizer_lr_weight_decay_one_group():
    set_random_seed(42)

    model = DummyModel().to("cuda")

    lr1 = 0.1
    weight_decay = 0.1

    named_params_or_groups = []
    for name, param in model.named_parameters():
        named_params_or_groups.append((name, param))
    named_params_or_groups = [{"named_params": named_params_or_groups, "lr": lr1, "weight_decay": weight_decay}]

    optimizer = NamedOptimizer(
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=lambda param_groups: optim.SGD(
            param_groups,
            lr=9999999,  # this is a dummy value that should be overwritten by the lr in the named_params_or_groups
        ),
    )

    input = torch.randn(10, 10).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for _ in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Compute gradient manually and apply weight decay
        with torch.no_grad():
            expected_fc1_weight = (1 - lr1 * weight_decay) * model.fc1.weight - lr1 * model.fc1.weight.grad
            expected_fc2_weight = (1 - lr1 * weight_decay) * model.fc2.weight - lr1 * model.fc2.weight.grad

        optimizer.step()

        updated_fc1_weight = model.fc1.weight
        updated_fc2_weight = model.fc2.weight

        torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
        torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)


def test_optimizer_lr_weight_decay_multiple_group():
    set_random_seed(42)

    model = DummyModel().to("cuda")

    lr1, lr2 = 0.1, 0.001
    weight_decay1, weight_decay2 = 0.1, 0.001

    named_params_or_groups = [
        {
            "named_params": [(name, param) for name, param in model.named_parameters() if "fc1" in name],
            "lr": lr1,
            "weight_decay": weight_decay1,
        },
        {
            "named_params": [(name, param) for name, param in model.named_parameters() if "fc2" in name],
            "lr": lr2,
            "weight_decay": weight_decay2,
        },
    ]

    optimizer = NamedOptimizer(
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=lambda param_groups: optim.SGD(
            param_groups,
            lr=9999999,  # this is a dummy value that should be overwritten by the lr in the named_params_or_groups
        ),
    )

    input = torch.randn(10, 10).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for _ in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Compute gradient manually and apply weight decay
        with torch.no_grad():
            expected_fc1_weight = (1 - lr1 * weight_decay1) * model.fc1.weight - lr1 * model.fc1.weight.grad
            expected_fc2_weight = (1 - lr2 * weight_decay2) * model.fc2.weight - lr2 * model.fc2.weight.grad

        optimizer.step()

        updated_fc1_weight = model.fc1.weight
        updated_fc2_weight = model.fc2.weight

        torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
        torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("accumulation_steps", [1, 10])
def test_optimizer_grad_accumulation_lr_one_group(half_precision: torch.dtype, accumulation_steps: int):
    set_random_seed(42)
    dtype = half_precision
    lr1 = 0.1

    model = DummyModel(dtype=dtype).to("cuda")

    # Need to convert the weights to NanotronParameter for the gradient accumulation to work
    model.fc1.weight = NanotronParameter(model.fc1.weight)
    model.fc2.weight = NanotronParameter(model.fc2.weight)

    named_params_or_groups = []
    for name, param in model.named_parameters():
        named_params_or_groups.append((name, param))

    named_params_or_groups = [{"named_params": named_params_or_groups, "lr": lr1}]

    # Optimizer
    def optimizer_builder(inp_param_groups):
        return NamedOptimizer(
            named_params_or_groups=inp_param_groups,
            optimizer_builder=lambda param_groups: optim.SGD(
                param_groups,
                lr=9999999,  # this is a dummy value that should be overwritten by the lr in the named_params_or_groups
            ),
        )

    optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=optimizer_builder,
    )

    accumulator = optimizer.gradient_accumulator

    input = torch.randn(10, 10, dtype=dtype).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for batch_idx in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output.float(), target)
        accumulator.backward(loss)

        if (batch_idx + 1) % accumulation_steps == 0:

            # Manual update weights for ref
            with torch.no_grad():
                fc1_grad = accumulator.get_grad_buffer(name="fc1.weight").to(dtype)
                expected_fc1_weight = model.fc1.weight - lr1 * fc1_grad

                fc2_grad = accumulator.get_grad_buffer(name="fc2.weight").to(dtype)
                expected_fc2_weight = model.fc2.weight - lr1 * fc2_grad

            optimizer.step()

            updated_fc1_weight = model.fc1.weight
            updated_fc2_weight = model.fc2.weight

            torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
            torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("accumulation_steps", [1, 10])
def test_optimizer_grad_accumulation_lr_multiple_group(half_precision: torch.dtype, accumulation_steps: int):
    set_random_seed(42)
    dtype = half_precision
    lr1, lr2 = 0.1, 0.001

    model = DummyModel(dtype=dtype).to("cuda")

    # Need to convert the weights to NanotronParameter for the gradient accumulation to work
    model.fc1.weight = NanotronParameter(model.fc1.weight)
    model.fc2.weight = NanotronParameter(model.fc2.weight)

    named_params_or_groups = [
        {"named_params": [(name, param) for name, param in model.named_parameters() if "fc1" in name], "lr": lr1},
        {"named_params": [(name, param) for name, param in model.named_parameters() if "fc2" in name], "lr": lr2},
    ]

    # Optimizer
    def optimizer_builder(inp_param_groups):
        return NamedOptimizer(
            named_params_or_groups=inp_param_groups,
            optimizer_builder=lambda param_groups: optim.SGD(
                param_groups,
                lr=9999999,  # this is a dummy value that should be overwritten by the lr in the named_params_or_groups
            ),
        )

    optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=optimizer_builder,
    )

    accumulator = optimizer.gradient_accumulator

    input = torch.randn(10, 10, dtype=dtype).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for batch_idx in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output.float(), target)
        accumulator.backward(loss)

        if (batch_idx + 1) % accumulation_steps == 0:

            # Manual update weights for ref
            with torch.no_grad():
                fc1_grad = accumulator.get_grad_buffer(name="fc1.weight").to(dtype)
                expected_fc1_weight = model.fc1.weight - lr1 * fc1_grad

                fc2_grad = accumulator.get_grad_buffer(name="fc2.weight").to(dtype)
                expected_fc2_weight = model.fc2.weight - lr2 * fc2_grad

            optimizer.step()

            updated_fc1_weight = model.fc1.weight
            updated_fc2_weight = model.fc2.weight

            torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
            torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("accumulation_steps", [1, 10])
def test_optimizer_grad_accumulation_lr_weight_decay_one_group(half_precision: torch.dtype, accumulation_steps: int):
    set_random_seed(42)
    dtype = half_precision
    lr1 = 0.1
    weight_decay = 0.1

    model = DummyModel(dtype=dtype).to("cuda")

    # Need to convert the weights to NanotronParameter for the gradient accumulation to work
    model.fc1.weight = NanotronParameter(model.fc1.weight)
    model.fc2.weight = NanotronParameter(model.fc2.weight)

    named_params_or_groups = []
    for name, param in model.named_parameters():
        named_params_or_groups.append((name, param))
    named_params_or_groups = [{"named_params": named_params_or_groups, "lr": lr1, "weight_decay": weight_decay}]

    # Optimizer
    def optimizer_builder(inp_param_groups):
        return NamedOptimizer(
            named_params_or_groups=inp_param_groups,
            optimizer_builder=lambda param_groups: optim.SGD(
                param_groups,
                lr=9999999,  # this is a dummy value that will be overwritten by the lr in the named_params_or_groups
                weight_decay=9999999,  # this is a dummy value that will be overwritten by the weight_decay in the named_params_or_groups
            ),
        )

    optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=optimizer_builder,
    )

    accumulator = optimizer.gradient_accumulator

    input = torch.randn(10, 10, dtype=dtype).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for batch_idx in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output.float(), target)
        accumulator.backward(loss)

        if (batch_idx + 1) % accumulation_steps == 0:

            # Manual update weights for ref
            with torch.no_grad():
                fc1_grad = accumulator.get_grad_buffer(name="fc1.weight").to(dtype)
                expected_fc1_weight = (1 - lr1 * weight_decay) * model.fc1.weight - lr1 * fc1_grad

                fc2_grad = accumulator.get_grad_buffer(name="fc2.weight").to(dtype)
                expected_fc2_weight = (1 - lr1 * weight_decay) * model.fc2.weight - lr1 * fc2_grad

            optimizer.step()

            updated_fc1_weight = model.fc1.weight
            updated_fc2_weight = model.fc2.weight

            torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
            torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("accumulation_steps", [1, 10])
def test_optimizer_grad_accumulation_lr_weight_decay_multiple_group(
    half_precision: torch.dtype, accumulation_steps: int
):
    set_random_seed(42)
    dtype = half_precision
    lr1, lr2 = 0.1, 0.001
    weight_decay1, weight_decay2 = 0.1, 0.001

    model = DummyModel(dtype=dtype).to("cuda")

    # Need to convert the weights to NanotronParameter for the gradient accumulation to work
    model.fc1.weight = NanotronParameter(model.fc1.weight)
    model.fc2.weight = NanotronParameter(model.fc2.weight)

    named_params_or_groups = [
        {
            "named_params": [(name, param) for name, param in model.named_parameters() if "fc1" in name],
            "lr": lr1,
            "weight_decay": weight_decay1,
        },
        {
            "named_params": [(name, param) for name, param in model.named_parameters() if "fc2" in name],
            "lr": lr2,
            "weight_decay": weight_decay2,
        },
    ]
    # Optimizer
    def optimizer_builder(inp_param_groups):
        return NamedOptimizer(
            named_params_or_groups=inp_param_groups,
            optimizer_builder=lambda param_groups: optim.SGD(
                param_groups,
                lr=9999999,  # this is a dummy value that will be overwritten by the lr in the named_params_or_groups
                weight_decay=9999999,  # this is a dummy value that will be overwritten by the weight_decay in the named_params_or_groups
            ),
        )

    optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=optimizer_builder,
    )

    accumulator = optimizer.gradient_accumulator

    input = torch.randn(10, 10, dtype=dtype).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for batch_idx in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output.float(), target)
        accumulator.backward(loss)

        if (batch_idx + 1) % accumulation_steps == 0:

            # Manual update weights for ref
            with torch.no_grad():
                fc1_grad = accumulator.get_grad_buffer(name="fc1.weight").to(dtype)
                expected_fc1_weight = (1 - lr1 * weight_decay1) * model.fc1.weight - lr1 * fc1_grad

                fc2_grad = accumulator.get_grad_buffer(name="fc2.weight").to(dtype)
                expected_fc2_weight = (1 - lr2 * weight_decay2) * model.fc2.weight - lr2 * fc2_grad

            optimizer.step()

            updated_fc1_weight = model.fc1.weight
            updated_fc2_weight = model.fc2.weight

            torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
            torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)


@pytest.mark.skipif(available_gpus() < 2, reason="Testing requires at least 2 gpus")
@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("accumulation_steps", [1, 10])
@rerun_if_address_is_in_use()
def test_ddp_optimizer_grad_accumulation_lr_weight_decay_multiple_group(
    half_precision: torch.dtype, accumulation_steps: int
):
    init_distributed(tp=1, dp=2, pp=1)(_test_ddp_optimizer_grad_accumulation_lr_weight_decay_multiple_group)(
        half_precision=half_precision,
        accumulation_steps=accumulation_steps,
    )


def _test_ddp_optimizer_grad_accumulation_lr_weight_decay_multiple_group(
    parallel_context: ParallelContext, half_precision: torch.dtype, accumulation_steps: int
):
    set_random_seed(42)
    dtype = half_precision
    # Making it bigger so that the difference is more visible during update
    lr1, lr2 = 0.04, 0.05
    weight_decay1, weight_decay2 = 0.5, 0.2

    model = DummyModel(dtype=dtype).to("cuda")
    # Need to convert the weights to NanotronParameter for the gradient accumulation to work
    model.fc1.weight = NanotronParameter(model.fc1.weight)
    model.fc2.weight = NanotronParameter(model.fc2.weight)

    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        process_group=parallel_context.dp_pg,
    )

    named_params_or_groups = [
        {
            "named_params": [(name, param) for name, param in model_ddp.named_parameters() if "fc1" in name],
            "lr": lr1,
            "weight_decay": weight_decay1,
        },
        {
            "named_params": [(name, param) for name, param in model_ddp.named_parameters() if "fc2" in name],
            "lr": lr2,
            "weight_decay": weight_decay2,
        },
    ]
    # Optimizer
    def optimizer_builder(inp_param_groups):
        return NamedOptimizer(
            named_params_or_groups=inp_param_groups,
            optimizer_builder=lambda param_groups: optim.SGD(
                param_groups,
                lr=9999999,  # this is a dummy value that will be overwritten by the lr in the named_params_or_groups
                weight_decay=9999999,  # this is a dummy value that will be overwritten by the weight_decay in the named_params_or_groups
            ),
        )

    optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=optimizer_builder,
    )

    accumulator = optimizer.gradient_accumulator

    input = torch.randn(10, 10, dtype=dtype).to(device="cuda")
    target = torch.randint(0, 2, (10,)).to(device="cuda")

    for batch_idx in range(100):
        optimizer.zero_grad()

        output = model(input)
        loss = F.cross_entropy(output.float(), target)
        accumulator.backward(loss)

        if (batch_idx + 1) % accumulation_steps == 0:

            # Manual update weights for ref
            with torch.no_grad():
                fc1_grad = accumulator.get_grad_buffer(name="module.fc1.weight").to(dtype)
                expected_fc1_weight = (1 - lr1 * weight_decay1) * model.fc1.weight - lr1 * fc1_grad

                fc2_grad = accumulator.get_grad_buffer(name="module.fc2.weight").to(dtype)
                expected_fc2_weight = (1 - lr2 * weight_decay2) * model.fc2.weight - lr2 * fc2_grad

            optimizer.step()

            updated_fc1_weight = model.fc1.weight
            updated_fc2_weight = model.fc2.weight

            torch.testing.assert_close(expected_fc1_weight, updated_fc1_weight)
            torch.testing.assert_close(expected_fc2_weight, updated_fc2_weight)
