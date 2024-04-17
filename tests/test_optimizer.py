import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nanotron.optim.gradient_accumulator import FP32GradientAccumulator
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.optim.optimizer_from_gradient_accumulator import OptimizerFromGradientAccumulator
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


def test_optimizer_grad_accumulation_lr_one_group():
    set_random_seed(42)
    dtype = torch.bfloat16
    lr1 = 0.1
    accumulation_steps = 10

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


def test_optimizer_grad_accumulation_lr_multiple_group():
    set_random_seed(42)
    dtype = torch.bfloat16
    accumulation_steps = 10
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


def test_optimizer_grad_accumulation_lr_weight_decay_one_group():
    set_random_seed(42)
    dtype = torch.bfloat16
    accumulation_steps = 10
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


def test_optimizer_grad_accumulation_lr_weight_decay_multiple_group():
    set_random_seed(42)
    dtype = torch.bfloat16
    accumulation_steps = 10
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


if __name__ == "__main__":
    # Optimizer
    test_optimizer_lr_one_group()
    test_optimizer_lr_multiple_group()
    test_optimizer_lr_weight_decay_one_group()
    test_optimizer_lr_weight_decay_multiple_group()

    # Grad accumulation
    test_optimizer_grad_accumulation_lr_one_group()
    test_optimizer_grad_accumulation_lr_multiple_group()
    test_optimizer_grad_accumulation_lr_weight_decay_one_group()
    test_optimizer_grad_accumulation_lr_weight_decay_multiple_group()

    # TODO(fmom): Zero
    # test_optimizer_grad_accumulation_zero_lr_one_group()
    # test_optimizer_grad_accumulation_zero_lr_multiple_group()
    # test_optimizer_grad_accumulation_zero_lr_weight_decay_one_group()
    # test_optimizer_grad_accumulation_zero_lr_weight_decay_multiple_group()
