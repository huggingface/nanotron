import snoop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nanotron.optim.gradient_accumulator import FP32GradientAccumulator
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.optim.optimizer_from_gradient_accumulator import OptimizerFromGradientAccumulator
from nanotron.parallel.parameters import NanotronParameter
from nanotron.random import set_random_seed

# import lovely_tensors as lt; lt.monkey_patch()


class DummyModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(DummyModel, self).__init__()
        # self.fc1 = nn.Linear(1, 2, bias=False).to(dtype=dtype)
        self.fc1 = nn.Linear(4, 4, bias=False).to(dtype=dtype)
        # self.fc2 = nn.Linear(20, 2, bias=False).to(dtype=dtype)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x


def test_optimizer_lr_one_group():
    model = DummyModel()

    lr1 = 0.01

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

    input = torch.randn(10, 10)
    target = torch.randint(0, 2, (10,))

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
    model = DummyModel()

    lr1, lr2 = 0.01, 0.001

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

    input = torch.randn(10, 10)
    target = torch.randint(0, 2, (10,))

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
    model = DummyModel()

    lr1 = 0.01
    weight_decay = 0.01

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

    input = torch.randn(10, 10)
    target = torch.randint(0, 2, (10,))

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
    model = DummyModel()

    lr1, lr2 = 0.01, 0.001
    weight_decay1, weight_decay2 = 0.01, 0.001

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

    input = torch.randn(10, 10)
    target = torch.randint(0, 2, (10,))

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


@snoop
def test_optimizer_grad_accumulation_lr_one_group():
    set_random_seed(42)
    dtype = torch.bfloat16
    lr1 = 0.01
    accumulation_steps = 1  # Number of steps to accumulate gradients

    model_ref = DummyModel(dtype=dtype).to("cuda")
    model_nanotron = DummyModel(dtype=dtype).to("cuda")
    with torch.inference_mode():
        model_nanotron.fc1.weight.copy_(model_ref.fc1.weight.data)
        # model_nanotron.fc2.weight.copy_(model_ref.fc2.weight.data)

    # Need to convert the weights to NanotronParameter for the gradient accumulation to work
    model_nanotron.fc1.weight = NanotronParameter(model_nanotron.fc1.weight)
    # model_nanotron.fc2.weight = NanotronParameter(model_nanotron.fc2.weight)

    torch.testing.assert_close(model_nanotron.fc1.weight, model_ref.fc1.weight)
    # torch.testing.assert_close(model_nanotron.fc2.weight, model_ref.fc2.weight)

    named_params_or_groups = []
    for name, param in model_nanotron.named_parameters():
        named_params_or_groups.append((name, param))

    named_parameters = named_params_or_groups
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
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(
            named_parameters=named_params,
            grad_buckets_named_params=named_parameters,
        ),
        named_params_or_groups=named_params_or_groups,
        optimizer_builder=optimizer_builder,
    )

    accumulator = optimizer.gradient_accumulator

    input = torch.randn(4, 4, dtype=dtype).to(device="cuda")
    target = torch.randint(0, 2, (4,)).to(device="cuda")

    for batch_idx in range(100):

        print(batch_idx)
        output_nanotron = model_nanotron(input)
        output_ref = model_ref(input)

        torch.testing.assert_close(output_nanotron, output_ref)

        loss_nanotron = F.cross_entropy(output_nanotron.float(), target)
        loss_ref = F.cross_entropy(output_ref.float(), target)

        torch.testing.assert_close(loss_nanotron, loss_ref)

        accumulator.backward(loss_nanotron / accumulation_steps)
        (loss_ref / accumulation_steps).backward()

        torch.testing.assert_close(
            accumulator.parameters["fc1.weight"]["fp32"].grad.to(dtype), model_ref.fc1.weight.grad
        )

        # torch.testing.assert_close(
        #     accumulator.parameters["fc2.weight"]["fp32"].grad.to(dtype),
        #     model_ref.fc2.weight.grad
        # )

        if (batch_idx + 1) % accumulation_steps == 0:
            # Manual update weights for ref
            with torch.no_grad():
                fc1_param = model_ref.fc1.weight
                model_ref.fc1.weight.copy_(fc1_param - lr1 * fc1_param.grad)

                # fc2_param = model_ref.fc2.weight
                # model_ref.fc2.weight.copy_(fc2_param - lr1 * fc2_param.grad)

            # Nanotron update weights
            optimizer.step()

            # Validate updated weights
            nanotron_fc1_weight = accumulator.parameters["fc1.weight"]["half"].data
            # nanotron_fc2_weight = accumulator.parameters["fc2.weight"]["half"].data

            torch.testing.assert_close(model_ref.fc1.weight, nanotron_fc1_weight)
            # torch.testing.assert_close(model_ref.fc2.weight, nanotron_fc2_weight)


if __name__ == "__main__":
    # TODO(fmom): convert this test to distributed settings
    # TODO(fmom): seed seed to each function

    # Optimizer
    # TODO(fmom): test using CUDA instead of CPU
    # test_optimizer_lr_one_group()
    # test_optimizer_lr_multiple_group()
    # test_optimizer_lr_weight_decay_one_group()
    # test_optimizer_lr_weight_decay_multiple_group()

    # Grad accumulation
    test_optimizer_grad_accumulation_lr_one_group()
    # test_optimizer_grad_accumulation_lr_multiple_group()
    # test_optimizer_grad_accumulation_lr_weight_decay_one_group()
    # test_optimizer_grad_accumulation_lr_weight_decay_multiple_group()

    # Zero
    # test_optimizer_grad_accumulation_zero_lr_one_group()
    # test_optimizer_grad_accumulation_zero_lr_multiple_group()
    # test_optimizer_grad_accumulation_zero_lr_weight_decay_one_group()
    # test_optimizer_grad_accumulation_zero_lr_weight_decay_multiple_group()
