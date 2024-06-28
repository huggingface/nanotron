import argparse
from copy import deepcopy
from dataclasses import asdict

# import deepspeed
# import msamp
import torch
import torch.nn.functional as F
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.loss_scaler import LossScaler
from nanotron.fp8.optim import Adam as RefAdam
from nanotron.fp8.utils import _log, convert_logs_to_flat_logs, convert_to_fp8_module
from torch import nn
from torch.optim import Adam

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def l1_norm_diff(loss, ref_loss):
    return (loss - ref_loss).abs().mean()


def get_cifar_dataloader(batch_size):
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ]
    )

    trainset = datasets.CIFAR10(root="./", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = datasets.CIFAR10(root="./", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    return classes, train_loader, test_loader


class Net(nn.Module):
    """Take this MLP from greg yang's mup repo"""

    def __init__(self, width, num_classes=10):
        super(Net, self).__init__()
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width * 4, bias=False)
        self.fc_3 = nn.Linear(width * 4, width * 4, bias=False)
        self.fc_4 = nn.Linear(width * 4, width, bias=False)
        self.fc_5 = nn.Linear(width, 16, bias=False)

    def forward(self, x):
        x = self.fc_5(F.relu(self.fc_4(F.relu(self.fc_3(F.relu(self.fc_2(F.relu(self.fc_1(x)))))))))
        return x[:, :10]


def format_billions(number):
    """
    Convert a number to a string with a 'b' suffix, representing billions.

    Args:
    number (float or int): The number to convert.

    Returns:
    str: The formatted number with one decimal place followed by 'b'.
    """
    return f"{number / 1_000_000_000:.1f}b"


def add_argument():
    """Add arguments."""
    parser = argparse.ArgumentParser(description="CIFAR")

    # data
    # cuda
    parser.add_argument(
        "--with_cuda", default=False, action="store_true", help="use CPU in case there's no GPU support"
    )
    parser.add_argument("--use_ema", default=False, action="store_true", help="whether use exponential moving average")

    # train
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("-e", "--epochs", default=30, type=int, help="number of total epochs (default: 30)")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")

    parser.add_argument("--log-interval", type=int, default=200, help="output logging information at a given interval")

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def check_if_overflow(optim):
    from nanotron.fp8.loss_scaler import is_overflow

    detected_overflow = False
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                if is_overflow(p.grad):
                    detected_overflow = True
                    break

    return detected_overflow


if __name__ == "__main__":
    BATCH_SIZE = 64
    INPUT_DIM = 64
    HIDDEN_SIZE = 512
    N_STEPS = 20_000
    # LR = 6e-4
    LR = 1e-3
    N_LAYERS = 16
    WITH_BIAS = True
    MODEL_NAME = "gpt2"
    DATA_NAME = "CohereForAI/aya_dataset"

    torch.cuda.empty_cache()

    fp32_linear = Net(HIDDEN_SIZE).to("cuda")

    # bf16_linear = deepcopy(fp32_linear)
    fp8_linear = deepcopy(fp32_linear)
    fp8_linear_with_scaler = deepcopy(fp32_linear)
    msamp_linear = deepcopy(fp32_linear)
    msamp_linear_with_scaler = deepcopy(fp32_linear)
    deepspeed_linear = deepcopy(fp32_linear)

    fp32_optim = RefAdam(fp32_linear.parameters(), lr=LR)

    # bf16_linear = bf16_linear.to(dtype=torch.bfloat16)
    # bf16_optim = Adam(bf16_linear.parameters(), lr=LR)

    msamp_optim = Adam(msamp_linear.parameters(), lr=LR)
    # msamp_linear, msamp_optim = msamp.initialize(msamp_linear, msamp_optim, opt_level="O2")

    msamp_optim_with_scaler = Adam(msamp_linear_with_scaler.parameters(), lr=LR)
    # msamp_linear_with_scaler, msamp_optim_with_scaler = msamp.initialize(
    #     msamp_linear_with_scaler, msamp_optim_with_scaler, opt_level="O2"
    # )

    fp8_linear = convert_to_fp8_module(fp8_linear, accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=LR)

    fp8_linear_with_scaler = convert_to_fp8_module(fp8_linear_with_scaler, accum_qtype=DTypes.KFLOAT16)
    fp8_optim_with_scaler = FP8Adam(fp8_linear_with_scaler.parameters(), lr=LR)

    msamp_scaler = torch.cuda.amp.GradScaler()
    fp8_scaler = LossScaler()

    # args = add_argument()
    # deepspeed.init_distributed()
    # deepspeed_linear, deepspeed_optim, _, _ = deepspeed.initialize(args=args, model=deepspeed_linear)

    loss_func = nn.CrossEntropyLoss()
    _, train_dataloader, test_dataloader = get_cifar_dataloader(BATCH_SIZE)

    fp32_losses = []
    fp8_with_loss_scaler_losses = []
    msamp_with_loss_scaler_losses = []

    num_params = sum(p.numel() for p in fp32_linear.parameters())
    wandb.init(
        project="fp8_for_nanotron",
        name=f"{format_billions(num_params)}b_params_and_n_layers_{N_LAYERS}_and_input_dim_{INPUT_DIM}_and_hidden_size_{HIDDEN_SIZE}_and_lr_{LR}_and_bias_{WITH_BIAS}_and_batch_size_{BATCH_SIZE}",
        config={
            "batch_size": BATCH_SIZE,
            "input_dim": INPUT_DIM,
            "hidden_size": HIDDEN_SIZE,
            "n_steps": N_STEPS,
            "lr": LR,
            "n_layers": N_LAYERS,
            "with_bias": WITH_BIAS,
            # "act_func": fp32_linear[1].__class__.__name__ if N_LAYERS > 1 else "None",
            "optim": fp32_optim.__class__.__name__,
            "optim_params": fp32_optim.defaults,
            "num_params": num_params,
            "fp8_recipe": asdict(FP8LM_RECIPE),
        },
    )

    # deepspeed_scaler = deepcopy(deepspeed_linear.optimizer.loss_scaler)

    fp32_linear_params_id_to_param_names = {id(p): n for n, p in fp32_linear.named_parameters()}
    fp8_linear_params_id_to_param_names = {id(p): n for n, p in fp8_linear.named_parameters()}
    fp8_linear_with_scaler_params_id_to_param_names = {id(p): n for n, p in fp8_linear_with_scaler.named_parameters()}

    def get_optim_logs(mappings, optim, prefix):
        optim_loggings = {}
        for p in optim.loggings:
            param_name = mappings[id(p)]
            optim_loggings[param_name] = optim.loggings[p]
        return convert_logs_to_flat_logs(optim_loggings, prefix=prefix)

    for step in range(N_STEPS):
        for step, batch in enumerate(train_dataloader):

            batch = [x.to("cuda") for x in batch]
            inputs, targets = batch
            inputs = inputs.view(inputs.size(0), -1)

            fp32_logs, fp32_handles = _log(fp32_linear)
            fp32_optim.zero_grad()
            ref_output = fp32_linear(inputs)
            fp32_loss = loss_func(ref_output, targets)
            fp32_loss.backward()
            fp32_optim.step()

            # bf16_optim.zero_grad()
            # bf16_output = bf16_linear(inputs.to(dtype=torch.bfloat16))
            # bf16_loss = loss_func(bf16_output, targets)
            # bf16_loss.backward()
            # bf16_optim.step()

            fp8_optim.zero_grad()
            fp8_logs, fp8_handles = _log(fp8_linear)
            fp8_output = fp8_linear(inputs)
            fp8_loss = loss_func(fp8_output, targets)
            fp8_loss.backward()
            fp8_optim.step()

            # # msamp_logs = _log(msamp_linear)
            # msamp_optim.zero_grad()
            # msamp_output = msamp_linear(inputs)
            # msamp_loss = loss_func(msamp_output, targets)
            # msamp_loss.backward()
            # msamp_optim.all_reduce_grads(msamp_linear)
            # msamp_optim.step()

            # # msamp_with_loss_scaler_logs = _log(msamp_linear_with_scaler)
            # msamp_optim_with_scaler.zero_grad()
            # msamp_output_with_scaler = msamp_linear_with_scaler(inputs)
            # msamp_loss_with_scaler = loss_func(msamp_output_with_scaler, targets)
            # scaled_msamp_loss_with_scaler = msamp_scaler.scale(msamp_loss_with_scaler)
            # scaled_msamp_loss_with_scaler.backward()
            # msamp_scaler.step(msamp_optim_with_scaler)
            # msamp_scaler.update()

            # deepspeed_output = deepspeed_linear(inputs.half())
            # deepspeed_loss = loss_func(deepspeed_output, targets)
            # deepspeed_linear.backward(deepspeed_loss)
            # deepspeed_linear.step()

            fp8_optim_with_scaler.zero_grad()
            fp8_with_scaler_logs, fp8_with_scaler_handles = _log(fp8_linear_with_scaler)
            fp8_output_with_scaler = fp8_linear_with_scaler(inputs)
            fp8_loss_with_scaler = loss_func(fp8_output_with_scaler, targets)
            # fp8_scaler.scaling_value = torch.tensor(deepspeed_scaler.loss_scale)
            # fp8_scaler.scaling_value = torch.tensor(deepspeed_linear.optimizer.loss_scaler.loss_scale, device="cuda")
            scaled_fp8_loss_with_scaler = fp8_scaler.scale(fp8_loss_with_scaler)
            scaled_fp8_loss_with_scaler.backward()
            # is_overflow_bool = check_if_overflow(fp8_optim_with_scaler)
            fp8_scaler.step(fp8_optim_with_scaler)

            is_overflow_bool = fp8_optim_with_scaler._is_overflow

            fp8_scaler.update()
            # deepspeed_scaler.update_scale(is_overflow_bool)

            fp32_losses.append(fp32_loss.item())
            fp8_with_loss_scaler_losses.append(fp8_loss_with_scaler.item())
            # msamp_with_loss_scaler_losses.append(msamp_loss_with_scaler.item())

            l1_norm_diff_fp8_with_loss_scaler_relative_to_fp32 = l1_norm_diff(fp8_loss_with_scaler, fp32_loss)
            # l1_norm_diff_msamp_with_loss_scaler_relative_to_fp32 = l1_norm_diff(msamp_loss_with_scaler, fp32_loss)

            # std_fp8_with_loss_scaler_relative_to_fp32 = (torch.tensor(fp8_with_loss_scaler_losses) - torch.tensor(fp32_losses)).std()
            # std_msamp_with_loss_scaler_relative_to_fp32 = (torch.tensor(msamp_with_loss_scaler_losses) - torch.tensor(fp32_losses)).std()

            # print(f"step: {step}, is_overflow={is_overflow_bool}, fp8_scaler.scaling_value: {fp8_scaler.scaling_value}")
            # print(f"step: {step}, f32_loss: {fp32_loss.item()}, fp8_loss: {fp8_loss.item()}, fp8_loss_with_scaler: {fp8_loss_with_scaler.item()}")
            print(
                f"step: {step}, is_overflow={is_overflow_bool}, f32_loss: {fp32_loss.item()}, fp8_loss: {fp8_loss.item()}"
            )

            fp32_optim_logs = get_optim_logs(
                fp32_linear_params_id_to_param_names, fp32_optim, prefix="fp32:optim_state:"
            )
            fp8_optim_logs = get_optim_logs(fp8_linear_params_id_to_param_names, fp8_optim, prefix="fp8:optim_state:")
            fp8_with_scaler_optim_logs = get_optim_logs(
                fp8_linear_with_scaler_params_id_to_param_names,
                fp8_optim_with_scaler,
                prefix="fp8_with_scaler:optim_state:",
            )

            wandb.log(
                {
                    "fp32_loss": fp32_loss.item(),
                    # "bf16_loss": bf16_loss.item(),
                    "fp8_loss": fp8_loss.item(),
                    "fp8_loss_with_scaler": fp8_loss_with_scaler.item(),
                    "fp8_scaling_value": fp8_scaler.scaling_value.item(),
                    "scaled_fp8_loss_with_scaler": scaled_fp8_loss_with_scaler.item(),
                    # "msamp_o2_loss": msamp_loss.item(),
                    # "msamp_o2_loss_with_scaler": msamp_loss_with_scaler.item(),
                    # "scaled_msamp_o2_loss_with_scaler": scaled_msamp_loss_with_scaler.item(),
                    # "deepspeed_scaling_value": deepspeed_linear.optimizer.loss_scaler.loss_scale,
                    # "deepspeed_loss": deepspeed_loss.item(),
                    "l1_norm_diff_fp8_relative_to_fp32": l1_norm_diff(fp8_loss, fp32_loss).item(),
                    "l1_norm_diff_fp8_with_loss_scaler_relative_to_fp32": l1_norm_diff(
                        fp8_loss_with_scaler, fp32_loss
                    ).item(),
                    # "l1_norm_diff_msamp_with_loss_scaler_relative_to_fp32": l1_norm_diff(
                    #     msamp_loss_with_scaler, fp32_loss
                    # ).item(),
                    # "l1_norm_diff_fp8_with_loss_scaler_relative_to_bf16": l1_norm_diff(fp8_loss_with_scaler, bf16_loss).item(),
                    # "l1_norm_diff_msamp_with_loss_scaler_relative_to_bf16": l1_norm_diff(msamp_loss_with_scaler, bf16_loss).item(),
                    **convert_logs_to_flat_logs(fp32_logs, prefix="fp32"),
                    **convert_logs_to_flat_logs(fp8_logs, prefix="fp8"),
                    **convert_logs_to_flat_logs(fp8_with_scaler_logs, prefix="fp8_with_scaler"),
                    **fp32_optim_logs,
                    **fp8_optim_logs,
                    **fp8_with_scaler_optim_logs,
                    "step": step,
                }
            )

            for list_handles in [fp32_handles, fp8_handles, fp8_with_scaler_handles]:
                for handle in list_handles:
                    handle[0].remove()

        # NOTE: 3e-4 is from msamp
        # torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0.1, atol=3e-4)
        # torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
