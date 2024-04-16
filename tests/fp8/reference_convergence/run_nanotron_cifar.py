from copy import deepcopy
from dataclasses import asdict

import msamp
import torch
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.loss_scaler import LossScaler
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.utils import convert_to_fp8_module
from torch import nn
from torch.optim import Adam
from nanotron.fp8.utils import _log, convert_logs_to_flat_logs
import torch.nn.functional as F
import deepspeed
import argparse

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def l1_norm_diff(loss, ref_loss):
    return (loss - ref_loss).abs().mean()


def get_cifar_dataloader(batch_size):
    from torchvision import datasets, transforms
    import torch
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root="./", train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    testset = datasets.CIFAR10(root="./", train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return classes, train_loader, test_loader


class Net(nn.Module):
    """Take this MLP from greg yang's mup repo"""
    def __init__(self, width, num_classes=10):
        super(Net, self).__init__()
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, 16, bias=False)

    def forward(self, x):
        x = self.fc_3(F.relu(self.fc_2(F.relu(self.fc_1(x)))))
        return x[:, :10]
    
    
def add_argument():
    """Add arguments."""
    parser = argparse.ArgumentParser(description='CIFAR')

    # data
    # cuda
    parser.add_argument(
        '--with_cuda', default=False, action='store_true', help="use CPU in case there\'s no GPU support"
    )
    parser.add_argument('--use_ema', default=False, action='store_true', help='whether use exponential moving average')

    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval', type=int, default=200, help='output logging information at a given interval')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args
    

if __name__ == "__main__":
    BATCH_SIZE = 64
    INPUT_DIM = 64
    HIDDEN_SIZE = 64
    N_STEPS = 1000
    # LR = 6e-4
    LR = 1e-3
    N_LAYERS = 16
    WITH_BIAS = True
    MODEL_NAME = "gpt2"
    DATA_NAME = "CohereForAI/aya_dataset"
    
    torch.cuda.empty_cache()

    fp32_linear = Net(128).to("cuda")

    # bf16_linear = deepcopy(fp32_linear)
    fp8_linear = deepcopy(fp32_linear)
    fp8_linear_with_scaler = deepcopy(fp32_linear)
    msamp_linear = deepcopy(fp32_linear)
    msamp_linear_with_scaler = deepcopy(fp32_linear)
    deepspeed_linear = deepcopy(fp32_linear)

    fp32_optim = Adam(fp32_linear.parameters(), lr=LR)

    # bf16_linear = bf16_linear.to(dtype=torch.bfloat16)
    # bf16_optim = Adam(bf16_linear.parameters(), lr=LR)

    msamp_optim = Adam(msamp_linear.parameters(), lr=LR)
    msamp_linear, msamp_optim = msamp.initialize(msamp_linear, msamp_optim, opt_level="O2")

    msamp_optim_with_scaler = Adam(msamp_linear_with_scaler.parameters(), lr=LR)
    msamp_linear_with_scaler, msamp_optim_with_scaler = msamp.initialize(
        msamp_linear_with_scaler, msamp_optim_with_scaler, opt_level="O2"
    )

    fp8_linear = convert_to_fp8_module(fp8_linear, accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=LR)

    fp8_linear_with_scaler = convert_to_fp8_module(fp8_linear_with_scaler, accum_qtype=DTypes.KFLOAT16)
    fp8_optim_with_scaler = FP8Adam(fp8_linear_with_scaler.parameters(), lr=LR)

    msamp_scaler = torch.cuda.amp.GradScaler()
    fp8_scaler = LossScaler()

    args = add_argument()
    deepspeed.init_distributed()
    deepspeed_linear, deepspeed_optim, _, _  = deepspeed.initialize(args=args, model=deepspeed_linear)

    loss_func = nn.CrossEntropyLoss()
    _, train_dataloader, test_dataloader = get_cifar_dataloader(BATCH_SIZE)

    fp32_losses = []
    fp8_with_loss_scaler_losses = []
    msamp_with_loss_scaler_losses = []
    
    wandb.init(
        project="fp8_for_nanotron",
        name=f"{get_time_name()}.n_layers_{N_LAYERS}_and_input_dim_{INPUT_DIM}_and_hidden_size_{HIDDEN_SIZE}_and_lr_{LR}_and_bias_{WITH_BIAS}_and_batch_size_{BATCH_SIZE}",
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
            "num_params": sum(p.numel() for p in fp32_linear.parameters()),
            "fp8_recipe": asdict(FP8LM_RECIPE),
        },
    )

    for step in range(N_STEPS):
        for step, batch in enumerate(train_dataloader):
            print(f"step: {step} /n /n")
            batch = [x.to("cuda") for x in batch]
            inputs, targets = batch
            inputs = inputs.view(inputs.size(0), -1)

            fp32_logs = _log(fp32_linear)
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
            
            if step == 2:
                assert 1 == 1
            
            fp8_optim.zero_grad()
            fp8_logs = _log(fp8_linear)
            fp8_output = fp8_linear(inputs)
            fp8_loss = loss_func(fp8_output, targets)
            fp8_loss.backward()
            fp8_optim.step()
            
            # msamp_logs = _log(msamp_linear)
            msamp_optim.zero_grad()
            msamp_output = msamp_linear(inputs)
            msamp_loss = loss_func(msamp_output, targets)
            msamp_loss.backward()
            msamp_optim.all_reduce_grads(msamp_linear)
            msamp_optim.step()

            # msamp_with_loss_scaler_logs = _log(msamp_linear_with_scaler)
            msamp_optim_with_scaler.zero_grad()
            msamp_output_with_scaler = msamp_linear_with_scaler(inputs)
            msamp_loss_with_scaler = loss_func(msamp_output_with_scaler, targets)
            scaled_msamp_loss_with_scaler = msamp_scaler.scale(msamp_loss_with_scaler)
            scaled_msamp_loss_with_scaler.backward()
            msamp_scaler.step(msamp_optim_with_scaler)
            msamp_scaler.update()
                    
            deepspeed_output = deepspeed_linear(inputs.half())
            deepspeed_loss = loss_func(deepspeed_output, targets)
            deepspeed_linear.backward(deepspeed_loss)
            deepspeed_linear.step()
            
            fp8_with_scaler_logs = _log(fp8_linear_with_scaler)
            fp8_output_with_scaler = fp8_linear_with_scaler(inputs)
            fp8_loss_with_scaler = loss_func(fp8_output_with_scaler, targets)
            fp8_scaler.scaling_value = torch.tensor(deepspeed_linear.optimizer.loss_scaler.loss_scale, device="cuda")
            scaled_fp8_loss_with_scaler = fp8_scaler.scale(fp8_loss_with_scaler)
            fp8_optim_with_scaler.zero_grad()
            scaled_fp8_loss_with_scaler.backward()
            fp8_scaler.step(fp8_optim_with_scaler)
            fp8_scaler.update()

            fp32_losses.append(fp32_loss.item())
            # fp8_with_loss_scaler_losses.append(fp8_loss_with_scaler.item())
            msamp_with_loss_scaler_losses.append(msamp_loss_with_scaler.item())

            # l1_norm_diff_fp8_with_loss_scaler_relative_to_fp32 = l1_norm_diff(fp8_loss_with_scaler, fp32_loss)
            # l1_norm_diff_msamp_with_loss_scaler_relative_to_fp32 = l1_norm_diff(msamp_loss_with_scaler, fp32_loss)

            # std_fp8_with_loss_scaler_relative_to_fp32 = (torch.tensor(fp8_with_loss_scaler_losses) - torch.tensor(fp32_losses)).std()
            # std_msamp_with_loss_scaler_relative_to_fp32 = (torch.tensor(msamp_with_loss_scaler_losses) - torch.tensor(fp32_losses)).std()

            wandb.log(
                {
                    "fp32_loss": fp32_loss.item(),
                    # "bf16_loss": bf16_loss.item(),
                    "fp8_loss": fp8_loss.item(),
                    "fp8_loss_with_scaler": fp8_loss_with_scaler,
                    "fp8_scaling_value": fp8_scaler.scaling_value,
                    "scaled_fp8_loss_with_scaler": scaled_fp8_loss_with_scaler.item(),
                    
                    "msamp_o2_loss": msamp_loss.item(),
                    "msamp_o2_loss_with_scaler": msamp_loss_with_scaler.item(),
                    "scaled_msamp_o2_loss_with_scaler": scaled_msamp_loss_with_scaler.item(),
                    
                    "deepspeed_scaling_value": deepspeed_linear.optimizer.loss_scaler.loss_scale,
                    "deepspeed_loss": deepspeed_loss.item(),
                    
                    "l1_norm_diff_fp8_relative_to_fp32": l1_norm_diff(
                        fp8_loss, fp32_loss
                    ).item(),
                    # "l1_norm_diff_fp8_with_loss_scaler_relative_to_fp32": l1_norm_diff(
                    #     fp8_loss_with_scaler, fp32_loss
                    # ).item(),
                    "l1_norm_diff_msamp_with_loss_scaler_relative_to_fp32": l1_norm_diff(
                        msamp_loss_with_scaler, fp32_loss
                    ).item(),
                    # "l1_norm_diff_fp8_with_loss_scaler_relative_to_bf16": l1_norm_diff(fp8_loss_with_scaler, bf16_loss).item(),
                    # "l1_norm_diff_msamp_with_loss_scaler_relative_to_bf16": l1_norm_diff(msamp_loss_with_scaler, bf16_loss).item(),
                    **convert_logs_to_flat_logs(fp32_logs, prefix="fp32"),
                    **convert_logs_to_flat_logs(fp8_logs, prefix="fp8"),
                    **convert_logs_to_flat_logs(fp8_with_scaler_logs, prefix="fp8_with_scaler"),
                }
            )

        # NOTE: 3e-4 is from msamp
        # torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0.1, atol=3e-4)
        # torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
