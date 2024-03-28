from copy import deepcopy

import msamp
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.utils import convert_linear_to_fp8
from torch import nn
from torch.optim import Adam

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


if __name__ == "__main__":
    BATCH_SIZE = 16
    HIDDEN_SIZE = 16
    N_STEPS = 20000
    LR = 1e-3

    ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    fp8_linear = deepcopy(ref_linear)

    msamp_linear = deepcopy(ref_linear)
    msamp_optim = Adam(msamp_linear.parameters(), lr=LR)
    msamp_linear, msamp_optim = msamp.initialize(msamp_linear, msamp_optim, opt_level="O2")

    msamp_linear_with_scaler = deepcopy(ref_linear)
    msamp_optim_with_scaler = Adam(msamp_linear_with_scaler.parameters(), lr=LR)
    msamp_linear_with_scaler, msamp_optim_with_scaler = msamp.initialize(
        msamp_linear_with_scaler, msamp_optim_with_scaler, opt_level="O2"
    )

    ref_optim = Adam(ref_linear.parameters(), lr=LR)
    # msamp_optim = LBAdam(msamp_linear.parameters(), lr=LR)

    fp8_linear = convert_linear_to_fp8(fp8_linear, accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=LR)

    # inputs = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=False)

    wandb.init(
        project="fp8_for_nanotron",
        name=f"{get_time_name()}.sanity_fp8",
    )

    msamp_scaler = torch.cuda.amp.GradScaler()

    # def loss_func(x):
    #     return F.sigmoid(x) * 5 + 0.5
    loss_func = nn.CrossEntropyLoss()

    for _ in range(N_STEPS):
        inputs = torch.randn(BATCH_SIZE, HIDDEN_SIZE).to("cuda")
        targets = torch.randint(0, HIDDEN_SIZE, (BATCH_SIZE,)).to("cuda")

        ref_output = ref_linear(inputs)
        loss = loss_func(ref_output, targets)
        loss.backward()
        ref_optim.step()
        ref_optim.zero_grad()

        fp8_output = fp8_linear(inputs)
        fp8_loss = loss_func(fp8_output, targets)
        fp8_loss.backward()
        fp8_optim.step()
        fp8_optim.zero_grad()

        msamp_output = msamp_linear(inputs)
        msamp_loss = loss_func(msamp_output, targets)
        msamp_loss.backward()
        msamp_optim.all_reduce_grads(msamp_linear)
        msamp_optim.step()
        msamp_optim.zero_grad()

        msamp_output_with_scaler = msamp_linear_with_scaler(inputs)
        msamp_loss_with_scaler = loss_func(msamp_output_with_scaler, targets)
        msamp_scaler.scale(msamp_loss_with_scaler).backward()
        msamp_scaler.step(msamp_optim_with_scaler)
        msamp_scaler.update()

        wandb.log(
            {
                "fp32_loss": loss.item(),
                "fp8_loss": fp8_loss.item(),
                "msamp_o2_loss": msamp_loss.item(),
                "msamp_o2_loss_with_scaler": msamp_loss_with_scaler.item(),
            }
        )

    # NOTE: 3e-4 is from msamp
    # torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0.1, atol=3e-4)
    # torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
