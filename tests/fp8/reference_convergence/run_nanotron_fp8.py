from copy import deepcopy

import msamp
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.utils import convert_linear_to_fp8, convert_to_fp8_module
from nanotron.fp8.loss_scaler import LossScaler
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
    N_STEPS = 1000
    LR = 1e-3
    N_LAYERS = 1

    # ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    ref_linear = nn.Sequential(*[nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE).to('cuda') for _ in range(N_LAYERS)])
    fp8_linear = deepcopy(ref_linear)
    fp8_linear_with_scaler = deepcopy(ref_linear)
    msamp_linear = deepcopy(ref_linear)
    
    ref_optim = Adam(ref_linear.parameters(), lr=LR)

    msamp_optim = Adam(msamp_linear.parameters(), lr=LR)
    msamp_linear, msamp_optim = msamp.initialize(msamp_linear, msamp_optim, opt_level="O2")

    msamp_linear_with_scaler = deepcopy(ref_linear)
    msamp_optim_with_scaler = Adam(msamp_linear_with_scaler.parameters(), lr=LR)
    msamp_linear_with_scaler, msamp_optim_with_scaler = msamp.initialize(
        msamp_linear_with_scaler, msamp_optim_with_scaler, opt_level="O2"
    )

    # msamp_optim = LBAdam(msamp_linear.parameters(), lr=LR)

    fp8_linear = convert_to_fp8_module(fp8_linear, accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=LR)
    
    fp8_linear_with_scaler = convert_to_fp8_module(fp8_linear_with_scaler, accum_qtype=DTypes.KFLOAT16)
    fp8_optim_with_scaler = FP8Adam(fp8_linear_with_scaler.parameters(), lr=LR)

    # inputs = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=False)

    wandb.init(
        project="fp8_for_nanotron",
        name=f"{get_time_name()}.sanity_fp8",
        config={
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "n_steps": N_STEPS,
            "lr": LR,
            "n_layers": 1
        }
    )

    msamp_scaler = torch.cuda.amp.GradScaler()
    fp8_scaler = LossScaler()

    # def loss_func(x):
    #     return F.sigmoid(x) * 5 + 0.5
    loss_func = nn.CrossEntropyLoss()

    # batch_inputs = []
    # batch_targets = []
    # inputs = torch.randn(BATCH_SIZE, HIDDEN_SIZE).to("cuda")
    # targets = torch.randint(0, HIDDEN_SIZE, (BATCH_SIZE,)).to("cuda")
    # for _ in range(N_STEPS):
    #     batch_inputs.append(inputs.clone())
    #     batch_targets.append(targets.clone())


    for step in range(N_STEPS):
        # inputs = batch_inputs[step]
        # targets = batch_targets[step]
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
        
        fp8_output_with_scaler = fp8_linear_with_scaler(inputs)
        fp8_loss_with_scaler = loss_func(fp8_output_with_scaler, targets)
        fp8_scaler.scale(fp8_loss_with_scaler)
        fp8_scaler.step(fp8_optim_with_scaler)
        fp8_scaler.update()

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
                "fp8_loss_with_scaler": fp8_loss_with_scaler.item(),
                "msamp_o2_loss": msamp_loss.item(),
                "msamp_o2_loss_with_scaler": msamp_loss_with_scaler.item(),
            }
        )

    # NOTE: 3e-4 is from msamp
    # torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0.1, atol=3e-4)
    # torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
