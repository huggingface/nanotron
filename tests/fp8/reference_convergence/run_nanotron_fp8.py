from copy import deepcopy

import msamp
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.loss_scaler import LossScaler
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.utils import convert_to_fp8_module
from torch import nn
from torch.optim import Adam

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


if __name__ == "__main__":
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    N_STEPS = 1000
    LR = 1e-3
    N_LAYERS = 1
    WITH_BIAS = True

    ref_linear = nn.Sequential(
        *[
            layer
            for _ in range(N_LAYERS)
            for layer in (nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=WITH_BIAS).to("cuda"), nn.ReLU())
        ]
    )
    # ref_linear = ref_linear[:-1] if N_LAYERS > 1 else ref_linear
    ref_linear = ref_linear[:-1]

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

    fp8_linear = convert_to_fp8_module(fp8_linear, accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=LR)

    fp8_linear_with_scaler = convert_to_fp8_module(fp8_linear_with_scaler, accum_qtype=DTypes.KFLOAT16)
    fp8_optim_with_scaler = FP8Adam(fp8_linear_with_scaler.parameters(), lr=LR)

    wandb.init(
        project="fp8_for_nanotron",
        name=f"{get_time_name()}.convergence_fp8_n_layers_{N_LAYERS}_and_hidden_size_{HIDDEN_SIZE}_and_lr_{LR}_with_bias_{WITH_BIAS}",
        config={
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "n_steps": N_STEPS,
            "lr": LR,
            "n_layers": N_LAYERS,
            "with_bias": WITH_BIAS,
            "act_func": ref_linear[1].__class__.__name__ if N_LAYERS > 1 else "None",
            "optim": ref_optim.__class__.__name__,
            "optim_params": ref_optim.defaults,
        },
    )

    msamp_scaler = torch.cuda.amp.GradScaler()
    fp8_scaler = LossScaler()

    loss_func = nn.CrossEntropyLoss()

    # batch_inputs = []
    # batch_targets = []
    # inputs = torch.randn(BATCH_SIZE, HIDDEN_SIZE).to("cuda")
    # targets = torch.randint(0, HIDDEN_SIZE, (BATCH_SIZE,)).to("cuda")
    # for _ in range(N_STEPS):
    #     batch_inputs.append(inputs.clone())
    #     batch_targets.append(targets.clone())

    for step in range(N_STEPS):
        print(f"step: {step} /n /n")
        # inputs = batch_inputs[step]
        # targets = batch_targets[step]
        inputs = torch.randn(BATCH_SIZE, HIDDEN_SIZE).to("cuda")
        targets = torch.randint(0, HIDDEN_SIZE, (BATCH_SIZE,)).to("cuda")

        ref_optim.zero_grad()
        ref_output = ref_linear(inputs)
        loss = loss_func(ref_output, targets)
        loss.backward()
        ref_optim.step()

        # fp8_output = fp8_linear(inputs)
        # fp8_loss = loss_func(fp8_output, targets)
        # fp8_loss.backward()
        # fp8_optim.step()
        # fp8_optim.zero_grad()

        fp8_optim_with_scaler.zero_grad()
        fp8_output_with_scaler = fp8_linear_with_scaler(inputs)
        fp8_loss_with_scaler = loss_func(fp8_output_with_scaler, targets)
        fp8_scaler.scale(fp8_loss_with_scaler)
        fp8_scaler.step(fp8_optim_with_scaler)
        fp8_scaler.update()

        msamp_optim.zero_grad()
        msamp_output = msamp_linear(inputs)
        msamp_loss = loss_func(msamp_output, targets)
        msamp_loss.backward()
        msamp_optim.all_reduce_grads(msamp_linear)
        msamp_optim.step()

        msamp_optim_with_scaler.zero_grad()
        msamp_output_with_scaler = msamp_linear_with_scaler(inputs)
        msamp_loss_with_scaler = loss_func(msamp_output_with_scaler, targets)
        msamp_scaler.scale(msamp_loss_with_scaler).backward()
        msamp_scaler.step(msamp_optim_with_scaler)
        msamp_scaler.update()

        wandb.log(
            {
                "fp32_loss": loss.item(),
                # "fp8_loss": fp8_loss.item(),
                "fp8_loss_with_scaler": fp8_loss_with_scaler.item(),
                "msamp_o2_loss": msamp_loss.item(),
                "msamp_o2_loss_with_scaler": msamp_loss_with_scaler.item(),
            }
        )

    # NOTE: 3e-4 is from msamp
    # torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0.1, atol=3e-4)
    # torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
