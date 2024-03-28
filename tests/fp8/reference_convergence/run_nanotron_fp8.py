from copy import deepcopy

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
    HIDDEN_SIZE = 16
    N_STEPS = 100
    LR = 1e-3

    ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    fp8_linear = deepcopy(ref_linear)
    # msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)

    ref_optim = Adam(ref_linear.parameters(), lr=LR)
    # msamp_optim = LBAdam(msamp_linear.parameters(), lr=LR)

    fp8_linear = convert_linear_to_fp8(fp8_linear, accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_linear.parameters(), lr=LR)

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=False)

    wandb.init(
        project="nanotron",
        name=f"{get_time_name()}.sanity_fp8",
    )

    # scaling_factors = []
    for _ in range(N_STEPS):
        ref_output = ref_linear(input).sum()
        fp8_output = fp8_linear(input).sum()
        # msamp_output = msamp_linear(input)

        ref_output.sum().backward()
        # msamp_output.sum().backward()
        fp8_output.sum().backward()

        ref_optim.step()
        # ref_optim.zero_grad()
        fp8_optim.step()

        # msamp_optim.all_reduce_grads(msamp_linear)
        # msamp_optim.step()
        # msamp_optim.zero_grad()

        # scaling_factors.append(deepcopy(msamp_linear.weight.meta.scale.item()))

        wandb.log({"ref_output": ref_output.item(), "fp8_output": fp8_output.item()})

    # NOTE: 3e-4 is from msamp
    # torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0.1, atol=3e-4)
    # torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
