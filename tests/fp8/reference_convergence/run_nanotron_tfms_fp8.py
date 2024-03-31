from copy import deepcopy

import msamp
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.loss_scaler import LossScaler
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.utils import convert_to_fp8_module
from torch import nn
from torch.optim import Adam
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def l1_norm_diff(loss, ref_loss):
    return (loss - ref_loss).abs().mean()

if __name__ == "__main__":
    BATCH_SIZE = 16
    HIDDEN_SIZE = 16
    N_STEPS = 1000
    LR = 1e-3
    N_LAYERS = 16
    WITH_BIAS = True
    SEED = 42
    MODEL_NAME = "gpt2"
    # NOTE: CohereForAI/aya_dataset: 200k examples
    # NOTE: stas/c4-en-10k
    DATA_NAME = "CohereForAI/aya_dataset"

    # fp32_linear = nn.Sequential(
    #     *[
    #         layer
    #         for _ in range(N_LAYERS)
    #         for layer in (nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=WITH_BIAS).to("cuda"), nn.ReLU())
    #     ]
    # )
    # fp32_linear = fp32_linear[:-1]
    
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(DATA_NAME)
    dataset = dataset.map(
        lambda x: tokenizer(x["inputs"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloaders = DataLoader(dataset["train"], batch_size=32, shuffle=True)

    fp32_linear = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")

    bf16_linear = deepcopy(fp32_linear)
    fp8_linear = deepcopy(fp32_linear)
    fp8_linear_with_scaler = deepcopy(fp32_linear)
    msamp_linear = deepcopy(fp32_linear)
    msamp_linear_with_scaler = deepcopy(fp32_linear)

    fp32_optim = Adam(fp32_linear.parameters(), lr=LR)
    
    bf16_linear = bf16_linear.to(dtype=torch.bfloat16)
    bf16_optim = Adam(bf16_linear.parameters(), lr=LR)

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

    wandb.init(
        project="fp8_for_nanotron",
        name=f"{get_time_name()}.convergence_fp8_n_layers_{N_LAYERS}_and_hidden_size_{HIDDEN_SIZE}_and_lr_{LR}_and_bias_{WITH_BIAS}_and_batch_size_{BATCH_SIZE}",
        config={
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "n_steps": N_STEPS,
            "lr": LR,
            "n_layers": N_LAYERS,
            "with_bias": WITH_BIAS,
            # "act_func": fp32_linear[1].__class__.__name__ if N_LAYERS > 1 else "None",
            "optim": fp32_optim.__class__.__name__,
            "optim_params": fp32_optim.defaults,
            "num_params": sum(p.numel() for p in fp32_linear.parameters()),
        },
    )

    msamp_scaler = torch.cuda.amp.GradScaler()
    fp8_scaler = LossScaler()

    # loss_func = nn.CrossEntropyLoss()
    def loss_func(outputs, input_ids):
        func = nn.CrossEntropyLoss()
        return func(outputs.logits.view(-1, outputs.logits.shape[-1]), input_ids.view(-1))

    # batch_inputs = []
    # batch_targets = []
    # inputs = torch.randn(BATCH_SIZE, HIDDEN_SIZE).to("cuda")
    # targets = torch.randint(0, HIDDEN_SIZE, (BATCH_SIZE,)).to("cuda")
    # for _ in range(N_STEPS):
    #     batch_inputs.append(inputs.clone())
    #     batch_targets.append(targets.clone())

    fp32_losses = []
    fp8_with_loss_scaler_losses = []
    msamp_with_loss_scaler_losses = []

    for step in range(N_STEPS):
        for step, batch in enumerate(dataloaders):
            print(f"step: {step} /n /n")

            input_ids = batch["input_ids"].to("cuda")
            # targets = batch["input_ids"][:, 1:].to("cuda")
            # inputs = batch_inputs[step]
            # targets = batch_targets[step]
            # inputs = torch.randn(BATCH_SIZE, HIDDEN_SIZE).to("cuda")
            # targets = torch.randint(0, HIDDEN_SIZE, (BATCH_SIZE,)).to("cuda")

            fp32_optim.zero_grad()
            ref_output = fp32_linear(input_ids)
            # fp32_loss = loss_func(ref_output, targets)
            # fp32_loss = loss_func(
            #     ref_output.logits.view(-1, ref_output.logits.shape[-1]), input_ids.view(-1)
            # )
            fp32_loss = loss_func(ref_output, input_ids)
            fp32_loss.backward()
            fp32_optim.step()
            
            # bf16_optim.zero_grad()
            # bf16_output = bf16_linear(input_ids.to(dtype=torch.bfloat16))
            # # bf16_loss = loss_func(bf16_output, targets)
            # bf16_loss = loss_func(ref_output, input_ids)
            # bf16_loss.backward()
            # bf16_optim.step()

            # fp8_output = fp8_linear(inputs)
            # fp8_loss = loss_func(fp8_output, targets)
            # fp8_loss.backward()
            # fp8_optim.step()
            # fp8_optim.zero_grad()

            # fp8_optim_with_scaler.zero_grad()
            # fp8_output_with_scaler = fp8_linear_with_scaler(input_ids)
            # # fp8_loss_with_scaler = loss_func(fp8_output_with_scaler, targets)
            # fp8_loss_with_scaler = loss_func(fp8_output_with_scaler, input_ids)
            # fp8_scaler.scale(fp8_loss_with_scaler)
            # fp8_scaler.step(fp8_optim_with_scaler)
            # fp8_scaler.update()

            msamp_optim.zero_grad()
            msamp_output = msamp_linear(input_ids)
            # msamp_loss = loss_func(msamp_output, targets)
            msamp_loss = loss_func(msamp_output, input_ids)
            msamp_loss.backward()
            msamp_optim.all_reduce_grads(msamp_linear)
            msamp_optim.step()

            msamp_optim_with_scaler.zero_grad()
            msamp_output_with_scaler = msamp_linear_with_scaler(input_ids)
            # msamp_loss_with_scaler = loss_func(msamp_output_with_scaler, targets)
            msamp_loss_with_scaler = loss_func(msamp_output_with_scaler, input_ids)
            msamp_scaler.scale(msamp_loss_with_scaler).backward()
            msamp_scaler.step(msamp_optim_with_scaler)
            msamp_scaler.update()

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
                    # "fp8_loss": fp8_loss.item(),
                    # "fp8_loss_with_scaler": fp8_loss_with_scaler.item(),
                    "msamp_o2_loss": msamp_loss.item(),
                    "msamp_o2_loss_with_scaler": msamp_loss_with_scaler.item(),
                    # "l1_norm_diff_fp8_with_loss_scaler_relative_to_fp32": l1_norm_diff(fp8_loss_with_scaler, fp32_loss).item(),
                    "l1_norm_diff_msamp_with_loss_scaler_relative_to_fp32": l1_norm_diff(msamp_loss_with_scaler, fp32_loss).item(),
                    
                    # "l1_norm_diff_fp8_with_loss_scaler_relative_to_bf16": l1_norm_diff(fp8_loss_with_scaler, bf16_loss).item(),
                    # "l1_norm_diff_msamp_with_loss_scaler_relative_to_bf16": l1_norm_diff(msamp_loss_with_scaler, bf16_loss).item(),
                }
            )

    # NOTE: 3e-4 is from msamp
    # torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0.1, atol=3e-4)
    # torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
