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
from transformers import BloomConfig, BloomModel, BloomTokenizerFast, BloomForCausalLM

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def l1_norm_diff(loss, ref_loss):
    return (loss - ref_loss).abs().mean()

if __name__ == "__main__":
    BATCH_SIZE = 16
    # HIDDEN_SIZE = 16
    N_EPOCHS = 1
    LR = 1e-3
    # N_LAYERS = 16
    # WITH_BIAS = True
    SEED = 42
    # MODEL_NAME = "gpt2"
    MODEL_NAME = "bigscience/bloom"
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
    torch.cuda.empty_cache()
    
    # config = BloomConfig(
    #     hidden_size=64,
    #     n_layer=5,
    #     slow_but_exact=True
    # )
    config = BloomConfig()

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # fp32_linear = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")
    fp32_linear = BloomForCausalLM(config)
    
    dataset = load_dataset(DATA_NAME)
    dataset = dataset.map(
        lambda x: tokenizer(x["inputs"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloaders = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)

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
    
    fp32_linear = fp32_linear.to("cuda")
    bf16_linear = bf16_linear.to("cuda")
    fp8_linear = fp8_linear.to("cuda")
    fp8_linear_with_scaler = fp8_linear_with_scaler.to("cuda")
    msamp_linear = msamp_linear.to("cuda")
    msamp_linear_with_scaler = msamp_linear_with_scaler.to("cuda")

    wandb.init(
        project="fp8_for_nanotron",
        name=f"{get_time_name()}.convergence_gpt2",
        config={
            "epochs": 1,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "optim": fp32_optim.__class__.__name__,
            "optim_params": fp32_optim.defaults,
            "num_params": sum(p.numel() for p in fp32_linear.parameters()),
            "model_config": config.to_dict(),
        },
    )

    msamp_scaler = torch.cuda.amp.GradScaler()
    fp8_scaler = LossScaler()

    def loss_func(outputs, targets):
        func = nn.CrossEntropyLoss()
        # logits = outputs.logits.squeeze(dim=1)
        # targets = input_ids.squeeze(dim=1)
        logits = outputs.logits
        logits = logits[:, :-1, :].contiguous()
        return func(logits.view(-1, logits.shape[-1]), targets.view(-1))
    
    fp32_losses = []
    fp8_with_loss_scaler_losses = []
    msamp_with_loss_scaler_losses = []
    
    for epoch in range(N_EPOCHS):
        for step, batch in enumerate(dataloaders):
            print(f"step: {step} /n /n")

            batch = {k: v.squeeze(dim=1).to("cuda") for k, v in batch.items()}
            targets = batch["input_ids"][:, 1:].contiguous()

            fp32_optim.zero_grad()
            fp32_output = fp32_linear(**batch)
            fp32_loss = loss_func(fp32_output, targets)
            # fp32_loss = fp32_output.loss
            fp32_loss.backward()
            fp32_optim.step()
            
            # bf16_optim.zero_grad()
            # bf16_output = bf16_linear(**batch)
            # bf16_loss = loss_func(bf16_output, batch["input_ids"])
            # bf16_loss.backward()
            # bf16_optim.step()

            # fp8_output = fp8_linear(**batch)
            # fp8_loss = loss_func(fp8_output, batch["input_ids"])
            # fp8_loss.backward()
            # fp8_optim.step()
            # fp8_optim.zero_grad()

            # fp8_optim_with_scaler.zero_grad()
            # fp8_output_with_scaler = fp8_linear_with_scaler(**batch)
            # fp8_loss_with_scaler = loss_func(fp8_output_with_scaler, batch["input_ids"])
            # fp8_scaler.scale(fp8_loss_with_scaler)
            # fp8_scaler.step(fp8_optim_with_scaler)
            # fp8_scaler.update()

            msamp_optim.zero_grad()
            msamp_output = msamp_linear(**batch)
            msamp_loss = loss_func(msamp_output, targets)
            # msamp_loss = msamp_output.loss
            msamp_loss.backward()
            msamp_optim.all_reduce_grads(msamp_linear)
            msamp_optim.step()

            msamp_optim_with_scaler.zero_grad()
            msamp_output_with_scaler = msamp_linear_with_scaler(**batch)
            msamp_loss_with_scaler = loss_func(msamp_output_with_scaler, targets)
            # msamp_loss_with_scaler = msamp_output_with_scaler.loss
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
