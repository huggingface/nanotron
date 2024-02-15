from __future__ import print_function

from copy import deepcopy

import msamp
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def main():
    """The main function."""

    SEED = 42
    LR = 1e-3
    N_EPOCHS = 1
    MODEL_NAME = "gpt2"
    # NOTE: CohereForAI/aya_dataset: 200k examples
    # NOTE: stas/c4-en-10k
    DATA_NAME = "CohereForAI/aya_dataset"
    OPT_LEVEL = "O2"

    wandb.init(
        project="nanotron",
        name=f"{get_time_name()}.test_msamp_gpt2_convergence",
        config={
            "seed": SEED,
            "lr": LR,
            "n_epochs": N_EPOCHS,
            "model_name": MODEL_NAME,
            "data_name": DATA_NAME,
            "opt_level": OPT_LEVEL,
            "optim": "Adam",
        },
    )

    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(DATA_NAME)
    dataset = dataset.map(
        lambda x: tokenizer(x["inputs"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloaders = DataLoader(dataset["train"], batch_size=32, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    fp8_model = deepcopy(model)
    bf16_model = deepcopy(model)
    fp32_model = deepcopy(model)

    bf16_model = bf16_model.to("cuda")
    fp8_model = fp8_model.to("cuda")
    fp32_model = fp32_model.to("cuda")

    # NOTE: check all model parameters are the same
    for (k1, v1), (k2, v2), (k3, v3) in zip(
        bf16_model.named_parameters(), fp8_model.named_parameters(), fp32_model.named_parameters()
    ):
        assert k1 == k2 == k3
        assert torch.equal(v1, v2)
        assert torch.equal(v1, v3)

    # NOTE: this is the reference format in the paper
    bf16_model = bf16_model.to(dtype=torch.bfloat16)
    bf16_optim = optim.Adam(bf16_model.parameters(), lr=LR)

    fp8_optim = optim.Adam(fp8_model.parameters(), lr=LR)
    fp8_model, fp8_optim = msamp.initialize(fp8_model, fp8_optim, opt_level=OPT_LEVEL)

    fp32_optim = optim.Adam(fp32_model.parameters(), lr=LR)

    fp8_model.train()
    bf16_model.train()
    fp32_model.train()

    for _ in range(N_EPOCHS):
        for inputs in dataloaders:
            # NOTE: move inputs to cuda
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            fp8_optim.zero_grad()
            bf16_optim.zero_grad()
            fp32_optim.zero_grad()

            # with torch.cuda.amp.autocast():
            fp8_output = fp8_model(**inputs)
            fp8_loss = nn.CrossEntropyLoss()(
                fp8_output.logits.view(-1, fp8_output.logits.shape[-1]), inputs["input_ids"].view(-1)
            )
            fp8_loss.backward()
            fp8_optim.step()

            bf16_output = bf16_model(**inputs)
            bf16_loss = nn.CrossEntropyLoss()(
                bf16_output.logits.view(-1, bf16_output.logits.shape[-1]), inputs["input_ids"].view(-1)
            )
            bf16_loss.backward()
            bf16_optim.step()

            fp32_output = fp32_model(**inputs)
            fp32_loss = nn.CrossEntropyLoss()(
                fp32_output.logits.view(-1, fp32_output.logits.shape[-1]), inputs["input_ids"].view(-1)
            )
            fp32_loss.backward()
            fp32_optim.step()

            wandb.log({"fp8_loss": fp8_loss.item(), "bf16_loss": bf16_loss.item(), "fp32_loss": fp32_loss.item()})


if __name__ == "__main__":
    main()
