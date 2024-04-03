from __future__ import print_function

from copy import deepcopy

import msamp
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.loss_scaler import LossScaler
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.utils import convert_to_fp8_module
from timm.models.layers import trunc_normal_
from torch.utils.data import DataLoader
from transformers import BloomConfig, BloomTokenizerFast

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def loss_func(outputs, targets):
    func = nn.CrossEntropyLoss()
    logits = outputs.logits
    logits = logits[:, :-1, :].contiguous()
    return func(logits.view(-1, logits.shape[-1]), targets.view(-1))


def l1_norm_diff(loss, ref_loss):
    return (loss - ref_loss).abs().mean()


from dataclasses import dataclass


@dataclass
class ModelOutput:
    logits: torch.Tensor


class ToyModel(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.net = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.vocab_size),
        # )
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

        # self.linear.weight.data.normal_(mean=0.0, std=0.02)
        # self.word_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        trunc_normal_(self.linear.weight.data, std=0.02)
        trunc_normal_(self.word_embeddings.weight.data, std=0.02)

    def forward(self, input_ids, attention_mask):
        x = self.word_embeddings(input_ids)
        # return self.net(x)
        return ModelOutput(logits=self.linear(x))


def main():
    """The main function."""

    SEED = 42
    # LR = 6e-4
    LR = 0.01
    SEQ_LEN = 128
    BATCH_SIZE = 16
    N_EPOCHS = 1

    # MODEL_NAME = "gpt2"
    MODEL_NAME = "bigscience/bloom"
    HIDDEN_SIZE = 16
    N_LAYERS = 1
    N_HEADS = 2

    # HIDDEN_SIZE = 16
    # N_LAYERS = 1
    # N_HEADS = 2

    # INITIALIZER_RANGE = math.sqrt(1 / HIDDEN_SIZE)
    INITIALIZER_RANGE = 0.02

    # NOTE: CohereForAI/aya_dataset: 200k examples
    # NOTE: stas/c4-en-10k
    DATA_NAME = "CohereForAI/aya_dataset"
    OPT_LEVEL = "O2"

    torch.manual_seed(SEED)
    torch.cuda.empty_cache()

    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(DATA_NAME)
    dataset = dataset.map(
        lambda x: tokenizer(
            x["inputs"], padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt"
        )
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloaders = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=False)

    config = BloomConfig(
        n_layer=N_LAYERS,
        hidden_size=HIDDEN_SIZE,
        n_head=N_HEADS,
        slow_but_exact=True,
        initializer_range=INITIALIZER_RANGE,
    )
    # model = BloomForCausalLM(config)
    model = ToyModel(config)

    fp32_model = deepcopy(model).to("cuda")
    bf16_model = deepcopy(model).to("cuda")
    fp8_model = deepcopy(model).to("cuda")
    msamp_model = deepcopy(model).to("cuda")

    # # NOTE: check all model parameters are the same
    # for (k1, v1), (k2, v2), (k3, v3) in zip(
    #     fp8_model.named_parameters(), msamp_model.named_parameters(), fp32_model.named_parameters()
    # ):
    #     assert k1 == k2 == k3
    #     assert torch.equal(v1, v2)
    #     assert torch.equal(v1, v3)

    fp32_optim = optim.Adam(fp32_model.parameters(), lr=LR)

    # NOTE: this is the reference format in the paper
    bf16_model = bf16_model.to(dtype=torch.bfloat16)
    optim.Adam(bf16_model.parameters(), lr=LR)

    fp8_model = convert_to_fp8_module(fp8_model, accum_qtype=DTypes.KFLOAT16)
    fp8_optim = FP8Adam(fp8_model.parameters(), lr=LR)

    msamp_optim = optim.Adam(msamp_model.parameters(), lr=LR)
    msamp_model, msamp_optim = msamp.initialize(msamp_model, msamp_optim, opt_level=OPT_LEVEL)

    msamp_model.train()
    bf16_model.train()
    fp32_model.train()

    for fp8_p, fp32_p in zip(fp8_model.parameters(), fp32_model.parameters()):
        assert fp8_p.shape == fp32_p.shape

    wandb.init(
        project="fp8_for_nanotron",
        name=f"{get_time_name()}.fp8_bloom_convergence_hidden_size_{HIDDEN_SIZE}_and_n_layers_{N_LAYERS}_and_n_heads_{N_HEADS}_and_lr_{LR}_and_init_range_{INITIALIZER_RANGE}",
        config={
            "seed": SEED,
            "lr": LR,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "model_name": MODEL_NAME,
            "data_name": DATA_NAME,
            "opt_level": OPT_LEVEL,
            "initializer_range": INITIALIZER_RANGE,
            "optim": "Adam",
            "num_params": sum(p.numel() for p in fp32_model.parameters()),
            "model_config": config.to_dict(),
        },
    )

    # fp8_scaler = torch.cuda.amp.GradScaler()
    fp8_scaler = LossScaler()
    msamp_scaler = torch.cuda.amp.GradScaler()

    for _ in range(N_EPOCHS):
        for step, batch in enumerate(dataloaders):
            print(f"step: {step} \n")
            # NOTE: move inputs to cuda
            batch = {k: v.squeeze(dim=1).to("cuda") for k, v in batch.items()}
            targets = batch["input_ids"][:, 1:].contiguous()

            fp32_optim.zero_grad()
            fp32_output = fp32_model(**batch)
            fp32_loss = loss_func(fp32_output, targets)
            fp32_loss.backward()
            fp32_optim.step()

            # bf16_optim.zero_grad()
            # bf16_output = bf16_model(**batch)
            # bf16_loss = loss_func(bf16_output, targets)
            # bf16_loss.backward()
            # bf16_optim.step()

            fp8_optim.zero_grad()
            fp8_output = fp8_model(**batch)
            fp8_loss = loss_func(fp8_output, targets)
            fp8_scaler.scale(fp8_loss).backward()
            fp8_scaler.step(fp8_optim)
            fp8_scaler.update()

            msamp_optim.zero_grad()
            msamp_output = msamp_model(**batch)
            msamp_loss = loss_func(msamp_output, targets)
            msamp_scaler.scale(msamp_loss).backward()
            msamp_scaler.step(msamp_optim)
            msamp_scaler.update()

            wandb.log(
                {
                    "fp32_loss": fp32_loss.item(),
                    # "bf16_loss": bf16_loss.item(),
                    "fp8_loss": fp8_loss.item(),
                    "msamp_loss": msamp_loss.item(),
                    "l1_norm_diff_fp8_with_loss_scaler_relative_to_fp32": l1_norm_diff(fp8_loss, fp32_loss).item(),
                    "l1_norm_diff_msamp_with_loss_scaler_relative_to_fp32": l1_norm_diff(msamp_loss, fp32_loss).item(),
                    # "l1_norm_diff_fp8_with_loss_scaler_relative_to_bf16": l1_norm_diff(fp8_loss, bf16_loss).item(),
                    # "l1_norm_diff_msamp_with_loss_scaler_relative_to_bf16": l1_norm_diff(msamp_loss, bf16_loss).item(),
                }
            )


if __name__ == "__main__":
    main()
