from __future__ import print_function

import time
from copy import deepcopy

import msamp
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def cuda_sleep(seconds):
    # Warm-up CUDA.
    torch.empty(1, device="cuda")

    # From test/test_cuda.py in PyTorch.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(1000000)
    end.record()
    end.synchronize()
    cycles_per_ms = 1000000 / start.elapsed_time(end)

    torch.cuda._sleep(int(seconds * cycles_per_ms * 1000))
    return cuda_sleep


def main():
    """The main function."""

    SEED = 42
    LR = 1e-3
    MODEL_NAME = "gpt2"
    OPT_LEVEL = "O2"
    N_ITERATIONS = 100

    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    # NOTE: create an input that batch size is 64
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    batch_size = 64
    inputs = {key: value.repeat(batch_size, 1) for key, value in inputs.items()}

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    fp8_model = deepcopy(model)
    bf16_model = deepcopy(model)
    # fp32_model = deepcopy(model)

    bf16_model = bf16_model.to("cuda")
    fp8_model = fp8_model.to("cuda")
    # fp32_model = fp32_model.to("cuda")

    # NOTE: check all model parameters are the same
    for (k1, v1), (k2, v2) in zip(bf16_model.named_parameters(), fp8_model.named_parameters()):
        assert k1 == k2
        assert torch.equal(v1, v2)

    # NOTE: this is the reference format in the paper
    bf16_model = bf16_model.to(dtype=torch.bfloat16)
    bf16_optim = optim.Adam(bf16_model.parameters(), lr=LR)

    fp8_optim = optim.Adam(fp8_model.parameters(), lr=LR)
    fp8_model, fp8_optim = msamp.initialize(fp8_model, fp8_optim, opt_level=OPT_LEVEL)

    fp8_model.train()
    bf16_model.train()

    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    cuda_sleep(1)

    for trial in range(10):

        torch.cuda.synchronize()
        start_time_bf16 = time.time()
        for _ in range(N_ITERATIONS):
            bf16_output = bf16_model(**inputs)

            bf16_optim.zero_grad()
            bf16_loss = nn.CrossEntropyLoss()(
                bf16_output.logits.view(-1, bf16_output.logits.shape[-1]), inputs["input_ids"].view(-1)
            )
            bf16_loss.backward()
            bf16_optim.step()

        torch.cuda.synchronize()
        end_time_bf16 = time.time()

        torch.cuda.synchronize()
        start_time_fp8 = time.time()
        for _ in range(N_ITERATIONS):
            fp8_output = fp8_model(**inputs)

            fp8_optim.zero_grad()
            fp8_loss = nn.CrossEntropyLoss()(
                fp8_output.logits.view(-1, fp8_output.logits.shape[-1]), inputs["input_ids"].view(-1)
            )
            fp8_loss.backward()
            fp8_optim.step()

        torch.cuda.synchronize()
        end_time_fp8 = time.time()

        print("-------------------")
        print(f"[trial={trial}] bf16 time: {end_time_bf16 - start_time_bf16}")
        print(f"[trial={trial}] fp8 time: {end_time_fp8 - start_time_fp8}")
        print(f"[trial={trial}] speedup: {(end_time_bf16 - start_time_bf16) / (end_time_fp8 - start_time_fp8)}")


if __name__ == "__main__":
    main()
