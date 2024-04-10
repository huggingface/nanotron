from copy import deepcopy
from dataclasses import asdict, dataclass

import msamp
import torch
from datasets import load_dataset
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.loss_scaler import LossScaler
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.utils import convert_to_fp8_module
from timm.models.layers import trunc_normal_
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from nanotron.fp8.utils import _log, convert_logs_to_flat_logs
import torch.nn.functional as F

import wandb


def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def l1_norm_diff(loss, ref_loss):
    return (loss - ref_loss).abs().mean()


def get_dataloader():
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader



class Net(nn.Module):
    """Define a Convolutional Neural Network."""
    def __init__(self):
        """Constructor."""
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        
        # self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc1 = nn.Linear(49152, 256)
        self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(128, 16)

    def forward(self, x):
        """Forward computation."""
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = inputs.view(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    BATCH_SIZE = 64
    INPUT_DIM = 64
    HIDDEN_SIZE = 64
    N_STEPS = 1000
    # LR = 1e-3
    LR = 6e-4
    N_LAYERS = 16
    WITH_BIAS = True
    MODEL_NAME = "gpt2"
    DATA_NAME = "CohereForAI/aya_dataset"
    
    torch.cuda.empty_cache()

    fp32_linear = Net().to("cuda")

    # bf16_linear = deepcopy(fp32_linear)
    fp8_linear = deepcopy(fp32_linear)
    fp8_linear_with_scaler = deepcopy(fp32_linear)
    msamp_linear = deepcopy(fp32_linear)
    msamp_linear_with_scaler = deepcopy(fp32_linear)

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

    # loss_func = nn.CrossEntropyLoss()
    # def loss_func(outputs, targets):
    #     func = nn.CrossEntropyLoss()
    #     logits = outputs.logits
    #     logits = logits[:, :-1, :].contiguous()
    #     return func(logits.view(-1, logits.shape[-1]), targets.view(-1))
    
    loss_func = nn.CrossEntropyLoss()

    # batch_inputs = []
    # batch_targets = []
    # inputs = torch.randn(BATCH_SIZE, HIDDEN_SIZE).to("cuda")
    # targets = torch.randint(0, HIDDEN_SIZE, (BATCH_SIZE,)).to("cuda")
    # for _ in range(N_STEPS):
    #     batch_inputs.append(inputs.clone())
    #     batch_targets.append(targets.clone())

    # dataset = load_dataset(DATA_NAME)
    # dataset = dataset.map(
    #     lambda x: tokenizer(x["inputs"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    # )
    # dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # dataloaders = DataLoader(dataset["train"], batch_size=32, shuffle=True)
    
    train_loader, test_loader = get_dataloader()

    fp32_losses = []
    fp8_with_loss_scaler_losses = []
    msamp_with_loss_scaler_losses = []
    
    wandb.init(
        project="fp8_for_nanotron",
        name=f"{get_time_name()}.convergence_fp8_n_layers_{N_LAYERS}_and_input_dim_{INPUT_DIM}_and_hidden_size_{HIDDEN_SIZE}_and_lr_{LR}_and_bias_{WITH_BIAS}_and_batch_size_{BATCH_SIZE}",
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
        for step, batch in enumerate(train_loader):
            print(f"step: {step} /n /n")
            inputs, labels = batch[0].to("cuda"), batch[1].to("cuda")

            # batch = {k: v.squeeze(dim=1).to("cuda") for k, v in batch.items()}
            # inputs = batch["input_ids"]
            # targets = batch["input_ids"][:, 1:].contiguous()

            # # inputs = batch_inputs[step]
            # # targets = batch_targets[step]
            # inputs = torch.randn(BATCH_SIZE, INPUT_DIM).to("cuda")
            # targets = torch.randint(0, HIDDEN_SIZE, (BATCH_SIZE,)).to("cuda")

            fp32_logs = _log(fp32_linear)
            fp32_optim.zero_grad()
            ref_output = fp32_linear(inputs)
            fp32_loss = loss_func(ref_output, labels)
            fp32_loss.backward()
            fp32_optim.step()

            # bf16_optim.zero_grad()
            # bf16_output = bf16_linear(inputs.to(dtype=torch.bfloat16))
            # bf16_loss = loss_func(bf16_output, targets)
            # bf16_loss.backward()
            # bf16_optim.step()
            
            # fp8_logs = _log(fp8_linear)
            # fp8_optim.zero_grad()
            # fp8_output = fp8_linear(inputs)
            # fp8_loss = loss_func(fp8_output, targets)
            # fp8_loss.backward()
            # fp8_optim.step()
            
            # fp8_logs = _log(fp8_linear)
            # fp8_optim.zero_grad()
            # fp8_output = fp8_linear(inputs)
            # fp8_loss = loss_func(fp8_output, labels)
            # fp8_loss.backward()
            # fp8_optim.step()

            # fp8_with_scaler_logs = _log(fp8_linear_with_scaler)
            # fp8_optim_with_scaler.zero_grad()
            # fp8_output_with_scaler = fp8_linear_with_scaler(inputs)
            # fp8_loss_with_scaler = loss_func(fp8_output_with_scaler, labels)
            # scaled_fp8_loss_with_scaler = fp8_scaler.scale(fp8_loss_with_scaler)
            # scaled_fp8_loss_with_scaler.backward()
            # fp8_scaler.step(fp8_optim_with_scaler)
            # fp8_scaler.update()

            # msamp_logs = _log(msamp_linear)
            msamp_optim.zero_grad()
            msamp_output = msamp_linear(inputs)
            msamp_loss = loss_func(msamp_output, labels)
            msamp_loss.backward()
            msamp_optim.all_reduce_grads(msamp_linear)
            msamp_optim.step()

            # msamp_with_loss_scaler_logs = _log(msamp_linear_with_scaler)
            msamp_optim_with_scaler.zero_grad()
            msamp_output_with_scaler = msamp_linear_with_scaler(inputs)
            msamp_loss_with_scaler = loss_func(msamp_output_with_scaler, labels)
            scaled_msamp_loss_with_scaler = msamp_scaler.scale(msamp_loss_with_scaler)
            scaled_msamp_loss_with_scaler.backward()
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
                    # "scaled_fp8_loss_with_scaler": scaled_fp8_loss_with_scaler.item(),
                    
                    "msamp_o2_loss": msamp_loss.item(),
                    "msamp_o2_loss_with_scaler": msamp_loss_with_scaler.item(),
                    "scaled_msamp_o2_loss_with_scaler": scaled_msamp_loss_with_scaler.item(),
                    
                    # "l1_norm_diff_fp8_relative_to_fp32": l1_norm_diff(
                    #     fp8_loss, fp32_loss
                    # ).item(),
                    # "l1_norm_diff_fp8_with_loss_scaler_relative_to_fp32": l1_norm_diff(
                    #     fp8_loss_with_scaler, fp32_loss
                    # ).item(),
                    "l1_norm_diff_msamp_with_loss_scaler_relative_to_fp32": l1_norm_diff(
                        msamp_loss_with_scaler, fp32_loss
                    ).item(),
                    **convert_logs_to_flat_logs(fp32_logs, prefix="fp32"),
                    # **convert_logs_to_flat_logs(fp8_logs, prefix="fp8"),
                    # **convert_logs_to_flat_logs(fp8_with_scaler_logs, prefix="fp8_with_scaler"),
                    # "l1_norm_diff_fp8_with_loss_scaler_relative_to_bf16": l1_norm_diff(fp8_loss_with_scaler, bf16_loss).item(),
                    # "l1_norm_diff_msamp_with_loss_scaler_relative_to_bf16": l1_norm_diff(msamp_loss_with_scaler, bf16_loss).item(),
                }
            )

        # NOTE: 3e-4 is from msamp
        # torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0.1, atol=3e-4)
        # torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
