"""The mnist exampe using MS-AMP. It is adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py."""

from __future__ import print_function

import msamp
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    """The main function."""
    torch.manual_seed(42)

    input = torch.randn(16, 16, device="cuda")
    model = nn.Linear(16, 16, bias=True, device="cuda")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model, optimizer = msamp.initialize(model, optimizer, opt_level="O2")

    model.train()
    for step in range(5):
        print(f"############# step: {step}")
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input)

        output.sum().backward()
        optimizer.step()


if __name__ == "__main__":
    main()
