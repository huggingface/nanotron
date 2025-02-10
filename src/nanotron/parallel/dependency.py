from typing import Dict, Tuple

import torch
from torch import Tensor

_phonies: Dict[Tuple[torch.device, bool], Tensor] = {}


def get_phony(device: torch.device, *, requires_grad: bool) -> Tensor:
    """Gets a phony. Phony is tensor without space. It is useful to make
    arbitrary dependency in a autograd graph because it doesn't require any
    gradient accumulation.

    .. note::

        Phonies for each device are cached. If an autograd function gets a phony
        internally, the phony must be detached to be returned. Otherwise, the
        autograd engine will mutate the cached phony in-place::

            class Phonify(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    phony = get_phony(input.device, requires_grad=False)
                    return phony.detach()  # detach() is necessary.

    """
    key = (device, requires_grad)

    try:
        phony = _phonies[key]
    except KeyError:
        with torch.cuda.stream(torch.cuda.default_stream(device)):
            phony = torch.empty(0, device=device, requires_grad=requires_grad)

        _phonies[key] = phony

    return phony


def fork(input: Tensor) -> Tuple[Tensor, Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)

    return input, phony


class Fork(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Fork", input: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        phony = get_phony(input.device, requires_grad=False)
        return input, phony.detach()

    @staticmethod
    def backward(ctx: "Fork", grad_input: Tensor, grad_grad: Tensor) -> Tensor:  # type: ignore
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        return grad_input


def join(input: Tensor, phony: Tensor) -> Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)

    return input


class Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Join", input: Tensor, phony: Tensor) -> Tensor:  # type: ignore
        return input

    @staticmethod
    def backward(ctx: "Join", grad_input: Tensor) -> Tuple[Tensor, None]:  # type: ignore
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        return grad_input, None


# def depend(fork_from, join_to) -> None:
#     # Ensure that batches[i-1] is executed after batches[i] in
#     # # backpropagation by an explicit dependency.
#     # if i != 0:
#     #     depend(batches[i-1], batches[i])
#     # depend(run_after, run_before)
#     fork_from, phony = fork(fork_from)
#     join_to = join(join_to, phony)
#     return fork_from, join_to


def depend(run_after, run_before) -> None:
    # Ensure that batches[i-1] is executed after batches[i] in
    # # backpropagation by an explicit dependency.
    # if i != 0:
    #     depend(batches[i-1], batches[i])
    # depend(run_after, run_before)
    run_after, phony = fork(run_after)
    run_before = join(run_before, phony)
    return run_after, run_before
