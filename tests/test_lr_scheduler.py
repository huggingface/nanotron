import pytest
import torch
from nanotron.config import LRSchedulerArgs
from nanotron.helpers import lr_scheduler_builder
from torch import nn


def _lr_curve(total_training_steps: int, **scheduler_kwargs) -> list:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.param_groups[0]["initial_lr"] = optimizer.param_groups[0]["lr"]

    scheduler = lr_scheduler_builder(
        optimizer=optimizer,
        lr_scheduler_args=LRSchedulerArgs(**scheduler_kwargs),
        total_training_steps=total_training_steps,
    )

    lrs = []
    for _ in range(total_training_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


@pytest.mark.parametrize(
    "total_training_steps,lr_warmup_steps,lr_decay_starting_step",
    [
        (1000, 100, 100),
        (1000, 100, 500),
        (1000, 200, 900),
        (100, 10, 10),
    ],
)
def test_decay_reaches_min_at_the_end_of_training(
    total_training_steps, lr_warmup_steps, lr_decay_starting_step
):
    """Decay spans lr_decay_starting_step..total_training_steps.

    lr_decay_starting_step is an absolute step index, so the decay horizon is
    total_training_steps - lr_decay_starting_step. Subtracting warmup as well
    ended decay early, leaving the tail of training pinned at min_decay_lr.
    """
    min_decay_lr = 0.1
    lrs = _lr_curve(
        total_training_steps,
        learning_rate=1.0,
        lr_warmup_steps=lr_warmup_steps,
        lr_warmup_style="linear",
        lr_decay_style="linear",
        lr_decay_starting_step=lr_decay_starting_step,
        min_decay_lr=min_decay_lr,
    )

    # Still decaying on the last step rather than long since floored.
    assert lrs[-1] > min_decay_lr
    assert lrs[-1] == pytest.approx(min_decay_lr, abs=0.01)

    # And strictly decreasing across the decay phase, so there is no cliff.
    decay_phase = lrs[lr_decay_starting_step:]
    assert all(b < a for a, b in zip(decay_phase, decay_phase[1:]))


def test_decay_starting_step_defaulting_to_warmup_is_a_no_op():
    """Setting lr_decay_starting_step to its own documented default changes nothing."""
    common = dict(
        learning_rate=1.0,
        lr_warmup_steps=100,
        lr_warmup_style="linear",
        lr_decay_style="linear",
        min_decay_lr=0.0,
    )
    implicit = _lr_curve(1000, lr_decay_starting_step=None, **common)
    explicit = _lr_curve(1000, lr_decay_starting_step=100, **common)
    assert implicit == pytest.approx(explicit)
