import pytest
import torch
from nanotron.config import LRSchedulerArgs
from nanotron.helpers import lr_scheduler_builder
from nanotron.optim.named_optimizer import NamedOptimizer


@pytest.mark.parametrize(
    "group_lrs",
    [
        pytest.param([3e-4], id="standard"),
        # With µP each parameter group has its own learning rate
        pytest.param([3e-4, 3e-4 * 0.25], id="mup"),
    ],
)
def test_lr_scheduler_without_warmup_or_decay_keeps_lr_constant(monkeypatch, group_lrs):
    # lr_scheduler_builder only needs a process group for logging
    monkeypatch.setattr("nanotron.helpers.log_rank", lambda *args, **kwargs: None)

    param_groups = [
        {"named_params": [(f"param_{i}", torch.nn.Parameter(torch.zeros(2)))], "lr": lr}
        for i, lr in enumerate(group_lrs)
    ]
    optimizer = NamedOptimizer(param_groups, optimizer_builder=lambda groups: torch.optim.SGD(groups, lr=group_lrs[0]))
    lr_scheduler_args = LRSchedulerArgs(
        learning_rate=group_lrs[0],
        lr_warmup_steps=0,
        lr_warmup_style="linear",
        lr_decay_steps=0,
    )
    lr_scheduler = lr_scheduler_builder(optimizer, lr_scheduler_args, total_training_steps=10)

    for _ in range(4):
        lrs = [group["lr"] for group in optimizer.get_base_optimizer().param_groups]
        assert lrs == pytest.approx(group_lrs)
        optimizer.step()
        lr_scheduler.step()
