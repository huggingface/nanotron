from typing import Union

import torch
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.utils import checkpoint_method
from torch import nn


class CheckpointedModel(nn.Module):
    def __init__(self, is_checkpointed: bool = False):
        super().__init__()
        self.dense1 = nn.Linear(10, 10)
        self.dense2 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(0.1)
        self.is_checkpointed = is_checkpointed
        self.fwd_counter = 0

    @checkpoint_method("is_checkpointed")
    def forward(self, x: Union[torch.Tensor, TensorPointer]):
        x = self.dense1(x)
        if self.is_checkpointed and self.fwd_counter == 0:
            assert not x.requires_grad, "x should not require grad when checkpointed, because fwd runs in no_grad mode"
            assert (
                x.grad_fn is None
            ), "x should not store any activation when checkpointed, because fwd runs in no_grad mode"
        x = self.dense2(x)
        x = self.dropout(x)
        self.fwd_counter += 1
        return x


class DummyModel(nn.Module):
    def __init__(self, is_checkpointed: bool = False):
        super().__init__()
        self.dense0 = nn.Linear(10, 10)
        self.checkpointed_model = CheckpointedModel(is_checkpointed=is_checkpointed)
        self.dense3 = nn.Linear(10, 10)

    def forward(self, x: Union[torch.Tensor, TensorPointer]):
        x = self.dense0(x)
        x = self.checkpointed_model(x)
        assert x.requires_grad  # inside forward, x should require grad even if calculated in no_grad mode
        x = self.dense3(x)
        return x


def test_activation_checkpointing():
    dtype = torch.float16
    device = torch.device("cuda")
    test_model = DummyModel(is_checkpointed=True)
    ref_model = DummyModel(is_checkpointed=False)
    for model in [test_model, ref_model]:
        model.to(device=device, dtype=dtype)

    # copy weights
    test_model.load_state_dict(ref_model.state_dict())
    assert test_model.checkpointed_model.is_checkpointed is True
    assert ref_model.checkpointed_model.is_checkpointed is False

    # generate random input
    x = torch.randn(10, 10, device=device, dtype=dtype)

    # Forward pass
    with torch.random.fork_rng(devices=["cuda"]):
        ref_output = ref_model(x)
    checkpointed_output = test_model(x)
    assert test_model.checkpointed_model.fwd_counter == 1
    torch.testing.assert_close(checkpointed_output, ref_output)

    # Backward pass (check that fwd is called twice, and that we don't store the activations)
    ref_output.sum().backward()
    assert ref_model.checkpointed_model.fwd_counter == 1, "ref_model fwd should not be called twice"

    # make sure grads are not synced between test_model and ref_model
    assert ref_model.dense0.weight.grad is not None
    assert test_model.dense0.weight.grad is None

    assert test_model.checkpointed_model.fwd_counter == 1
    checkpointed_output.sum().backward()
    assert test_model.checkpointed_model.fwd_counter == 2, "test_model fwd should be called twice"

    # compare all models grads
    for ref_param, checkpointed_param in zip(ref_model.parameters(), test_model.parameters()):
        torch.testing.assert_close(ref_param.grad, checkpointed_param.grad)


# TODO @nouamanetazi: test `checkpoint_method` vs `torch.utils.checkpoint.checkpoint`
# TODO @nouamanetazi: test a method with kwargs values
# TODO @nouamanetazi: test `checkpoint_method` in a distributed setting
# TODO @nouamanetazi: test BatchNorm layers with checkpointing
