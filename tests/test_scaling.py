import pytest
import torch
from helpers.dummy import DummyModel
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.scaling import mu_transfer
from torch import nn


def build_dummy_model_without_init(parallel_context: ParallelContext, device: torch.device) -> nn.Module:
    p2p = P2P(pg=parallel_context.pp_pg, device=device)
    model = DummyModel(p2p=p2p)
    return model


@pytest.mark.parametrize("n_layers", [1, 3, 4])
def test_mu_transfer(n_layers):
    HIDDEN_SIZE = 16

    class Model(nn.Module):
        def __init__(self, hidden_size: int, n_layers: int):
            super().__init__()
            self.net = nn.Sequential(
                *[layer for _ in range(n_layers) for layer in (nn.Linear(hidden_size, hidden_size), nn.ReLU())]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    ref_model = Model(HIDDEN_SIZE, n_layers)
    model = Model(HIDDEN_SIZE, n_layers)
    model = mu_transfer(model)

    assert isinstance(model, nn.Module)
    assert len(list(model.modules())) == len(list(ref_model.modules()))
    assert model.numel() == ref_model.numel()

    for ref_module, module in zip(ref_model.modules(), model.modules()):
        assert ref_module.__class__ is module.__class__
        if isinstance(ref_module, nn.Linear):
            assert ref_module.weight.shape == module.weight.shape
            assert ref_module.bias == module.bias.shape


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(1, i, 1) for i in range(1, min(4, available_gpus()) + 1)])
@rerun_if_address_is_in_use()
def test_mu_transformer_parametrization(tp: int, dp: int, pp: int):
    init_distributed(pp=pp, dp=dp, tp=tp)(_test_mu_transformer_parametrization)()


def _test_mu_transformer_parametrization(parallel_context: ParallelContext):

    # ref_model = init_dummy_model(parallel_context=parallel_context)
    # model = init_dummy_model(parallel_context=parallel_context)
    build_dummy_model_without_init(parallel_context=parallel_context, device=torch.device("cuda"))
    model = build_dummy_model_without_init(parallel_context=parallel_context, device=torch.device("cuda"))
    model = mu_transfer(model, dtype=torch.float32, parallel_context=parallel_context)

    parallel_context.destroy()


# TODO(xrsrke): test scaling in MLP, attention, logits, input embeddings

# TODO(xrsrke): initialize TP modules using the original width
