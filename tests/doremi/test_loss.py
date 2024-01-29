import torch
import torch.nn.functional as F
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.doremi.loss import DoReMiLossForProxyTraining


def test_doremi_loss():
    BATCH_SIZE = 512
    SEQ_LEN = 128
    N_DOMAINS = 5

    domain_keys = [f"domain {i}" for i in range(N_DOMAINS)]
    domain_weights = F.softmax(torch.ones(N_DOMAINS, requires_grad=False, device="cuda"), dim=-1)
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)

    losses = torch.rand(BATCH_SIZE, SEQ_LEN, device="cuda")
    ref_losses = torch.rand(BATCH_SIZE, SEQ_LEN, device="cuda")
    domain_idxs = torch.randint(0, N_DOMAINS, (BATCH_SIZE,), device="cuda")
    loss_func = DoReMiLossForProxyTraining(doremi_context)

    excess_loss, domain_losses, domain_weights = loss_func(losses, ref_losses, domain_idxs)

    assert excess_loss.shape == (BATCH_SIZE, SEQ_LEN)
    assert domain_losses.shape == (N_DOMAINS,)
    assert domain_weights.shape == (N_DOMAINS,)
    assert torch.allclose(domain_weights.sum(dim=-1), torch.tensor(1.0))
