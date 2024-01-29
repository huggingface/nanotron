import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from helpers.utils import (
    init_distributed,
)
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.doremi.loss import DoReMiLossForProxyTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy


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

    # NOTE: no values in excess_loss should be negative
    assert excess_loss.min() >= 0.0
    assert excess_loss.shape == (BATCH_SIZE, SEQ_LEN)

    assert domain_losses.min() >= 0.0
    assert domain_losses.shape == (N_DOMAINS,)

    assert domain_weights.shape == (N_DOMAINS,)
    assert torch.allclose(domain_weights.sum(dim=-1), torch.tensor(1.0))


def _test_computing_per_token_loss(parallel_context: ParallelContext, logits, targets, ref_losses):
    def get_partition(logits, parallel_context):
        tp_size = dist.get_world_size(parallel_context.tp_pg)
        tp_rank = dist.get_rank(parallel_context.tp_pg)
        VOCAB_SIZE = logits.shape[-1]
        per_partition = VOCAB_SIZE // tp_size
        chunks = torch.split(logits, per_partition, dim=-1)
        return chunks[tp_rank]

    logits = logits.to("cuda")
    targets = targets.to("cuda")
    parallel_logits = get_partition(logits, parallel_context)

    loss = sharded_cross_entropy(parallel_logits, targets, parallel_context.tp_pg)

    assert torch.allclose(loss.cpu().view(-1), ref_losses)


@pytest.mark.parametrize("tp", [1, 2])
def test_computing_per_token_loss(tp: int):
    BATCH_SIZE = 512
    SEQ_LEN = 128
    VOCAB_SIZE = 4

    torch.manual_seed(69)

    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    ref_losses = F.cross_entropy(logits.view(-1, logits.size(2)), targets.view(-1), reduction="none")

    init_distributed(tp=tp, dp=1, pp=1)(_test_computing_per_token_loss)(
        logits=logits, targets=targets, ref_losses=ref_losses
    )
