import torch
import torch.nn.functional as F
from nanotron.doremi.loss import compute_excess_loss, normalize_domain_weights, update_domain_weights


def test_compute_per_domain_excess_loss():
    BATCH_SIZE, SEQ_LEN = 10, 5

    ref_log_probs = F.softmax(torch.randn(BATCH_SIZE, SEQ_LEN), dim=-1)
    proxy_log_probs = F.softmax(torch.randn(BATCH_SIZE, SEQ_LEN), dim=-1)

    excess_loss = compute_excess_loss(ref_log_probs, proxy_log_probs)

    # NOTE: DoReMi paper's page 2
    # "Since the Group DRO optimizer (Sagawa et al., 2020) requires
    # a non-negative loss, we clip the per-token excess loss at 0"
    assert excess_loss.dim() == 0


def test_update_domain_weights():
    LEARNING_RATE, NUM_DOMAINS, EXCESS_LOSS = 1e-3, 10, 0.69
    weights = torch.rand(NUM_DOMAINS)

    updated_weights = update_domain_weights(LEARNING_RATE, EXCESS_LOSS, weights)

    assert updated_weights.shape == weights.shape


def test_normalize_domain_weights():
    NUM_DOMAINS, SMOOTHING_PARAM = 10, 1e-3
    weights = torch.rand(NUM_DOMAINS)

    normalized_weights = normalize_domain_weights(weights, SMOOTHING_PARAM)

    assert normalized_weights.shape == weights.shape
    assert torch.allclose(normalized_weights.sum(), torch.tensor(1.0))
