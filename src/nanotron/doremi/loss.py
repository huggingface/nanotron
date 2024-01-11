import torch
import torch.nn.functional as F


def compute_excess_loss(ref_log_probs: torch.Tensor, proxy_log_probs: torch.Tensor) -> torch.Tensor:
    """Algorithm 1 DoReMi domain reweighting (Step 2)."""
    excess_loss = F.relu(ref_log_probs - proxy_log_probs)
    excess_loss = excess_loss.sum(dim=-1).sum()

    # NOTE: normalize the loss by summing over the context length of all samples in a batch
    # TODO(xrsrke): make this generalizable to other contexts
    CONTEXT_LENGTH = ref_log_probs.shape[-1]
    BATCH_SIZE = ref_log_probs.shape[0]
    excess_loss /= CONTEXT_LENGTH * BATCH_SIZE
    return excess_loss


def update_domain_weights(
    learning_rate: torch.Tensor, excess_loss: torch.Tensor, weights: torch.Tensor
) -> torch.tensor:
    """
    Compute the new domain weights.
    Update domain weights (exp is entrywise): alpha_t' = alpha_(t-1) * exp(eta * lambda_t)
    Algorithm 1 DoReMi domain reweighting (Step 2).
    """
    return weights * learning_rate * excess_loss


def _init_initial_domain_weights(num_domains: int) -> torch.Tensor:
    return torch.ones(num_domains) / num_domains


def normalize_domain_weights(weights: torch.Tensor, smoothing_param: float = 1e-3) -> torch.Tensor:
    """
    Renormalize and smooth domain weights.
    alpha_t = (1 - c) * (alpha_t' / sum(i=1 to k of alpha_t'[i])) + c * u
    Algorithm 1 DoReMi domain reweighting (Step 2).
    """
    NUM_DOMAINS = weights.shape[0]
    uniform_weights = torch.ones(NUM_DOMAINS) / NUM_DOMAINS
    normalized_weight = (1 - smoothing_param) * weights / weights.sum() + (smoothing_param * uniform_weights)
    return normalized_weight
