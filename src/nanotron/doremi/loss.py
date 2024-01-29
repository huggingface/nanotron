import torch
from nanotron.doremi.doremi_context import DoReMiContext


class DoReMiLossForProxyTraining:
    def __init__(self, doremi_context: DoReMiContext):
        self.doremi_context = doremi_context

    def __call__(self, losses: torch.Tensor, ref_losses: torch.Tensor, domain_idxs: torch.Tensor):
        assert losses.shape == ref_losses.shape

        # NOTE: per token loss
        # losses = (logprobs * label_mask).sum(dim=-1) / label_mask.sum(dim=-1)
        # NOTE: sometimes you'll see the domain losses equal to zero.
        # this doesn't mean there are bugs, it just means that in that case,
        # the proxy model is performing better than the reference model
        # => clamp(lower loss - higher loss, 0) = clamp(negative, 0) = 0.
        excess_losses = (losses - ref_losses).clamp(min=0)

        # NOTE: Calculate total loss per domain
        domain_idxs = domain_idxs.view(-1)
        domain_losses = torch.zeros(domain_idxs.max() + 1, device="cuda")

        BATCH_SIZE = excess_losses.shape[0]
        for i in range(BATCH_SIZE):
            domain_losses[domain_idxs[i]] += excess_losses[i].sum(dim=-1)

        # for i in range(len(excess_losses)):
        #     domain_losses[domain_idxs[i]] += excess_losses[i]

        # if self.iteration == 4:
        #     assert 1 == 1

        # NOTE: Normalize and smooth domain weights
        tokens_per_domain = torch.bincount(domain_idxs, minlength=domain_idxs.max() + 1)
        normalized_domain_losses = domain_losses / tokens_per_domain

        # NOTE: α_t′ ← α_t-1 exp(η λ_t)
        updated_domain_weights = self.doremi_context.domain_weights * torch.exp(
            self.doremi_context.step_size * normalized_domain_losses
        )
        smooth_domain_weights = self._normalize_domain_weights(
            updated_domain_weights, self.doremi_context.smoothing_param
        )
        self.doremi_context.domain_weights = smooth_domain_weights.detach()

        return excess_losses, normalized_domain_losses, smooth_domain_weights

    def _normalize_domain_weights(self, weights: torch.Tensor, smoothing_param: float) -> torch.Tensor:
        """
        Renormalize and smooth domain weights.
        alpha_t = (1 - c) * (alpha_t' / sum(i=1 to k of alpha_t'[i])) + c * u
        Algorithm 1 DoReMi domain reweighting (Step 2).
        """
        # NUM_DOMAINS = weights.shape[0]
        NUM_DOMAINS = self.doremi_context.num_domains
        uniform_weights = torch.ones(NUM_DOMAINS, device=weights.device) / NUM_DOMAINS
        normalized_weight = (1 - smoothing_param) * weights / weights.sum(dim=-1) + (smoothing_param * uniform_weights)
        return normalized_weight
