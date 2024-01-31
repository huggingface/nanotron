from typing import Dict

import torch
import torch.distributed as dist
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.doremi.utils import masked_mean
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from torch import nn


def compute_per_domain_loss(
    losses: torch.Tensor, domain_idxs: torch.Tensor, doremi_context: DoReMiContext, parallel_context: ParallelContext
) -> torch.Tensor:
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_pg = parallel_context.dp_pg

    # NOTE: can't do allgather([tensor_list], [tensor]) if a tensor in tensor_list is not contiguous
    losses_dp = [torch.empty_like(losses, device="cuda").contiguous() for _ in range(dp_size)]
    dist.all_gather(losses_dp, losses.contiguous(), group=dp_pg)
    losses_dp = torch.cat(losses_dp, dim=0)

    domain_ids_dp = [torch.empty_like(domain_idxs, device="cuda").contiguous() for _ in range(dp_size)]
    dist.all_gather(domain_ids_dp, domain_idxs.contiguous(), group=dp_pg)
    domain_ids_dp = torch.cat(domain_ids_dp, dim=0)

    # NOTE: Calculate total loss per domain
    N_DOMAINS = doremi_context.num_domains
    domain_losses = torch.zeros(N_DOMAINS, device="cuda")
    domain_ids_dp = domain_ids_dp.view(-1)

    assert losses_dp.shape[0] == domain_ids_dp.shape[0]
    GLOBAL_BATCH_SIZE = losses_dp.shape[0]

    for i in range(GLOBAL_BATCH_SIZE):
        # NOTE: sum the excess losses of all tokens in the batch
        # then add it to the domain loss of the corresponding domain
        # domain_losses[domain_idxs[i]] += excess_losses[i].sum(dim=-1)
        domain_losses[domain_ids_dp[i]] += losses_dp[i].sum(dim=-1)

    # NOTE: Normalize and smooth domain weights
    samples_per_domain = torch.bincount(domain_ids_dp, minlength=N_DOMAINS)
    SEQ_LEN = losses.shape[1]
    normalized_domain_losses = domain_losses / (samples_per_domain * SEQ_LEN)
    # NOTE: if the domain loss is zero, then the normalized domain loss is NaN
    normalized_domain_losses[torch.isnan(normalized_domain_losses)] = 0.0

    return normalized_domain_losses, samples_per_domain


class DoReMiLossForProxyTraining:
    def __init__(self, doremi_context: DoReMiContext, parallel_context: ParallelContext):
        self.doremi_context = doremi_context
        self.parallel_context = parallel_context

    def __call__(self, losses: torch.Tensor, ref_losses: torch.Tensor, domain_idxs: torch.Tensor):
        assert losses.shape == ref_losses.shape, "losses and ref_losses must have the same shape"
        assert (
            domain_idxs.shape[0] == losses.shape[0]
        ), "the batch size of domain_idxs must match the batch size of losses"

        # NOTE: sometimes you'll see the domain losses equal to zero.
        # this doesn't mean there are bugs, it just means that in that case,
        # the proxy model is performing better than the reference model
        # => clamp(lower loss - higher loss, 0) = clamp(negative, 0) = 0.
        excess_losses = (losses - ref_losses).clamp(min=0)

        dp_size = dist.get_world_size(self.parallel_context.dp_pg)

        # NOTE: can't do allgather([tensor_list], [tensor]) if a tensor in tensor_list is not contiguous
        excess_losses_dp = [torch.empty_like(excess_losses, device="cuda").contiguous() for _ in range(dp_size)]
        dist.all_gather(excess_losses_dp, excess_losses.contiguous(), group=self.parallel_context.dp_pg)
        excess_losses_dp = torch.cat(excess_losses_dp, dim=0)

        domain_ids_dp = [torch.empty_like(domain_idxs, device="cuda").contiguous() for _ in range(dp_size)]
        dist.all_gather(domain_ids_dp, domain_idxs.contiguous(), group=self.parallel_context.dp_pg)
        domain_ids_dp = torch.cat(domain_ids_dp, dim=0)

        # NOTE: Calculate total loss per domain
        N_DOMAINS = self.doremi_context.num_domains
        domain_losses = torch.zeros(N_DOMAINS, device="cuda")
        domain_ids_dp = domain_ids_dp.view(-1)

        assert excess_losses_dp.shape[0] == domain_ids_dp.shape[0]
        GLOBAL_BATCH_SIZE = excess_losses_dp.shape[0]
        for i in range(GLOBAL_BATCH_SIZE):
            # NOTE: sum the excess losses of all tokens in the batch
            # then add it to the domain loss of the corresponding domain
            # domain_losses[domain_idxs[i]] += excess_losses[i].sum(dim=-1)
            domain_losses[domain_ids_dp[i]] += excess_losses_dp[i].sum(dim=-1)

        # NOTE: Normalize and smooth domain weights
        samples_per_domain = torch.bincount(domain_ids_dp, minlength=N_DOMAINS)
        SEQ_LEN = losses.shape[1]
        normalized_domain_losses = domain_losses / (samples_per_domain * SEQ_LEN)
        # NOTE: if the domain loss is zero, then the normalized domain loss is zero
        normalized_domain_losses[torch.isnan(normalized_domain_losses)] = 0.0

        # NOTE: α_t′ ← α_t-1 exp(η λ_t)
        # updated_domain_weights = self.doremi_context.domain_weights * torch.exp(
        #     self.doremi_context.step_size * normalized_domain_losses
        # )
        # smooth_domain_weights = self._normalize_domain_weights(
        #     updated_domain_weights, self.doremi_context.smoothing_param
        # )

        domain_weights = self.doremi_context.domain_weights
        step_size = self.doremi_context.step_size
        smoothing_param = self.doremi_context.smoothing_param
        log_new_train_domain_weights = torch.log(domain_weights) + step_size * normalized_domain_losses
        log_new_train_domain_weights = log_new_train_domain_weights - torch.logsumexp(
            log_new_train_domain_weights, dim=0
        )
        train_domain_weights = (1 - smoothing_param) * torch.exp(log_new_train_domain_weights) + smoothing_param / len(
            log_new_train_domain_weights
        )
        self.doremi_context.domain_weights = train_domain_weights.detach()

        # return excess_losses, normalized_domain_losses, smooth_domain_weights
        return excess_losses_dp, normalized_domain_losses, train_domain_weights

    # def _normalize_domain_weights(self, weights: torch.Tensor, smoothing_param: float) -> torch.Tensor:
    #     """
    #     Renormalize and smooth domain weights.
    #     alpha_t = (1 - c) * (alpha_t' / sum(i=1 to k of alpha_t'[i])) + c * u
    #     Algorithm 1 DoReMi domain reweighting (Step 2).
    #     """
    #     # NUM_DOMAINS = weights.shape[0]
    #     NUM_DOMAINS = self.doremi_context.num_domains
    #     uniform_weights = torch.ones(NUM_DOMAINS, device=weights.device) / NUM_DOMAINS
    #     normalized_weight = (1 - smoothing_param) * weights / weights.sum(dim=-1) + (smoothing_param * uniform_weights)
    #     return normalized_weight


class CrossEntropyWithPerDomainLoss(nn.Module):
    def __init__(self, doremi_context: DoReMiContext, parallel_context: ParallelContext):
        super().__init__()
        self.doremi_context = doremi_context
        self.parallel_context = parallel_context

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
        domain_idxs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # loss = sharded_cross_entropy(
        #     sharded_logits, label_ids.transpose(0, 1).contiguous(), group=tp_pg, dtype=torch.float
        # ).transpose(0, 1)
        # per_token_loss = sharded_cross_entropy(
        #     sharded_logits, label_ids.transpose(0, 1).contiguous(), group=self.parallel_context.tp_pg, dtype=torch.float
        # ).transpose(0, 1)
        per_token_loss = sharded_cross_entropy(
            sharded_logits, label_ids, group=self.parallel_context.tp_pg, dtype=torch.float
        )
        lm_loss = masked_mean(per_token_loss, label_mask, dtype=torch.float)
        domain_losses, samples_per_domain = compute_per_domain_loss(
            per_token_loss, domain_idxs, self.doremi_context, self.parallel_context
        )
        return {"loss": lm_loss, "domain_losses": domain_losses, "samples_per_domain": samples_per_domain}
