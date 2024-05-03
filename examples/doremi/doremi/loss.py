from typing import Dict, Tuple

import torch
import torch.distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from torch import nn

from .doremi_context import DoReMiContext
from .utils import masked_mean


def compute_per_domain_loss(
    losses: torch.Tensor, domain_idxs: torch.Tensor, doremi_context: DoReMiContext, parallel_context: ParallelContext
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_pg = parallel_context.dp_pg

    # NOTE: can't do allgather([tensor_list], [tensor]) if a tensor in tensor_list is not contiguous
    losses_dp = [
        torch.empty_like(losses, device="cuda", memory_format=torch.contiguous_format) for _ in range(dp_size)
    ]
    dist.all_gather(losses_dp, losses.contiguous(), group=dp_pg)
    losses_dp = torch.cat(losses_dp, dim=0)

    domain_ids_dp = [
        torch.empty_like(domain_idxs, device="cuda", memory_format=torch.contiguous_format) for _ in range(dp_size)
    ]
    dist.all_gather(domain_ids_dp, domain_idxs.contiguous(), group=dp_pg)
    domain_ids_dp = torch.cat(domain_ids_dp, dim=0)

    # NOTE: Calculate total loss per domain
    n_domains = doremi_context.num_domains
    domain_losses = torch.zeros(n_domains, device="cuda")
    domain_ids_dp = domain_ids_dp.view(-1)

    assert losses_dp.shape[0] == domain_ids_dp.shape[0]
    GLOBAL_BATCH_SIZE = losses_dp.shape[0]

    for i in range(GLOBAL_BATCH_SIZE):
        # NOTE: sum the excess losses of all tokens in the batch
        # then add it to the domain loss of the corresponding domain
        domain_losses[domain_ids_dp[i]] += losses_dp[i].sum(dim=-1)

    # NOTE: Normalize and smooth domain weights
    samples_per_domain = torch.bincount(domain_ids_dp, minlength=n_domains)
    SEQ_LEN = losses.shape[1]
    normalized_domain_losses = domain_losses / (samples_per_domain * SEQ_LEN)
    # NOTE: if the domain loss is zero, then the normalized domain loss is NaN
    normalized_domain_losses[torch.isnan(normalized_domain_losses)] = 0.0
    return losses_dp, normalized_domain_losses, samples_per_domain


def compute_domain_loss_per_replicas(
    losses: torch.Tensor, domain_idxs: torch.Tensor, doremi_context: DoReMiContext
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    domain_idxs = domain_idxs.view(-1)

    # NOTE: Calculate total loss per domain
    n_domains = doremi_context.num_domains
    domain_losses = torch.zeros(n_domains, device="cuda")

    assert losses.shape[0] == domain_idxs.shape[0]
    GLOBAL_BATCH_SIZE = domain_idxs.shape[0]

    for i in range(GLOBAL_BATCH_SIZE):
        # NOTE: sum the excess losses of all tokens in the batch
        # then add it to the domain loss of the corresponding domain
        domain_losses[domain_idxs[i]] += losses[i].sum(dim=-1)

    # NOTE: Normalize domain weights
    SEQ_LEN = losses.shape[1]
    samples_per_domain = torch.bincount(domain_idxs, minlength=n_domains)
    normalized_domain_losses = domain_losses / (samples_per_domain * SEQ_LEN)

    # NOTE: if the domain loss is zero, then the normalized domain loss is NaN
    normalized_domain_losses[torch.isnan(normalized_domain_losses)] = 0.0
    return normalized_domain_losses, samples_per_domain


class DomainLossForProxyTraining:
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
        normalized_domain_losses, samples_per_domain = compute_domain_loss_per_replicas(
            excess_losses, domain_idxs, self.doremi_context
        )

        domain_weights = self.doremi_context.domain_weights
        step_size = self.doremi_context.step_size
        smoothing_param = self.doremi_context.smoothing_param
        log_new_domain_weights = torch.log(domain_weights) + step_size * normalized_domain_losses
        log_new_domain_weights = log_new_domain_weights - torch.logsumexp(log_new_domain_weights, dim=0)
        train_domain_weights = (1 - smoothing_param) * torch.exp(log_new_domain_weights) + smoothing_param / len(
            log_new_domain_weights
        )
        dro_loss = (train_domain_weights * normalized_domain_losses).sum(dim=-1)

        return {
            "dro_loss": dro_loss,
            "domain_losses": normalized_domain_losses,
            "domain_weights": train_domain_weights,
            "samples_per_domain": samples_per_domain,
        }


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
        per_token_loss = sharded_cross_entropy(
            sharded_logits, label_ids, group=self.parallel_context.tp_pg, dtype=torch.float
        )
        ce_loss = masked_mean(per_token_loss, label_mask, dtype=torch.float)
        _, domain_losses, samples_per_domain = compute_per_domain_loss(
            per_token_loss, domain_idxs, self.doremi_context, self.parallel_context
        )
        return {"ce_loss": ce_loss, "domain_losses": domain_losses, "samples_per_domain": samples_per_domain}


class DoReMiLossForProxyTraining(nn.Module):
    def __init__(self, doremi_context: DoReMiContext, parallel_context: ParallelContext):
        super().__init__()
        self.parallel_context = parallel_context
        self.doremi_loss = DomainLossForProxyTraining(doremi_context, parallel_context)

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
        domain_idxs: torch.Tensor,
        ref_losses: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss = sharded_cross_entropy(
            sharded_logits,
            label_ids,
            group=self.parallel_context.tp_pg,
            dtype=torch.float,
        )
        ce_loss = masked_mean(loss, label_mask, dtype=torch.float)
        doremi_loss_outputs = self.doremi_loss(loss, ref_losses, domain_idxs)

        return {
            "ce_loss": ce_loss,
            "loss": doremi_loss_outputs["dro_loss"],  # NOTE: this is the one we optimize
            "domain_losses": doremi_loss_outputs["domain_losses"],
            "domain_weights": doremi_loss_outputs["domain_weights"],
            "samples_per_domain": doremi_loss_outputs["samples_per_domain"],
        }
