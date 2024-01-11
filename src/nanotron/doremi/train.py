from copy import deepcopy

import torch
import torch.nn.functional as F
from nanotron.doremi.loss import compute_excess_loss, normalize_domain_weights
from torch.optim import Adam
from torch.utils.data import DataLoader


def train(model, dataset):
    STEP_SIZE = 0.1
    NUM_EPOCHS = 1
    ref_model = deepcopy(model)
    optim = Adam(model.parameters(), lr=1e-3)
    NUM_DOMAINS = len(dataset)

    model.train()
    domain_weights = torch.ones(NUM_DOMAINS, requires_grad=False) / NUM_DOMAINS
    with torch.no_grad():
        accumulted_domain_weights = domain_weights.clone().detach()

    for epoch in range(NUM_EPOCHS):
        excess_losses = {domain_id: [] for domain_id in dataset}

        for domain_id in dataset:
            dataloader = DataLoader(dataset[domain_id], batch_size=5)

            for batch in dataloader:
                logits = model(**batch, labels=batch["input_ids"]).logits
                log_probs = F.log_softmax(logits, dim=-1)

                with torch.no_grad():
                    ref_logits = ref_model(**batch, labels=batch["input_ids"]).logits
                    # TODO(xrsrke): remove the noise for testing purposes
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1) + torch.randn_like(ref_logits) * 1e-1

                excess_loss = compute_excess_loss(ref_log_probs, log_probs)
                excess_losses[domain_id].append(excess_loss)

        # NOTE: sum the excess losses within each domain
        per_domain_excess_losses = {domain_id: torch.stack(excess_losses[domain_id]).sum() for domain_id in dataset}
        per_domain_excess_losses = [per_domain_excess_losses[domain_id] for domain_id in dataset]
        per_domain_excess_losses = torch.stack(per_domain_excess_losses, dim=0)

        with torch.no_grad():
            # update weight
            domain_weights = domain_weights * torch.exp(STEP_SIZE * per_domain_excess_losses)
            domain_weights = normalize_domain_weights(domain_weights)

        # loss
        loss = torch.dot(domain_weights, per_domain_excess_losses).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            accumulted_domain_weights += domain_weights

    return accumulted_domain_weights / NUM_EPOCHS
