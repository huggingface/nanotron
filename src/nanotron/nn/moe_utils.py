import torch

moe_load_balance_loss_tracker = {}
moe_token_distribution_tracker = {}


def clear_token_per_expert_tracker():
    """Clear the token per expert tracker."""
    moe_token_distribution_tracker["token_per_expert"] = {}


def save_token_per_expert(
    routing_indices: torch.Tensor,
    layer_number: int,
    num_experts: int,
    num_layers: int,
):
    """Save the token per expert, and total tokens for logging."""
    if "token_per_expert" not in moe_token_distribution_tracker:
        moe_token_distribution_tracker["token_per_expert"] = {}

    if layer_number not in moe_token_distribution_tracker["token_per_expert"]:
        moe_token_distribution_tracker["token_per_expert"][layer_number] = torch.zeros(
            num_experts, device=routing_indices.device
        )

    num_tokens_per_expert = torch.bincount(routing_indices.flatten(), minlength=num_experts)
    for expert_idx in range(num_experts):
        moe_token_distribution_tracker["token_per_expert"][layer_number][expert_idx] += num_tokens_per_expert[
            expert_idx
        ]


def log_token_distribution_per_layer(name: str, iteration: int, wandb_writer):
    token_data = moe_token_distribution_tracker["token_per_expert"]
    total_tokens = moe_token_distribution_tracker["token_per_expert"][0].sum()

    for layer, expert_counts in token_data.items():
        if total_tokens == 0:
            raise ValueError(f"Total tokens for layer {layer} is 0")

        # Fraction of tokens per expert
        fractions = (expert_counts / total_tokens).tolist()
        num_experts = len(fractions)

        # Store the current iteration's fractions
        # Note: You'll need to maintain these across iterations
        if not hasattr(log_token_distribution_per_layer, "iterations"):
            log_token_distribution_per_layer.iterations = {}

        layer_key = f"{name}/layer_{layer}"
        if layer_key not in log_token_distribution_per_layer.iterations:
            log_token_distribution_per_layer.iterations[layer_key] = {
                "iterations": [],
                "expert_data": [[] for _ in range(num_experts)],
            }

        # Add current iteration data
        log_token_distribution_per_layer.iterations[layer_key]["iterations"].append(iteration)
        for expert_idx, fraction in enumerate(fractions):
            log_token_distribution_per_layer.iterations[layer_key]["expert_data"][expert_idx].append(fraction)

        # Create the line series plot
        xs = log_token_distribution_per_layer.iterations[layer_key]["iterations"]
        ys = log_token_distribution_per_layer.iterations[layer_key]["expert_data"]
        keys = [f"Expert {i}" for i in range(num_experts)]

        wandb_writer.log(
            {
                f"{layer_key}_combined": wandb_writer.plot.line_series(
                    xs=xs, ys=ys, keys=keys, title=f"Layer {layer} Expert Fractions", xname="Iteration"
                )
            },
            step=iteration,
        )


def save_aux_losses(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    avg_group: torch.distributed.ProcessGroup = None,
):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
        mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
    """
    if name not in moe_load_balance_loss_tracker:
        moe_load_balance_loss_tracker[name] = {}
        moe_load_balance_loss_tracker[name]["values"] = torch.zeros(num_layers, device=loss.device)
    moe_load_balance_loss_tracker[name]["values"][
        layer_number - 1
    ] += loss.detach()  # Aggregate the loss for the layer.
    moe_load_balance_loss_tracker[name]["reduce_group"] = reduce_group
    moe_load_balance_loss_tracker[name]["avg_group"] = avg_group


def clear_aux_losses_tracker():
    """Clear the auxiliary losses."""
    for name in moe_load_balance_loss_tracker:
        moe_load_balance_loss_tracker[name]["values"].zero_()
        moe_load_balance_loss_tracker[name]["reduce_group"] = None
        moe_load_balance_loss_tracker[name]["avg_group"] = None


def track_moe_metrics(
    loss_scale: float,
    iteration: int,
    num_layers,
    wandb_writer,
    per_layer_logging=False,
    # track_names: Optional[List[str]] = None,
):
    """Track the MoE metrics for logging.
    Args:
        loss_scale (float): The loss scale. number of microbatches.
        iteration (int): The iteration number.
        writer: The wandb writer.
        num_layers (int): The number of total layers.
        wandb_writer: The wandb writer.
        total_loss_dict: The total loss dictionary.
    """
    aux_losses = {k: v["values"].float() * loss_scale for k, v in moe_load_balance_loss_tracker.items()}
    num_moe_layers = num_layers

    for name, loss_list in aux_losses.items():
        # W&B logging lacks support for logging multiple scalars simultaneously.
        # As a workaround, we log each scalar individually first, then we can create
        # a custom panel to manually group them to a single plot.
        if wandb_writer:
            wandb_writer.log({f"{name}": loss_list.sum() / num_moe_layers}, iteration)
            if per_layer_logging:
                wandb_writer.log(
                    {f"moe/{name}_layer_{i}": loss for i, loss in enumerate(loss_list.tolist())},
                    iteration,
                )
            log_token_distribution_per_layer("moe", iteration, wandb_writer)

    clear_aux_losses_tracker()
    clear_token_per_expert_tracker()
