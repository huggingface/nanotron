import torch
import torch.distributed as dist

from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_all_gather


def switch_aux_loss(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    aux_loss_coeff: float,
    top_k: int,
    sequence_partition_group: dist.ProcessGroup = None,
):
    """
    Compute the auxiliary loss for the MoE layer from the Switch Transformer paper.
    Ref: https://arxiv.org/abs/2101.03961
    Args:
        probs: softmaxed probabilities of the experts [num_tokens, num_experts]
        tokens_per_expert: number of tokens per expert [num_experts]
        sequence_partition_group: context parallel group + sequence parallel group
    """
    num_sub_sequences = 1

    # gather the whole sequence when using Context Parallelism or Sequence Parallelism
    if sequence_partition_group is not None:
        num_sub_sequences = dist.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    total_num_tokens = probs.shape[0] * num_sub_sequences
    num_experts = probs.shape[1]
    aggregated_probs_per_expert = probs.sum(dim=0)

    # aggregated_probs_per_expert: P * total_num_tokens in paper, shape:[num_experts]
    # tokens_per_expert:           F * total_num_tokens * top_k in paper, shape:[num_experts]
    # formula: a * e * dot(P, F)
    aux_loss = (
        aux_loss_coeff
        * num_experts
        * torch.sum(aggregated_probs_per_expert * tokens_per_expert)
        / (total_num_tokens**2 * top_k)
    )

    return aux_loss


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.

    Args:
        logits (torch.Tensor): The logits before router after gating.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss


def sequence_wise_aux_loss(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    batch_size: int,
    seq_length: int,
    top_k: int,
    aux_loss_coeff: float,
    sequence_partition_group: dist.ProcessGroup = None,
):
    """
    Compute the Sequence-Wise Auxiliary Loss. from the DeepSeek-V3 paper.
    Ref: https://arxiv.org/html/2412.19437v1
    Args:
        probs: softmaxed probabilities of the experts [num_tokens, num_experts]
        routing_map: routing map of the experts [num_tokens, num_experts]
        sequence_partition_group: context parallel group + sequence parallel group
    """
    num_sub_sequence = 1
    num_experts = probs.shape[1]

    probs_for_aux_loss = probs.view(seq_length, batch_size, -1)
    routing_map = routing_map.view(seq_length, batch_size, -1)

    # gather the whole sequence when using Context Parallelism or Sequence Parallelism
    if sequence_partition_group is not None:
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        seq_length *= num_sub_sequence
        probs_for_aux_loss = differentiable_all_gather(probs_for_aux_loss, group=sequence_partition_group)

    cost_coeff = routing_map.sum(dim=0, dtype=torch.float).div_(seq_length * top_k / num_experts)
    seq_aux_loss = (cost_coeff * probs_for_aux_loss.mean(dim=0)).sum(dim=1).mean()
    seq_aux_loss *= aux_loss_coeff

    return seq_aux_loss


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    # used to scale the gradient when using PP
    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=aux_loss.device)
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            MoEAuxLossAutoScaler.main_loss_backward_scale.copy_(scale)
