import torch
from einops import rearrange

# @torch.no_grad()
# def _get_dispatched_routing_indices_ver4(global_routing_indices, expert_parallel_size, num_experts):
#     # generated https://poe.com/chat/id7k0jlziw87ng97zd
#     num_local_experts = num_experts // expert_parallel_size

#     # Compute the shape directly instead of using einops
#     num_tokens = global_routing_indices.size(0) // expert_parallel_size

#     # Reshape without using einops (which may introduce extra operations)
#     global_routing_indices_per_device = global_routing_indices.reshape(
#         expert_parallel_size, num_tokens, -1
#     ).view(expert_parallel_size, -1)

#     # Sort the indices
#     sorted_indices, _ = torch.sort(global_routing_indices_per_device)

#     # Pre-allocate tensors for each expert parallel rank
#     dispatched_indices = []

#     # Create ranges for each expert
#     expert_ranges = torch.arange(num_experts, device=global_routing_indices.device)
#     expert_ranges = expert_ranges.reshape(expert_parallel_size, num_local_experts)

#     for ep_rank in range(expert_parallel_size):
#         # Get the range for this rank
#         start_idx = ep_rank * num_local_experts
#         end_idx = start_idx + num_local_experts

#         # Create a mask for indices in this range
#         # Instead of boolean indexing, we'll use torch.masked_select which is more CUDA-friendly
#         mask = (sorted_indices[ep_rank] >= start_idx) & (sorted_indices[ep_rank] < end_idx)
#         selected_indices = torch.masked_select(sorted_indices[ep_rank], mask)
#         dispatched_indices.append(selected_indices)

#     return dispatched_indices


@torch.no_grad()
def _get_dispatched_routing_indices(global_routing_indices, expert_parallel_size, num_experts):
    num_local_experts = num_experts // expert_parallel_size
    global_routing_indices_per_device = rearrange(
        global_routing_indices,
        "(expert_parallel_size num_tokens) num_experts -> expert_parallel_size (num_tokens num_experts)",
        expert_parallel_size=expert_parallel_size,
    )
    sorted_global_routing_indices_per_device, _ = torch.sort(global_routing_indices_per_device)

    dispatched_indices = []
    for ep_rank in range(expert_parallel_size):
        start_idx = ep_rank * num_local_experts
        end_idx = start_idx + num_local_experts

        mask = (sorted_global_routing_indices_per_device >= start_idx) & (
            sorted_global_routing_indices_per_device < end_idx
        )
        dispatched_indices.append(sorted_global_routing_indices_per_device[mask])

    return dispatched_indices


def compiled_get_dispatched_routing_indices(global_routing_indices, num_ranks, num_experts_per_rank):
    # NOTE: use torch.compile to compile the function
    return torch.compile(_get_dispatched_routing_indices)(global_routing_indices, num_ranks, num_experts_per_rank)
