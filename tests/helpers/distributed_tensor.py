import torch
from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup, get_global_rank


def assert_tensor_equal_over_group(tensor: torch.Tensor, group: ProcessGroup, assert_: bool = True) -> bool:
    """We assume that tensors are already of correct size."""
    reference_rank = 0
    if dist.get_rank(group) == reference_rank:
        reference_tensor = tensor
    else:
        reference_tensor = torch.empty_like(tensor)
    dist.broadcast(
        reference_tensor,
        src=get_global_rank(group=group, group_rank=reference_rank),
        group=group,
    )
    if assert_:
        torch.testing.assert_close(tensor, reference_tensor, atol=0, rtol=0)
    else:
        result = torch.allclose(tensor, reference_tensor, atol=0.0, rtol=0.0)
        results = [0] * group.size()
        dist.all_gather_object(results, result, group)
        return all(results)
