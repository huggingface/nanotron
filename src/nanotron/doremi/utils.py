import torch


@torch.jit.script
def masked_mean(loss: torch.Tensor, label_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # type: (Tensor, Tensor, torch.dtype) -> Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()
