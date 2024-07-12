import torch


def delete_tensor_from_memory(tensor: torch.Tensor):
    assert isinstance(tensor, torch.Tensor)
    del tensor
    torch.cuda.empty_cache()
