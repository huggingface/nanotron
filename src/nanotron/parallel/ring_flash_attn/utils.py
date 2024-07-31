from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = ["update_out_and_lse", "RingComm"]

# reference: https://github.com/zhuzilin/ring-flash-attention
@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty((num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


def extract_local(tensor: torch.Tensor, rank: int, world_size: int, dim: int = 1) -> torch.Tensor:
    value_chunks = tensor.chunk(2 * world_size, dim=dim)
    local_value = torch.cat([value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim)
    return local_value.contiguous()


def zigzag_split(rank: int, world_size: int, *args: torch.Tensor, dim=1) -> Tuple[torch.Tensor, ...]:
    local_values = []
    for tensor in args:
        local_value = extract_local(tensor, rank, world_size, dim)
        local_values.append(local_value)
    return tuple(local_values)


def normal_split(rank: int, world_size: int, *args: torch.Tensor, dim=1) -> Tuple[torch.Tensor, ...]:
    local_values = []
    for value in args:
        local_value = value.chunk(world_size, dim=dim)[rank]
        local_values.append(local_value)
    return tuple(local_values)


## a function to merge the tensor in a zigzag way. inverse of zigzag_split
def zigzag_merge(
    rank: int, world_size: int, local_tensor: torch.Tensor, process_group: dist.ProcessGroup, dim: int = 1
) -> Tuple[torch.Tensor, ...]:

    # Split the local tensor into two chunks
    chunk1, chunk2 = local_tensor.chunk(2, dim=dim)

    # Create placeholders for all chunks
    all_chunks = [torch.zeros_like(chunk1) for _ in range(2 * world_size)]

    # Gather all chunks from all processes
    dist.all_gather(all_chunks[:world_size], chunk1, group=process_group)
    dist.all_gather(all_chunks[world_size:], chunk2, group=process_group)

    # Reverse the order of the second half of chunks
    all_chunks[world_size:] = all_chunks[world_size:][::-1]

    del chunk1, chunk2
    torch.cuda.empty_cache()

    # Concatenate all chunks to form the full tensor
    merged_tensor = torch.cat(all_chunks, dim=dim)

    del all_chunks
    torch.cuda.empty_cache()

    return merged_tensor
