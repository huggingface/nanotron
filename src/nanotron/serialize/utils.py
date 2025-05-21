import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import SlicesPair
from nanotron.serialize.metadata import TensorMetadata


class ObjectType(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


@dataclass
class CheckpointParallelRanks:
    """
    ep_rank is optional because it's only applicable to moe params

    NOTE: because a non-moe has no ep_rank, we need to pass it as an optional
    """

    pp_rank: int
    pp_world_size: int

    tp_rank: int
    tp_world_size: int

    ep_rank: Optional[int] = None
    ep_world_size: Optional[int] = None


def get_exp_tp_pp_rank_and_size_from(
    world_rank: int, parallel_context: ParallelContext
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    result = parallel_context.get_local_ranks(world_rank=world_rank)

    if parallel_context.enabled_moe is True:
        return CheckpointParallelRanks(
            pp_rank=result["pp"],
            pp_world_size=parallel_context.pp_pg.size(),
            tp_rank=result["tp"],
            tp_world_size=parallel_context.tp_pg.size(),
            ep_rank=result["ep"],
            ep_world_size=parallel_context.ep_pg.size(),
        )
    else:
        return CheckpointParallelRanks(
            pp_rank=result["pp"],
            pp_world_size=parallel_context.pp_pg.size(),
            tp_rank=result["tp"],
            tp_world_size=parallel_context.tp_pg.size(),
        )


def get_path(
    tensor_name: str,
    type: ObjectType,
    checkpoint_parallel_ranks: CheckpointParallelRanks,
    is_expert_sharded: bool,
    prefix: Optional[Path] = None,
) -> List[str]:
    def get_checkpoint_name_from_parallel_ranks(checkpoint_parallel_ranks: CheckpointParallelRanks, is_moe: bool):
        pp_rank = checkpoint_parallel_ranks.pp_rank
        pp_size = checkpoint_parallel_ranks.pp_world_size

        tp_rank = checkpoint_parallel_ranks.tp_rank
        tp_size = checkpoint_parallel_ranks.tp_world_size

        ep_rank = checkpoint_parallel_ranks.ep_rank
        ep_size = checkpoint_parallel_ranks.ep_world_size

        if not is_moe:
            return f"pp-rank-{pp_rank}-of-{pp_size}_tp-rank-{tp_rank}-of-{tp_size}"
        else:
            return f"pp-rank-{pp_rank}-of-{pp_size}_tp-rank-{tp_rank}-of-{tp_size}_exp-rank-{ep_rank}-of-{ep_size}"

    suffix = tensor_name.split(".")
    suffix_path, suffix_name = suffix[:-1], suffix[-1]

    if checkpoint_parallel_ranks:
        # We always show pp_rank and tp_rank if `exp_tp_pp_rank_and_size` is provided
        suffix_name = f"{type.value}_{suffix_name}_{get_checkpoint_name_from_parallel_ranks(checkpoint_parallel_ranks, is_moe=is_expert_sharded)}.safetensors"
    else:
        # NOTE: for params that aren't sharded, we don't need to add any parallel ranks
        suffix_name = f"{type.value}_{suffix_name}.safetensors"

    suffix_path.append(suffix_name)
    if prefix is None:
        return suffix_path
    else:
        return prefix.joinpath(*suffix_path)


def extract_tp_pp_rank_from_shard_path(shard_path: Path):
    from nanotron.serialize.utils import CheckpointParallelRanks

    pattern = r"pp-rank-(\d+)-of-(\d+)_tp-rank-(\d+)-of-(\d+)"
    match = re.search(pattern, str(shard_path))
    pp_rank, pp_size, tp_rank, tp_size = match.groups()

    return CheckpointParallelRanks(
        pp_rank=int(pp_rank), pp_world_size=int(pp_size), tp_rank=int(tp_rank), tp_world_size=int(tp_size)
    )


def merge_and_shard_tp_tensors(
    buffer: torch.Tensor,
    unsharded_buffer: torch.Tensor,
    shards_and_slices_maps: List[Tuple[torch.Tensor, Tuple[SlicesPair, ...]]],
    shard_metadata: TensorMetadata,
) -> torch.Tensor:
    for shard, slices_pairs in shards_and_slices_maps:
        for slices_pair in slices_pairs:
            local_slices = slices_pair.local_slices
            global_slices = slices_pair.global_slices
            unsharded_buffer[global_slices] = shard[local_slices]

    for slices_pair in shard_metadata.local_global_slices_pairs:
        local_slices = slices_pair.local_slices
        global_slices = slices_pair.global_slices
        buffer[local_slices] = unsharded_buffer[global_slices]

    return buffer


def merge_and_shard_ep_tensors():
    # ep_rank = dist.get_rank(parallel_context.ep_pg)
    # _data = fi.get_tensor("data")
    # _num_experts = _data.shape[0]
    # _num_local_experts = _num_experts // parallel_context.ep_pg.size()
    # _start_idx = ep_rank * _num_local_experts
    # _end_idx = _start_idx + _num_local_experts
    # param_or_buffer[:] = _data[_start_idx:_end_idx, :, :]
    pass
