import re
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


def get_exp_tp_pp_rank_and_size_from(
    world_rank: int, parallel_context: ParallelContext
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    result = parallel_context.get_local_ranks(world_rank=world_rank)
    return (
        (result[0], parallel_context.expert_pg.size()),
        (result[3], parallel_context.tp_pg.size()),
        (result[1], parallel_context.pp_pg.size()),
    )


def get_path(
    tensor_name: str,
    type: ObjectType,
    exp_tp_pp_rank_and_size: Tuple[Tuple[int, int], Tuple[int, int]],
    is_expert_sharded: bool,
    prefix: Optional[Path] = None,
) -> List[str]:
    suffix = tensor_name.split(".")
    suffix_path, suffix_name = suffix[:-1], suffix[-1]

    if exp_tp_pp_rank_and_size:
        # We always show pp_rank and tp_rank if `exp_tp_pp_rank_and_size` is provided
        (exp_rank, exp_size), (tp_rank, tp_size), (pp_rank, pp_size) = exp_tp_pp_rank_and_size
        if not is_expert_sharded or exp_size == 1:
            suffix_name = (
                f"{type.value}_{suffix_name}_pp-rank-{pp_rank}-of-{pp_size}_tp-rank-{tp_rank}-of-{tp_size}.safetensors"
            )
        else:
            # We only show exp_rank if tensor is exp_sharded and exp_size > 1
            suffix_name = f"{type.value}_{suffix_name}_pp-rank-{pp_rank}-of-{pp_size}_tp-rank-{tp_rank}-of-{tp_size}_exp-rank-{exp_rank}-of-{exp_size}.safetensors"
    else:
        suffix_name = f"{type.value}_{suffix_name}.safetensors"

    suffix_path.append(suffix_name)
    if prefix is None:
        return suffix_path
    else:
        return prefix.joinpath(*suffix_path)


def extract_tp_pp_rank_from_shard_path(shard_path: Path):
    pattern = r"pp-rank-(\d+)-of-\d+_tp-rank-(\d+)-of-\d+"
    match = re.search(pattern, str(shard_path))
    pp_rank, tp_rank = match.groups()
    return pp_rank, tp_rank


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
