from enum import Enum
from typing import List, Optional, Tuple

from nanotron.distributed import ParallelContext, ParallelMode


class ObjectType(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


def get_tp_and_pp_rank_and_size_from(
    world_rank: int, parallel_context: ParallelContext
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    result = parallel_context.get_3d_ranks(local_rank=world_rank, parallel_mode=ParallelMode.GLOBAL)
    tp_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    pp_world_size = parallel_context.get_world_size(ParallelMode.PIPELINE)

    return (result[2], tp_world_size), (result[0], pp_world_size)


def get_path(
    tensor_name: str,
    type: ObjectType,
    # Return rank and size
    # TODO @thomasw21: make a topology agnostic system
    tp_and_pp_rank_and_size: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> List[str]:
    suffix = tensor_name.split(".")
    suffix_path, suffix_name = suffix[:-1], suffix[-1]

    if tp_and_pp_rank_and_size:
        (tp_rank, tp_size), (pp_rank, pp_size) = tp_and_pp_rank_and_size
        suffix_name = (
            f"{type.value}_{suffix_name}_pp-rank-{pp_rank}-of-{pp_size}_tp-rank-{tp_rank}-of-{tp_size}.safetensors"
        )
    else:
        suffix_name = f"{type.value}_{suffix_name}.safetensors"

    suffix_path.append(suffix_name)
    return suffix_path
