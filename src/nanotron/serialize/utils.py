from enum import Enum
from typing import List, Optional, Tuple

from nanotron.core.process_groups import DistributedProcessGroups


class ObjectType(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


def get_tp_and_pp_rank_and_size_from(
    world_rank: int, dpg: DistributedProcessGroups
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    result = dpg.get_3d_ranks(world_rank=world_rank)
    return (result[2], dpg.tp_pg.size()), (result[0], dpg.pp_pg.size())


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
