import dataclasses
from typing import Dict, MutableMapping, Optional, Tuple

import numpy as np
import torch

from brrr.core import distributed as dist


@dataclasses.dataclass
class RandomState:
    random: Tuple[int, Tuple[int, ...], None]
    numpy: Tuple[str, np.array, int, int, float]
    torch_cpu: torch.Tensor
    torch_cuda: Optional[torch.Tensor]

    def __eq__(self, other):
        return (
            isinstance(other, RandomState)
            and all(v1 == v2 for v1, v2 in zip(self.random, other.random))
            and all(
                np.array_equal(v1, v2) if isinstance(v1, np.ndarray) else v1 == v2
                for v1, v2 in zip(self.numpy, other.numpy)
            )
            and torch.equal(self.torch_cpu, other.torch_cpu)
            and (
                other.torch_cuda is None if self.torch_cuda is None else torch.equal(self.torch_cuda, other.torch_cuda)
            )
        )


class RandomStates(MutableMapping[str, RandomState]):
    def __init__(self, dict: dict):
        for key, value in dict.items():
            self.check_type(key, value)
        # TODO @thomasw21: We make a copy for safety measure.
        self._dict = dict.copy()

    @staticmethod
    def check_type(key, value):
        if not isinstance(key, str):
            raise ValueError(f"Expected key to be of type str. Got {type(key)}")
        if not isinstance(value, RandomState):
            raise ValueError(f"Expected value to be of type `brrr.dataclass.RandomState`. Got {type(value)}")

    def __getitem__(self, item):
        return self._dict[item]

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)

    def __delitem__(self, key):
        raise ValueError("Can't delete a random states key")

    def __setitem__(self, key, value):
        if key not in self._dict:
            raise ValueError("Can't add a new random states after initialisation")
        self.check_type(key, value)
        return self._dict.__setitem__(key, value)

    def __eq__(self, other):
        if not isinstance(other, RandomStates):
            return False

        return self._dict == other._dict


@dataclasses.dataclass
class DistributedProcessGroups:
    # Default process group, all the ranks are in the same process group
    world_pg: dist.ProcessGroup
    # Convention, dimensions are [pp,dp,tp] (with values equal to 1 when no parallelism)
    world_rank_matrix: np.array

    # process dependent process groups
    tp_pg: dist.ProcessGroup
    dp_pg: dist.ProcessGroup
    pp_pg: dist.ProcessGroup

    # Mapping from sorted list of world ranks to process group
    world_ranks_to_pg: Dict[Tuple[int, ...], dist.ProcessGroup] = dataclasses.field(default_factory=dict)

    def get_3d_ranks(self, world_rank: int) -> Tuple[int, int, int]:
        pp_rank = (world_rank // (self.tp_pg.size() * self.dp_pg.size())) % self.pp_pg.size()
        dp_rank = (world_rank // self.tp_pg.size()) % self.dp_pg.size()
        tp_rank = world_rank % self.tp_pg.size()
        return (pp_rank, dp_rank, tp_rank)
