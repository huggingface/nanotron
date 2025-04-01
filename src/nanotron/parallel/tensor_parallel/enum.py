from enum import Enum, auto


class TensorParallelLinearMode(Enum):
    ALL_REDUCE = auto()
    REDUCE_SCATTER = auto()

    def __format__(self, format_spec):
        return self.name

    def __str__(self):
        return self.name
