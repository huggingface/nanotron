from enum import Enum, auto


# TODO @thomasw21: python 3.11 introduces `StrEnum` which would've been great to use.
class TensorParallelLinearMode(Enum):
    ALL_REDUCE = auto()
    REDUCE_SCATTER = auto()

    def __format__(self, format_spec):
        return self.name

    def __str__(self):
        return self.name
