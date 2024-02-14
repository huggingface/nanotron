from enum import Enum, auto


class DTypes(Enum):
    FP8E4M3 = auto()
    FP8E5M2 = auto()
    
    # TODO(xrsrke): move KFLOAT16 out of DTypes
    KFLOAT16 = auto()
