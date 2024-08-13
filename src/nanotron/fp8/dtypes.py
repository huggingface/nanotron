from enum import Enum


# TODO(xrsrke): don't use plural
# TODO(xrsrke): change to QDType, so we don't mistaken it with the torch dtype
# QDType = Quantization DType
class DTypes(Enum):
    FP8E4M3 = "FP8E4M3"
    FP8E5M2 = "FP8E5M2"
    KFLOAT16 = "KFLOAT16"
    KFLOAT32 = "KFLOAT32"
    KBFLOAT16 = "KBFLOAT16"
