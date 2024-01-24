from enum import Enum, auto

import torch

FP8_GPU_NAMES = ["h100", "rtx 4090"]

# FP8_DTYPES = [torch.fp8e4m3, torch.fp8e5m2]
# FP8E4M3_DTYPE = torch.fp8e4m3
# FP8E5M2_DTYPE = torch.fp8e5m2

FP8_DTYPES = [torch.int8, torch.uint8]
FP8E4M3_DTYPE = torch.int8
FP8E5M2_DTYPE = torch.uint8


DEFAULT_SCALE = 0.1
DEFAULT_FP8_FORMAT = torch.int8


class DTypes(Enum):
    FP8E4M3 = auto()
    FP8E5M2 = auto()
    kfloat16 = auto()


DTYPE_TO_FP8_MAX = {DTypes.FP8E4M3: 448.0, DTypes.FP8E5M2: 57344.0, DTypes.kfloat16: 65504.0}


FP8_LINEAR_META_MAPPING = {
    "input": {"dtype": DTypes.FP8E4M3},
    "weight": {"dtype": "float16", "fp8_max": 65504.0},
    "weight_gradient": {"dtype": DTypes.FP8E4M3, "fp8_max": 448.0},
    "output_gradient": {"dtype": DTypes.FP8E5M2, "fp8_amax": 57344.0},
}
