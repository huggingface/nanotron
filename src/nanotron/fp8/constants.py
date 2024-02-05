import torch

from nanotron.fp8.dtypes import DTypes

FP8_GPU_NAMES = ["h100", "rtx 4090"]

# FP8_DTYPES = [torch.fp8e4m3, torch.fp8e5m2]
# FP8E4M3_DTYPE = torch.fp8e4m3
# FP8E5M2_DTYPE = torch.fp8e5m2

FP8_DTYPES = [torch.int8, torch.uint8]
FP8E4M3_DTYPE = torch.int8
FP8E5M2_DTYPE = torch.uint8


DEFAULT_SCALE = 0.1
DEFAULT_FP8_FORMAT = torch.int8

DTYPE_TO_FP8_MAX = {DTypes.FP8E4M3: 448.0, DTypes.FP8E5M2: 57344.0, DTypes.KFLOAT16: 65504.0}
