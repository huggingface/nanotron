import torch
import transformer_engine as te  # noqa

from nanotron.fp8.constants import FP8_GPU_NAMES
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta


def is_fp8_available() -> bool:
    """Check if FP8 is available on the current device."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
        return any(gpu_name in device_name for gpu_name in FP8_GPU_NAMES)
    else:
        return False


def get_tensor_fp8_metadata(tensor: torch.Tensor, dtype: DTypes) -> FP8Meta:
    from nanotron.fp8.constants import INITIAL_SCALING_FACTOR
    from nanotron.fp8.tensor import update_scaling_factor

    amax = tensor.abs().max().clone()
    assert amax.dtype == torch.float32

    scale = update_scaling_factor(amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), dtype)
    assert scale.dtype == torch.float32

    fp8_meta = FP8Meta(amax, scale, dtype)
    return fp8_meta


# TODO(xrsrke): shorter name
def is_overflow_underflow_nan(tensor: torch.Tensor) -> bool:
    overflow = torch.isinf(tensor).any().item()
    underflow = torch.isneginf(tensor).any().item()
    nan = torch.isnan(tensor).any().item()

    return True if (overflow or underflow or nan) else False
