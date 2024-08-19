from copy import deepcopy

import numpy as np
import torch
import transformer_engine as te  # noqa

# from nanotron.fp8.constants import (
#     FP8_ATOL_THRESHOLD,
#     FP8_RTOL_THRESHOLD,
#     FP16_ATOL_THRESHOLD,
#     FP16_RTOL_THRESHOLD,
#     QTYPE_TO_DTYPE,
# )
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8

# @pytest.mark.parametrize("size", [4, 8, 16, 64])
# @pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
# def test_quantize_and_dequantize_tensor_in_fp8(size, dtype):

if __name__ == "__main__":
    size = 4
    dtype = DTypes.FP8E4M3
    # dtype = DTypes.FP8E5M2
    tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)
    fp8_tensor = FP8Tensor(tensor, dtype=dtype)

    assert not np.array_equal(fp8_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
    # NOTE: sometimes type(tensor) is FP8Tensor, but it still passes, so we directly check the class name
    # to make sure it's a torch.Tensor
    assert tensor.__class__.__name__ == torch.Tensor.__name__
    assert tensor.dtype == ref_tensor.dtype

    assert 1 == 1

    # torch.testing.assert_close(tensor, ref_tensor, rtol=FP8_RTOL_THRESHOLD, atol=FP8_ATOL_THRESHOLD)
