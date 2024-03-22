import torch
import pytest
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.dtypes import DTypes
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2, DTypes.KFLOAT16])
def test_fp8_meta(dtype):
    AMAX = torch.randn(1, dtype=torch.float32) * 3
    SCALE = torch.randn(1, dtype=torch.float32)
    
    fp8_meta = FP8Meta(amax=AMAX, scale=SCALE, dtype=dtype)

    assert torch.equal(fp8_meta.amax, AMAX)
    assert torch.equal(fp8_meta.scale, SCALE)
    assert torch.equal(fp8_meta.inverse_scale, 1/fp8_meta.scale)
    assert isinstance(fp8_meta.fp8_max, float)
    assert isinstance(fp8_meta.te_dtype, tex.DType)
