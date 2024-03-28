import pytest
import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2, DTypes.KFLOAT16])
def test_fp8_meta(dtype):
    AMAX = torch.randn(1, dtype=torch.float32) * 3
    SCALE = torch.randn(1, dtype=torch.float32)
    INTERVAL = 5
    is_dynamic_scaling = True

    fp8_meta = FP8Meta(amax=AMAX, scale=SCALE, dtype=dtype, interval=INTERVAL, is_dynamic_scaling=is_dynamic_scaling)

    assert torch.equal(fp8_meta.amax, AMAX)
    assert torch.equal(fp8_meta.scale, SCALE)
    assert torch.equal(fp8_meta.inverse_scale, 1 / fp8_meta.scale)
    assert fp8_meta.is_dynamic_scaling == is_dynamic_scaling
    assert isinstance(fp8_meta.fp8_max, float)
    assert isinstance(fp8_meta.te_dtype, tex.DType)


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2, DTypes.KFLOAT16])
@pytest.mark.parametrize(
    "modifications",
    [
        # NOTE: no modifications
        {},
        {"amax": torch.randn(1, dtype=torch.float32)},
        {"scale": torch.randn(1, dtype=torch.float32)},
        {"dtype": True},
        {"is_dynamic_scaling": True},
    ],
)
def test_fp8_meta_equality(dtype, modifications):
    def modify_fp8_meta(fp8_meta, modifications):
        if "dtype" in modifications:
            modifications["dtype"] = next(d for d in [DTypes.FP8E5M2, DTypes.FP8E4M3, DTypes.KFLOAT16] if d != dtype)

        for attr, new_value in modifications.items():
            setattr(fp8_meta, attr, new_value)

    AMAX = torch.randn(1, dtype=torch.float32) * 3
    SCALE = torch.randn(1, dtype=torch.float32)
    INTERVAL = 5

    fp8_meta = FP8Meta(amax=AMAX, scale=SCALE, dtype=dtype, interval=INTERVAL)
    ref_fp8_meta = FP8Meta(amax=AMAX, scale=SCALE, dtype=dtype, interval=INTERVAL)

    modify_fp8_meta(ref_fp8_meta, modifications)

    if not modifications:
        assert fp8_meta == ref_fp8_meta
    else:
        assert fp8_meta != ref_fp8_meta
