from copy import deepcopy
from typing import cast

import pytest
import torch
import transformer_engine as te  # noqa
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@pytest.mark.parametrize("interval", [1, 5, 10])
@pytest.mark.parametrize("is_quantized", [True, False])
def test_fp8_tensor_track_amaxs(dtype, interval, is_quantized):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval)
    fp8_meta = cast(FP8Meta, fp8_tensor.fp8_meta)

    assert isinstance(fp8_meta.amaxs, list)
    assert len(fp8_meta.amaxs) == 1
    assert torch.equal(fp8_meta.amaxs[0], tensor.abs().max())

    for i in range(2, (interval * 2) + 1):
        new_data = torch.randn(fp8_tensor.shape, dtype=torch.float32, device="cuda")
        new_data = FP8Tensor(new_data, dtype=dtype) if is_quantized else new_data
        fp8_tensor.set_data(new_data)

        # NOTE: we expect it only maintains amaxs in
        # a window of interval, not more than that
        assert len(fp8_meta.amaxs) == i if i < interval else interval

        expected_amax = new_data.fp8_meta.amax if is_quantized else new_data.abs().max()
        assert torch.equal(fp8_meta.amaxs[-1], expected_amax)


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@pytest.mark.parametrize("interval", [1, 5, 10])
@pytest.mark.parametrize("is_dynamic_scaling", [True, False])
def test_immediately_rescale_if_encounter_overflow_underflow(dtype, interval, is_dynamic_scaling):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval, is_dynamic_scaling=is_dynamic_scaling)
    fp8_meta = cast(FP8Meta, fp8_tensor.fp8_meta)
    new_data = deepcopy(tensor)
    history_sf = []
    has_overflow = False
    history_sf.append(fp8_meta.scale)

    for i in range(2, (interval * 1) + 1):
        new_data = new_data.clone() * (i * 2)

        temp_data = deepcopy(new_data)
        if interval - 1 == i:
            has_overflow = True
            temp_data[0] = torch.tensor(float("inf"))

        fp8_tensor.set_data(temp_data)

        if is_dynamic_scaling:
            if has_overflow is True:
                # NOTE: this is right after or after overflow happens
                if interval - 1 == i:
                    assert (
                        fp8_meta.scale not in history_sf
                    ), f"i: {i}, interval: {interval}, fp8_meta.scale: {fp8_meta.scale}, history_sf: {history_sf}"
                else:
                    # NOTE: because we reset the interval after overflow, so we expect
                    # the next iteration will use the same scaling value
                    assert (
                        fp8_meta.scale in history_sf
                    ), f"i: {i}, interval: {interval}, fp8_meta.scale: {fp8_meta.scale}, history_sf: {history_sf}"
            else:
                # NOTE: we expect it to use the same scaling value
                # until it reaches the interval or overflow
                assert (
                    fp8_meta.scale in history_sf
                ), f"i: {i}, interval: {interval}, fp8_meta.scale: {fp8_meta.scale}, history_sf: {history_sf}"
        else:
            if has_overflow is True:
                if interval - 1 == i:
                    # NOTE: because if overflow, we use the amaxs from earlier iterations
                    # so we should get the same scaling value in one of the earlier iterations
                    assert (
                        fp8_meta.scale in history_sf
                    ), f"i: {i}, interval: {interval}, fp8_meta.scale: {fp8_meta.scale}, history_sf: {history_sf}"
                else:
                    assert (
                        fp8_meta.scale not in history_sf
                    ), f"i: {i}, interval: {interval}, fp8_meta.scale: {fp8_meta.scale}, history_sf: {history_sf}"
            else:
                # NOTE: if it's not delayed scaling, then it doesn't matter
                # whether it's overflow or not, we should get new scaling
                assert (
                    fp8_meta.scale not in history_sf
                ), f"i: {i}, interval: {interval}, fp8_meta.scale: {fp8_meta.scale}, history_sf: {history_sf}"

        history_sf.append(fp8_meta.scale)


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@pytest.mark.parametrize("interval", [1, 5, 10])
@pytest.mark.parametrize("is_dynamic_scaling", [True, False])
def test_delay_scaling_fp8_tensor(dtype, interval, is_dynamic_scaling):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval, is_dynamic_scaling=is_dynamic_scaling)
    fp8_meta = cast(FP8Meta, fp8_tensor.fp8_meta)

    new_data = deepcopy(tensor)
    history_sf = []
    history_sf.append(fp8_meta.scale)

    for i in range(2, (interval * 1) + 1):
        new_data = new_data.clone() * 2
        fp8_tensor.set_data(new_data)

        if is_dynamic_scaling:
            is_new_interval = i % interval == 0
            assert fp8_meta.is_ready_to_scale == (i - 1 % interval == 0)
            assert (fp8_meta.scale not in history_sf) is is_new_interval
        else:
            assert fp8_meta.is_ready_to_scale is True
            assert fp8_meta.scale not in history_sf

        history_sf.append(fp8_meta.scale)


# @pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
# @pytest.mark.parametrize("interval", [1, 5, 10])
# @pytest.mark.parametrize("is_dynamic_scaling", [True, False])
# def test_fp8_dynamic_quantization(dtype, interval, is_dynamic_scaling):
#     tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")

#     fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval, is_dynamic_scaling=is_dynamic_scaling)
#     fp8_meta = cast(FP8Meta, fp8_tensor.fp8_meta)

#     history_sf = []
#     history_sf.append(fp8_meta.scale)
#     new_data = deepcopy(tensor)

#     for i in range(2, interval + 1):
#         new_data = new_data.clone() * 2
#         fp8_tensor.set_data(new_data)

#         if is_dynamic_scaling:
#             is_new_interval = i % interval == 0

#             if is_new_interval:
#                 assert fp8_meta.scale not in history_sf
#             else:
#                 assert fp8_meta.scale in history_sf
#         else:
#             # NOTE: if it's not dynamic quantization, then we should get new scaling
#             # value for every new data changes
#             assert fp8_meta.scale not in history_sf

#         history_sf.append(fp8_meta.scale)


# # TODO(xrsrke): handling overflow before warmup
# @pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
# @pytest.mark.parametrize("interval", [5, 10])
# @pytest.mark.parametrize("is_overflow_after_warmup", [True])
# def test_fp8_dynamic_quantization_under_overflow(dtype, interval, is_overflow_after_warmup):
#     tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
#     new_data = deepcopy(tensor)
#     fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval, is_dynamic_scaling=True)
#     fp8_meta = cast(FP8Meta, fp8_tensor.fp8_meta)

#     past_steps = interval if is_overflow_after_warmup else 0

#     for _ in range(past_steps):
#         new_data = new_data.clone() * 2
#         fp8_tensor.set_data(new_data)

#     current_scale = deepcopy(fp8_meta.scale)
#     new_data = new_data.clone() * 2
#     new_data[0] = torch.tensor(float("inf"))
#     fp8_tensor.set_data(new_data)

#     assert not torch.equal(fp8_meta.scale, current_scale)
