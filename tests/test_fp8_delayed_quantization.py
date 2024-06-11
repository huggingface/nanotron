from copy import deepcopy
from typing import cast

import pytest
import torch
import transformer_engine as te  # noqa
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import FP8TensorParallelColumnLinear


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
def test_immediately_rescale_if_encounter_overflow_underflow(dtype, interval):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval)
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

        if interval > 1:
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
def test_delay_scaling_fp8_tensor(dtype, interval):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval)
    fp8_meta = cast(FP8Meta, fp8_tensor.fp8_meta)

    new_data = deepcopy(tensor)
    history_sf = []
    history_sf.append(fp8_meta.scale)

    for i in range(2, (interval * 1) + 1):
        new_data = new_data.clone() * 2
        fp8_tensor.set_data(new_data)

        if interval > 1:
            is_new_interval = i % interval == 0
            assert fp8_meta.is_ready_to_scale == (i - 1 % interval == 0)
            assert (fp8_meta.scale not in history_sf) is is_new_interval
        else:
            assert fp8_meta.is_ready_to_scale is True
            assert fp8_meta.scale not in history_sf

        history_sf.append(fp8_meta.scale)


# @pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
# @pytest.mark.parametrize("interval", [1, 5, 10])
# @pytest.mark.parametrize("is_delayed_scaling", [True, False])
# def test_fp8_dynamic_quantization(dtype, interval, is_delayed_scaling):
#     tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")

#     fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval, is_delayed_scaling=is_delayed_scaling)
#     fp8_meta = cast(FP8Meta, fp8_tensor.fp8_meta)

#     history_sf = []
#     history_sf.append(fp8_meta.scale)
#     new_data = deepcopy(tensor)

#     for i in range(2, interval + 1):
#         new_data = new_data.clone() * 2
#         fp8_tensor.set_data(new_data)

#         if is_delayed_scaling:
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
#     fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=interval, is_delayed_scaling=True)
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


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@pytest.mark.parametrize("total_steps", [1, 5, 20, 25])
@rerun_if_address_is_in_use()
def test_delayed_quantization_for_fp8_linear(tp: int, dp: int, pp: int, total_steps: int):
    IN_FEATURES = 32
    OUT_FEATURES_PER_RANK = 16
    LR = 0.001
    BATCH_SIZE = 16

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_delayed_quantization_for_fp8_linear)(
        in_features=IN_FEATURES,
        out_features_per_rank=OUT_FEATURES_PER_RANK,
        lr=LR,
        batch_size=BATCH_SIZE,
        total_steps=total_steps,
    )


def _test_delayed_quantization_for_fp8_linear(
    parallel_context: ParallelContext,
    in_features: int,
    out_features_per_rank: int,
    lr: int,
    batch_size: int,
    total_steps: int,
):
    def count_unique_values(xs):
        # NOTE: sometimes, we compute a new scaling value
        # but the value could be the same as the previous iterations
        # but the memory address is different
        return len({x.data_ptr() for x in xs})

    out_features = parallel_context.tp_pg.size() * out_features_per_rank

    linear = FP8TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
    )
    optim = FP8Adam(linear.parameters(), lr=lr)

    # NOTE: this take the assumption of the fp8 recipe
    # at the time writing this test
    input_scales = []
    weight_scales = []
    input_grad_scales = []
    weight_grad_scales = []
    # output_grad_scales = []

    # input_grad_ready_to_scales = []
    # weight_ready_to_scales = []

    for i in range(total_steps):
        inputs = torch.randn(batch_size, in_features, device="cuda")
        optim.zero_grad()
        outputs = linear(inputs).sum()
        outputs.backward()
        optim.step()

        linear.weight = cast(FP8Parameter, linear.weight)

        input_scales.append(linear.metadatas.input.scale)
        weight_scales.append(linear.weight.fp8_meta.scale)
        input_grad_scales.append(linear.metadatas.input_grad.scale)
        weight_grad_scales.append(linear.metadatas.weight_grad.scale)

        # input_grad_ready_to_scales.append(linear.metadatas.input_grad.is_ready_to_scale)
        # weight_ready_to_scales.append(linear.metadatas.weight_grad.is_ready_to_scale)

    # NOTE: we expect it computes a new scaling value only if it reaches the interval
    # NOTE: plus 1 is taking into account the initial scaling value
    # assert count_unique_values(input_scales) == total_steps // linear.metadatas.input.interval + 1
    assert count_unique_values(input_scales) == total_steps // linear.metadatas.input.interval
    assert count_unique_values(weight_scales) == total_steps // linear.weight.fp8_meta.interval
    # NOTE: input grad's interval is 16, so the first step is a new scaling value,
    # then 16th step is a new scaling value => n / 16 + 1
    # assert count_unique_values(input_grad_scales) == total_steps // linear.metadatas.input_grad.interval + 1
    assert count_unique_values(input_grad_scales) == total_steps // linear.metadatas.input_grad.interval
    assert count_unique_values(weight_grad_scales) == total_steps // linear.metadatas.weight_grad.interval

    parallel_context.destroy()


# def _test_delayed_quantization_for_fp8_linear(total_steps):
#     def count_unique_values(xs):
#         # NOTE: sometimes, we compute a new scaling value
#         # but the value could be the same as the previous iterations
#         # but the memory address is different
#         return len({x.data_ptr() for x in xs})

#     torch.randn(16, 16, device="cuda")
#     linear = nn.Linear(16, 16, device="cuda")
#     linear = convert_linear_to_fp8(linear, accum_qtype=DTypes.KFLOAT16)
#     optim = FP8Adam(linear.parameters(), lr=0.01)

#     # NOTE: this take the assumption of the fp8 recipe
#     # at the time writing this test
#     input_scales = []
#     weight_scales = []
#     input_grad_scales = []
#     weight_grad_scales = []
#     # output_grad_scales = []

#     # input_grad_ready_to_scales = []
#     # weight_ready_to_scales = []

#     for i in range(total_steps):
#         inputs = torch.randn(16, 16, device="cuda", requires_grad=True)
#         optim.zero_grad()
#         linear(inputs).sum().backward()
#         optim.step()

#         linear.weight = cast(FP8Parameter, linear.weight)

#         input_scales.append(linear.metadatas.input.scale)
#         weight_scales.append(linear.weight.fp8_meta.scale)
#         input_grad_scales.append(linear.metadatas.input_grad.scale)
#         weight_grad_scales.append(linear.metadatas.weight_grad.scale)

#         # input_grad_ready_to_scales.append(linear.metadatas.input_grad.is_ready_to_scale)
#         # weight_ready_to_scales.append(linear.metadatas.weight_grad.is_ready_to_scale)

#     # NOTE: we expect it computes a new scaling value only if it reaches the interval
#     # NOTE: plus 1 is taking into account the initial scaling value
#     # assert count_unique_values(input_scales) == total_steps // linear.metadatas.input.interval + 1
#     assert count_unique_values(input_scales) == total_steps // linear.metadatas.input.interval
#     assert count_unique_values(weight_scales) == total_steps // linear.weight.fp8_meta.interval
#     # NOTE: input grad's interval is 16, so the first step is a new scaling value,
#     # then 16th step is a new scaling value => n / 16 + 1
#     # assert count_unique_values(input_grad_scales) == total_steps // linear.metadatas.input_grad.interval + 1
#     assert count_unique_values(input_grad_scales) == total_steps // linear.metadatas.input_grad.interval
#     assert count_unique_values(weight_grad_scales) == total_steps // linear.metadatas.weight_grad.interval
