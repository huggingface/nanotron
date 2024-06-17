from copy import deepcopy
from typing import cast

import pytest
import torch
import transformer_engine as te  # noqa
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8LinearMeta
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor
from nanotron.helpers import init_optimizer_and_grad_accumulator
from nanotron.parallel import ParallelContext
from nanotron.scaling.parametrization import ParametrizationMethod
from nanotron.testing.parallel import init_distributed, rerun_if_address_is_in_use
from nanotron.testing.utils import DEFAULT_OPTIMIZER_CONFIG, create_nanotron_model


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
@pytest.mark.parametrize("num_steps", [1, 5, 10, 25])
@rerun_if_address_is_in_use()
def test_delayed_quantization_for_fp8_linear(tp: int, dp: int, pp: int, num_steps: int):
    EXPECTED_SCALES = {
        # NOTE: num_inp_scales, num_w_scales, num_inp_grad_scales, num_w_grad_scales
        1: (1, 1, 1, 1),
        5: (1, 5, 1, 5),
        10: (1, 10, 1, 10),
        25: (2, 25, 2, 25),
    }
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    label_ids = torch.randint(0, 100, size=(16, 64))
    label_mask = torch.ones_like(label_ids).bool()

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_delayed_quantization_for_fp8_linear)(
        input_ids=input_ids,
        input_mask=input_mask,
        label_ids=label_ids,
        label_mask=label_mask,
        num_steps=num_steps,
        expected_scales=EXPECTED_SCALES[num_steps],
    )


def _test_delayed_quantization_for_fp8_linear(
    parallel_context: ParallelContext,
    num_steps: int,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    label_ids: torch.Tensor,
    label_mask: torch.Tensor,
    expected_scales,
):
    def count_unique_values(xs):
        # NOTE: sometimes, we compute a new scaling value
        # but the value could be the same as the previous iterations
        # but the memory address is different
        return len({x.data_ptr() for x in xs})

    input_ids = input_ids.to("cuda")
    input_mask = input_mask.to("cuda")
    label_ids = label_ids.to("cuda")
    label_mask = label_mask.to("cuda")

    nanotron_model = create_nanotron_model(parallel_context, dtype=torch.int8)
    optim, _ = init_optimizer_and_grad_accumulator(
        parametrization_method=ParametrizationMethod.STANDARD,
        model=nanotron_model,
        optimizer_args=DEFAULT_OPTIMIZER_CONFIG,
        parallel_context=parallel_context,
    )

    # NOTE: this take the assumption of the fp8 recipe
    # at the time writing this test
    input_scales = []
    weight_scales = []
    input_grad_scales = []
    weight_grad_scales = []

    input_grad_ready_to_scales = []
    weight_ready_to_scales = []

    metadatas = nanotron_model.model.decoder[0].pp_block.mlp.gate_up_proj.metadatas
    weight_metadatas = nanotron_model.model.decoder[0].pp_block.mlp.gate_up_proj.weight.data.fp8_meta
    metadatas = cast(FP8LinearMeta, metadatas)
    weight_metadatas = cast(FP8Meta, weight_metadatas)

    for i in range(num_steps):
        optim.zero_grad()
        loss = nanotron_model(input_ids=input_ids, input_mask=input_mask, label_ids=label_ids, label_mask=label_mask)[
            "loss"
        ]
        loss.backward()
        optim.step()

        input_scales.append(metadatas.input.scale)
        weight_scales.append(weight_metadatas.scale)
        input_grad_scales.append(metadatas.input_grad.scale)
        weight_grad_scales.append(metadatas.weight_grad.scale)

        input_grad_ready_to_scales.append(metadatas.input_grad.is_ready_to_scale)
        weight_ready_to_scales.append(metadatas.weight_grad.is_ready_to_scale)

    assert count_unique_values(input_scales) == expected_scales[0]
    assert count_unique_values(weight_scales) == expected_scales[1]
    assert count_unique_values(input_grad_scales) == expected_scales[2]
    assert count_unique_values(weight_grad_scales) == expected_scales[3]

    parallel_context.destroy()
