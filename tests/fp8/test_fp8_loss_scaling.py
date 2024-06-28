import pytest
import torch
import transformer_engine as te  # noqa
from nanotron.fp8.loss_scaler import LossScaler
from nanotron.helpers import init_optimizer_and_grad_accumulator
from nanotron.parallel import ParallelContext
from nanotron.scaling.parametrization import ParametrizationMethod
from nanotron.testing.parallel import init_distributed, rerun_if_address_is_in_use
from nanotron.testing.utils import DEFAULT_OPTIMIZER_CONFIG, create_nanotron_model


def test_loss_scaler_attributes():
    scaling_value = torch.tensor(1.0)
    scaling_factor = torch.tensor(2.0)
    interval = 10

    loss_scaler = LossScaler(scaling_value, scaling_factor, interval)

    assert loss_scaler.scaling_value == scaling_value
    assert loss_scaler.scaling_factor == scaling_factor
    assert loss_scaler.interval == interval


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_scaled_fp8_gradients(tp: int, dp: int, pp: int):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    label_ids = torch.randint(0, 100, size=(16, 64))
    label_mask = torch.ones_like(label_ids).bool()

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_scaled_fp8_gradients)(
        input_ids=input_ids,
        input_mask=input_mask,
        label_ids=label_ids,
        label_mask=label_mask,
    )


def _test_scaled_fp8_gradients(
    parallel_context: ParallelContext,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    label_ids: torch.Tensor,
    label_mask: torch.Tensor,
):
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

    loss_scaler = LossScaler()

    for _ in range(3):
        optim.zero_grad()
        outputs = nanotron_model(
            input_ids=input_ids, input_mask=input_mask, label_ids=label_ids, label_mask=label_mask
        )
        loss = outputs["loss"]
        scaled_loss = loss_scaler.scale(loss)
        scaled_loss.backward()
        optim.step()

    parallel_context.destroy()
