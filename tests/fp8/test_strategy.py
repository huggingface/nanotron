import pytest
import torch
from nanotron.fp8.strategy import (
    # DelayStrategy,
    IntimeStrategy,
    SkipOverflowStrategy,
    SkipZeroOnlyStrategy,
    WarmupStrategy,
)
from nanotron.fp8.tracker import DynamicScaler
from nanotron.fp8.utils import convert_linear_to_fp8
from torch import nn


@pytest.fixture
def fp8_linear():
    linear = nn.Linear(16, 16, bias=True, device="cuda")
    return convert_linear_to_fp8(linear)


@pytest.mark.skip
@pytest.mark.parametrize("interval", [1, 5, 10])
@pytest.mark.parametrize("n_updates", [1, 2, 5])
def test_warmup_strategy(interval, n_updates, fp8_linear):
    input = torch.randn((16, 16), dtype=torch.float32, device="cuda")

    fp8_linear = DynamicScaler.track(fp8_linear, strategy=[WarmupStrategy(interval=interval)])

    count = 0
    scaling_factors = []
    for _ in range(interval * n_updates):
        # TODO(xrsrke): how to setup this up without the edge case of overflow/underflow
        fp8_linear(input).sum().backward()

        assert fp8_linear.is_warmup is False if count % interval == 0 else fp8_linear.is_warmup is True
        count += 1

    # TODO(xrsrke): is this correct?
    assert len(set(scaling_factors)) == n_updates


@pytest.mark.skip
@pytest.mark.parametrize("interval", [1, 5, 10])
@pytest.mark.parametrize("n_updates", [1, 2, 5])
def test_delay_strategy(interval, n_updates, fp8_linear):
    input = torch.randn((16, 16), dtype=torch.float32, device="cuda")

    fp8_linear = DynamicScaler.track(fp8_linear, strategy=[DelayStrategy])

    count = 0
    scaling_factors = []
    for _ in range(interval * n_updates):
        # TODO(xrsrke): how to setup this up without the edge case of overflow/underflow
        fp8_linear(input).sum().backward()
        count += 1

    assert len(set(scaling_factors)) == n_updates


@pytest.mark.skip
@pytest.mark.parametrize("n_updates", [1, 2, 5])
def test_intime_strategy(n_updates, fp8_linear):
    input = torch.randn(16, 16, device="cuda")

    fp8_linear = DynamicScaler.track(fp8_linear, strategy=[IntimeStrategy])

    scaling_factors = []
    for _ in range(n_updates):
        fp8_linear(input).sum().backward()

    assert len(set(scaling_factors)) == n_updates


# TODO(xrsrke): better naming? something like we immeditely skip deplay scaling if
# we encounter overflow/underflow
@pytest.mark.skip
@pytest.mark.parametrize("n_updates", [1, 2, 5])
def test_skip_overflow_underflow_strategy(n_updates, fp8_linear):
    # TODO(xrsrke): make the input leads to overflow/underflow
    input = torch.randn(16, 16, device="cuda")

    fp8_linear = DynamicScaler.track(fp8_linear, strategy=[SkipOverflowStrategy])

    scaling_factors = []
    for _ in range(n_updates):
        fp8_linear(input).sum().backward()

    assert len(set(scaling_factors)) == n_updates


# TODO(xrsrke): better naming?
@pytest.mark.skip
@pytest.mark.parametrize("n_updates", [1, 2, 5])
def test_skip_zero_only_strategy(n_updates, fp8_linear):
    # TODO(xrsrke): make the input leads to overflow/underflow
    input = torch.randn(16, 16, device="cuda")

    fp8_linear = DynamicScaler.track(fp8_linear, strategy=[SkipZeroOnlyStrategy])

    scaling_factors = []
    for _ in range(n_updates):
        fp8_linear(input).sum().backward()

    assert len(set(scaling_factors)) == n_updates


# TODO(xrsrke): decoupling the dynamic quantization from optimizer => fp8linear should work with torch optimizer
