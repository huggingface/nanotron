import pytest
import torch
from helpers.utils import available_gpus
from nanotron.timer import TimeRecorder


def hardcore_function():
    x = torch.tensor(69.0, device="cuda:0")
    for _ in range(1_000):
        x += x


@pytest.mark.skipif(available_gpus() < 1, reason="Need at least 1 GPU to run this test")
@pytest.mark.parametrize("stream", ["default", "non_default"])
def test_record_time(stream):
    stream = torch.cuda.default_stream() if stream == "default" else torch.cuda.Stream()
    EVENT_NAME = "test_timing"
    timer = TimeRecorder()

    timer.start(EVENT_NAME, stream=stream)

    with torch.cuda.stream(stream):
        hardcore_function()

    timer.end(EVENT_NAME, stream=stream)

    elapsed_time = timer.elapsed(EVENT_NAME, stream=stream)
    assert elapsed_time > 0
