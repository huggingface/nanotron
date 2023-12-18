import logging
from typing import Dict, Literal, Tuple

import torch

from nanotron.constants import TIME_RECORDER_MESSAGE


class TimeRecorder:
    """Record time between two events on a given CUDA stream."""

    LOG_LEVELS = ["INFO", "DEBUG", "WARNING", "ERROR"]

    def __init__(
        self,
        is_logging: bool = True,
        save_log: bool = True,
        log_level: Literal["INFO", "DEBUG", "WARNING", "ERROR"] = "INFO",
    ):
        self.is_logging = is_logging
        self.save_log = save_log
        self.log_level = getattr(logging, log_level)

        self._start_events: Dict[Tuple[str, torch.cuda.Stream], torch.cuda.Event] = {}
        self._end_events: Dict[Tuple[str, torch.cuda.Stream], torch.cuda.Event] = {}
        self._streams: Dict[Tuple[str, torch.cuda.Stream], torch.cuda.Stream] = {}

        self._setup_logger()

    def _setup_logger(self):
        formatter = logging.Formatter("nanotron - %(name)s - %(levelname)s: %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        self._logger = logging.getLogger(__name__)
        self._logger.addHandler(handler)

    def start(self, name: str, stream: torch.cuda.Stream = None):
        """Start recording time in a given stream.

        Args:
            name: Name of the event.
            stream: CUDA stream to record the event on.
        """
        assert name not in self._start_events, f"Start event for {name} already exists"
        stream = stream or torch.cuda.current_stream()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self._start_events[(name, stream)] = start_event
        self._end_events[(name, stream)] = end_event

        start_event.record(stream)

    def end(self, name: str, stream: torch.cuda.Stream = None):
        """End recording time in a given stream.

        Args:
            name: Name of the event.
            stream: CUDA stream to record the event on.
        """
        stream = stream or torch.cuda.current_stream()
        self._sanity_check(name, stream)
        self._end_events[(name, stream)].record(stream)

        if self.is_logging:
            self._logger.log(
                self.log_level, TIME_RECORDER_MESSAGE.format(name=name, elapsed_time=self.elapsed(name, stream))
            )

    def elapsed(self, name: str, stream: torch.cuda.Stream = None) -> float:
        """Return elapsed time in a given stream.

        Args:
            name: Name of the event.
            stream: CUDA stream to record the event on.
        """
        stream = stream or torch.cuda.current_stream()
        self._sanity_check(name, stream)

        end_event = self._end_events[(name, stream)]
        end_event.synchronize()
        return self._start_events[(name, stream)].elapsed_time(end_event)

    def _sanity_check(self, name: str, stream: torch.cuda.Stream = None):
        assert (name, stream) in self._start_events, f"Start event for {name} and this stream not found"
        assert (name, stream) in self._end_events, f"Stream for {name} and this stream not found"

    def reset(self):
        self._start_events = {}
        self._end_events = {}
        self._streams = {}
