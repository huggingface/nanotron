from typing import Dict, Tuple

import torch


class TimeRecorder:
    """Record time between two events on a given CUDA stream."""

    def __init__(self):
        self._start_events: Dict[Tuple[str, torch.cuda.Stream], torch.cuda.Event] = {}
        self._end_events: Dict[Tuple[str, torch.cuda.Stream], torch.cuda.Event] = {}
        self._streams: Dict[Tuple[str, torch.cuda.Stream], torch.cuda.Stream] = {}

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

    def elapsed(self, name: str, stream: torch.cuda.Stream = None) -> float:
        """Return elapsed time in a given stream.

        Args:
            name: Name of the event.
            stream: CUDA stream to record the event on.
        """
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
