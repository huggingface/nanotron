import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import torch

from nanotron import distributed as dist
from nanotron import logging

logger = logging.get_logger(__name__)


class TimerType(Enum):
    CPU = "cpu"  # Regular CPU timer (uses time.time())
    CUDA = "cuda"  # CUDA-aware timer (uses CUDA events)


@dataclass
class TimerRecord:
    """Records timing information for a single timer."""

    name: str
    timer_type: TimerType = TimerType.CPU
    start_time: float = 0.0
    end_time: float = 0.0
    running: bool = False
    call_count: int = 0
    total_time: float = 0.0

    # CUDA specific fields
    _start_event: Optional[torch.cuda.Event] = None
    _end_event: Optional[torch.cuda.Event] = None
    _last_elapsed_time: float = 0.0

    def start(self) -> "TimerRecord":
        """Start the timer."""
        if self.running:
            logger.warning(f"Timer '{self.name}' already running. Restarting.")

        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available():
                # Create CUDA events with timing enabled
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)

                # Record the start event
                self._start_event.record()
            else:
                logger.warning("CUDA timer requested but CUDA is not available. Falling back to CPU timer.")
                self.timer_type = TimerType.CPU
                self.start_time = time.time()
        else:
            self.start_time = time.time()

        self.running = True
        return self

    def end(self) -> float:
        """End the timer and return elapsed time in seconds."""
        if not self.running:
            logger.warning(f"Timer '{self.name}' was not running. Ignoring end call.")
            return 0.0

        elapsed = 0.0
        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available() and self._start_event is not None and self._end_event is not None:
                # Record the end event
                self._end_event.record()

                # Waits for all preceding CUDA operations to complete
                self._end_event.synchronize()

                # Get the elapsed time in milliseconds and convert to seconds
                elapsed = self._start_event.elapsed_time(self._end_event) / 1000.0
                self._last_elapsed_time = elapsed
            else:
                logger.warning("CUDA timer end called but CUDA events are not available.")
                self.timer_type = TimerType.CPU
                self.end_time = time.time()
                elapsed = self.end_time - self.start_time
        else:
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time

        self.total_time += elapsed
        self.call_count += 1
        self.running = False
        return elapsed

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = 0.0
        self.end_time = 0.0
        self.running = False
        self.call_count = 0
        self.total_time = 0.0
        self._start_event = None
        self._end_event = None
        self._last_elapsed_time = 0.0

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if not self.running:
            if self.timer_type == TimerType.CUDA:
                return self._last_elapsed_time
            return self.end_time - self.start_time

        # Timer is still running
        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available() and self._start_event is not None:
                # Create a temporary end event
                tmp_end_event = torch.cuda.Event(enable_timing=True)
                tmp_end_event.record()
                tmp_end_event.synchronize()
                return self._start_event.elapsed_time(tmp_end_event) / 1000.0
            else:
                logger.warning("CUDA timer elapsed called but CUDA events are not available.")
                return time.time() - self.start_time
        else:
            return time.time() - self.start_time

    @property
    def average_time(self) -> float:
        """Get average time per call in seconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count


class Timers:
    """A collection of timers for tracking execution time in Nanotron."""

    _instance = None
    _enabled = os.environ.get("ENABLE_TIMERS", "1") == "1"  # Add global enable/disable flag

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timers, cls).__new__(cls)
            cls._instance._timers: Dict[str, TimerRecord] = {}
        return cls._instance

    @classmethod
    def enable(cls) -> None:
        """Enable all timing operations."""
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        """Disable all timing operations."""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if timers are enabled."""
        return cls._enabled

    def __call__(
        self, name: str, timer_type: Union[TimerType, str] = TimerType.CPU, cuda_sync: bool = True
    ) -> TimerRecord:
        """Get or create a timer with the given name.

        Args:
            name: Name of the timer
            timer_type: Type of timer, either TimerType.CPU or TimerType.CUDA
                        (or 'cpu'/'cuda' strings)
        """
        if not self._enabled:
            # Return a dummy timer that does nothing when timing is disabled
            return TimerRecord(name="dummy", timer_type=TimerType.CPU)

        if isinstance(timer_type, str):
            timer_type = TimerType(timer_type)

        if name not in self._timers:
            self._timers[name] = TimerRecord(name=name, timer_type=timer_type)
        elif self._timers[name].timer_type != timer_type:
            logger.warning(
                f"Timer '{name}' already exists with type {self._timers[name].timer_type}. "
                f"Requested type {timer_type} will be ignored."
            )
        return self._timers[name]

    def reset_all(self) -> None:
        """Reset all timers."""
        for timer in self._timers.values():
            timer.reset()

    def reset(self, name: str) -> None:
        """Reset a specific timer."""
        if name in self._timers:
            self._timers[name].reset()

    def log(self, name: str, logger=None, rank: Optional[int] = 0, group=None) -> None:
        """Log a specific timer on the specified rank."""
        if name not in self._timers:
            return

        if logger is None:
            logger = logging.get_logger(__name__)

        world_rank = dist.get_rank() if group is None else dist.get_rank(group)
        if rank is not None and world_rank != rank:
            return

        timer = self._timers[name]
        if timer.call_count > 0:
            avg_time = timer.average_time * 1000  # Convert to ms
            total_time = timer.total_time * 1000  # Convert to ms
            logger.info(
                f"Timer '{name}' ({timer.timer_type.value}): {total_time:.2f}ms total, "
                f"{avg_time:.2f}ms avg, {timer.call_count} calls"
            )

    def log_all(self, logger=None, rank: Optional[int] = 0, group=None) -> None:
        """Log all timers on the specified rank."""
        if logger is None:
            logger = logging.get_logger(__name__)

        world_rank = dist.get_rank() if group is None else dist.get_rank(group)
        if rank is not None and world_rank != rank:
            return

        # Sort timers by name for consistent output
        sorted_timers = sorted(self._timers.items())

        if sorted_timers:
            logger.info("---- Timing Information ----")
            for name, timer in sorted_timers:
                if timer.call_count > 0:
                    avg_time = timer.average_time * 1000  # Convert to ms
                    total_time = timer.total_time * 1000  # Convert to ms
                    logger.info(
                        f"Timer '{name}' ({timer.timer_type.value}): {total_time:.2f}ms total, "
                        f"{avg_time:.2f}ms avg, {timer.call_count} calls"
                    )
            logger.info("----------------------------")


# Create a singleton instance
nanotron_timer = Timers()
