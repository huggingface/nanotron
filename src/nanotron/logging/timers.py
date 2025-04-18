import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch

from nanotron import distributed as dist
from nanotron import logging

logger = logging.get_logger(__name__)


class TimerType(Enum):
    CPU = "cpu"  # Regular CPU timer (uses time.time())
    CUDA = "cuda"  # CUDA-aware timer (uses CUDA events)


@dataclass
class TimerRecord:
    """
    Records timing information for a single timer.

    By default, uses CUDA events for timing GPU operations, which provides more accurate
    measurements of GPU execution time without forcing CPU-GPU synchronization.

    For CPU-only operations, you can use CPU-based timing by specifying timer_type=TimerType.CPU.
    """

    name: str
    timer_type: TimerType = TimerType.CUDA
    start_time: float = 0.0
    end_time: float = 0.0
    running: bool = False
    call_count: int = 0
    cuda_sync: bool = False  # Option to add CUDA synchronization for more accurate timings
    enabled: bool = True  # Allow individual timer to be enabled/disabled

    # For CPU timers we still track total_time
    _cpu_total_time: float = 0.0

    # CUDA specific fields
    _cuda_events: List[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    _current_start_event: Optional[torch.cuda.Event] = None

    def __enter__(self):
        """Context manager support: Start the timer when entering a context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support: End the timer when exiting a context."""
        self.end()
        return False  # Don't suppress exceptions

    def start(self) -> "TimerRecord":
        """Start the timer."""
        if self.name == "dummy" or not self.enabled:  # disabled
            return self

        if self.running:
            logger.warning(f"Timer '{self.name}' already running. Restarting.")

        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available():
                # Synchronize before starting timing if requested
                if self.cuda_sync:
                    torch.cuda.synchronize()
                # Create a new start event - we'll create the end event when end() is called
                self._current_start_event = torch.cuda.Event(enable_timing=True)
                self._current_start_event.record()
            else:
                logger.warning("CUDA timer requested but CUDA is not available. Falling back to CPU timer.")
                self.timer_type = TimerType.CPU
                self.start_time = time.time()
        else:
            self.start_time = time.time()

        self.running = True
        return self

    def end(self) -> None:
        """End the timer, but don't compute elapsed time yet."""

        if self.name == "dummy" or not self.enabled:  # disabled
            return

        if not self.running:
            logger.warning(f"Timer '{self.name}' was not running. Ignoring end call.")
            return

        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available() and self._current_start_event is not None:
                # Synchronize before ending timing if requested
                if self.cuda_sync:
                    torch.cuda.synchronize()
                # Create and record an end event
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()

                # Store the start/end event pair for later querying
                self._cuda_events.append((self._current_start_event, end_event))
                self._current_start_event = None
            else:
                logger.warning("CUDA timer end called but CUDA events are not available.")
                self.timer_type = TimerType.CPU
                self.end_time = time.time()
                self._cpu_total_time += self.end_time - self.start_time
        else:
            self.end_time = time.time()
            self._cpu_total_time += self.end_time - self.start_time

        self.call_count += 1
        self.running = False

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = 0.0
        self.end_time = 0.0
        self.running = False
        self.call_count = 0
        self._cpu_total_time = 0.0
        self._cuda_events = []
        self._current_start_event = None

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds for the current timer."""
        if not self.running:
            if self.timer_type == TimerType.CPU:
                return self.end_time - self.start_time

            # For CUDA timers, we need to synchronize to get the last elapsed time
            if not self._cuda_events:
                return 0.0

            # Get the last event pair
            start_event, end_event = self._cuda_events[-1]
            end_event.synchronize()  # Make sure the event is complete
            return start_event.elapsed_time(end_event) / 1000.0  # Convert ms to sec

        # Timer is still running
        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available() and self._current_start_event is not None:
                # Create a temporary end event to measure elapsed time so far
                if self.cuda_sync:
                    torch.cuda.synchronize()
                tmp_end_event = torch.cuda.Event(enable_timing=True)
                tmp_end_event.record()
                tmp_end_event.synchronize()
                return self._current_start_event.elapsed_time(tmp_end_event) / 1000.0
            else:
                return time.time() - self.start_time
        else:
            return time.time() - self.start_time

    @property
    def total_time(self) -> float:
        """
        Get total time in seconds across all calls.
        Warning: For CUDA timers, this will synchronize all events!
        """
        if self.timer_type == TimerType.CPU:
            return self._cpu_total_time

        # For CUDA timers, we need to sum up all the event pairs
        total = 0.0
        for start_event, end_event in self._cuda_events:
            end_event.synchronize()  # Make sure the event is complete
            total += start_event.elapsed_time(end_event) / 1000.0  # Convert ms to sec
        return total

    @property
    def average_time(self) -> float:
        """
        Get average time per call in seconds.
        Warning: For CUDA timers, this will synchronize all events!
        """
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count


class Timers:
    """
    A collection of timers for tracking execution time in Nanotron.

    By default, timers use CUDA events for timing GPU operations, which provides several benefits:
    1. More accurate measurement of GPU execution time
    2. Reduced need for explicit CUDA synchronization
    3. Lower overhead compared to CPU-based timing with synchronization
    4. Better performance monitoring for distributed training

    For CPU-only operations, you can still use CPU-based timing by specifying timer_type=TimerType.CPU.
    """

    _instance = None
    _enabled = os.environ.get("ENABLE_TIMERS", "0") == "1"  # Add global enable/disable flag

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
        self,
        name: str,
        timer_type: Union[TimerType, str] = TimerType.CUDA,
        cuda_sync: bool = False,
        enabled: bool = bool(int(os.environ.get("ENABLE_TIMERS", "0"))),
    ) -> TimerRecord:
        """Get or create a timer with the given name.

        Can be used as a decorator, context manager, or directly:
        - @nanotron_timer  # As decorator with default CUDA timing
        - @nanotron_timer("my_function")  # As decorator with custom name
        - @nanotron_timer(timer_type=TimerType.CPU)  # As decorator with CPU timing
        - with nanotron_timer("name"): ...  # As context manager
        - nanotron_timer("name").start(); ...; nanotron_timer("name").end()  # Direct use

        Args:
            name: Name of the timer
            timer_type: Type of timer, either TimerType.CUDA (default) or TimerType.CPU
                        (or 'cuda'/'cpu' strings)
            cuda_sync: Whether to perform torch.cuda.synchronize() for more accurate CUDA timing.
                       Default is False to avoid unnecessary synchronization overhead.
            enabled: Override default enabled setting from environment variable

        Raises:
            ValueError: If a timer with the same name already exists with different settings
        """
        if isinstance(timer_type, str):
            timer_type = TimerType(timer_type)

        if callable(name):
            # Being used as a decorator with specified or default settings
            func = name
            timer_name = func.__name__
            return self._create_timer_decorator(timer_name, timer_type, cuda_sync, enabled)(func)

        if name in self._timers:
            existing_timer = self._timers[name]
            if (
                existing_timer.timer_type != timer_type
                or existing_timer.cuda_sync != cuda_sync
                or existing_timer.enabled != enabled
            ):
                raise ValueError(
                    f"Timer '{name}' already exists with different settings.\n"
                    f"Existing: type={existing_timer.timer_type}, cuda_sync={existing_timer.cuda_sync}, enabled={existing_timer.enabled}\n"
                    f"New: type={timer_type}, cuda_sync={cuda_sync}, enabled={enabled}"
                )
            return existing_timer

        self._timers[name] = TimerRecord(name=name, timer_type=timer_type, cuda_sync=cuda_sync, enabled=enabled)

        # Check if we're being called as a decorator
        if not callable(name):
            timer_record = self._timers[name]
            # Return the timer which can be used directly or as a context manager
            return timer_record

        # If we get here, we're being called as @nanotron_timer("name", timer_type)
        return self._create_timer_decorator(name, timer_type, cuda_sync, enabled)

    def _create_timer_decorator(self, name, timer_type=TimerType.CUDA, cuda_sync=False, enabled=None):
        """Create a decorator that times the execution of a function."""

        def decorator(func):
            import functools

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self(name, timer_type, cuda_sync, enabled):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

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
            # This will trigger synchronization for CUDA timers
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
                    # This will trigger synchronization for CUDA timers
                    avg_time = timer.average_time * 1000  # Convert to ms
                    total_time = timer.total_time * 1000  # Convert to ms
                    logger.info(
                        f"Timer '{name}' ({timer.timer_type.value}): {total_time:.2f}ms total, "
                        f"{avg_time:.2f}ms avg, {timer.call_count} calls"
                    )
            logger.info("----------------------------")

    def items(self):
        if not self._enabled:
            return []
        return self._timers.items()


# Create a singleton instance
nanotron_timer = Timers()
