# CUDA Event-Based Timing in Nanotron

## Overview

Nanotron now uses CUDA events for timing GPU operations instead of CPU-based timing with `time.time()`. This change provides several benefits:

1. **More accurate measurement of GPU execution time**: CUDA events are recorded directly on the GPU timeline, providing more precise timing of GPU operations.
2. **Reduced need for explicit CUDA synchronization**: CPU-based timing requires synchronization between CPU and GPU to get accurate measurements, which can introduce overhead and affect performance.
3. **Lower overhead**: CUDA event-based timing has minimal impact on the execution of GPU operations.
4. **Better performance monitoring**: More accurate timing leads to better performance analysis and optimization.

## Implementation Details

The implementation uses `torch.cuda.Event` with `enable_timing=True` to create start and end events that are recorded on the GPU timeline. The elapsed time is then calculated using `start_event.elapsed_time(end_event)`, which returns the time in milliseconds.

### Key Changes

1. **Default Timer Type**: The default timer type in `nanotron/src/nanotron/logging/timers.py` has been changed from `TimerType.CPU` to `TimerType.CUDA`.

2. **Iteration Timing**: The iteration timing in `trainer.py` now uses CUDA events instead of `time.time()`.

3. **Synchronization Control**: By default, CUDA event-based timers do not force synchronization unless explicitly requested with `cuda_sync=True`.

## Usage

### Basic Usage

```python
# Create and use a CUDA timer (default)
with nanotron_timer("my_operation"):
    # Your GPU operation here
    ...

# Explicitly specify CUDA timing
with nanotron_timer("my_operation", timer_type="cuda"):
    # Your GPU operation here
    ...

# For CPU-only operations, you can still use CPU-based timing
with nanotron_timer("cpu_operation", timer_type="cpu"):
    # Your CPU operation here
    ...

# As a decorator with default CUDA timing
@nanotron_timer
def my_function():
    # Your GPU operation here
    ...

# As a decorator with custom name
@nanotron_timer("custom_name")
def my_function():
    # Your GPU operation here
    ...

# As a decorator with CPU timing
@nanotron_timer(timer_type=TimerType.CPU)
def my_cpu_function():
    # Your CPU operation here
    ...
```

### Advanced Usage

```python
# Start and end a timer manually
timer = nanotron_timer("my_operation")
timer.start()
# Your operation here
timer.end()

# Get the elapsed time in seconds
elapsed_time = timer.elapsed

# Get the total time across all calls
total_time = timer.total_time

# Get the average time per call
avg_time = timer.average_time
```

## Considerations

1. **Synchronization**: By default, CUDA event-based timers do not force synchronization to avoid overhead. If you need more accurate timing at the cost of performance, you can set `cuda_sync=True`.

2. **Units**: CUDA events measure time in milliseconds, but the timer API converts this to seconds for consistency with the previous CPU-based timing.

3. **Fallback**: If CUDA is not available, the timer will automatically fall back to CPU-based timing.

## Performance Impact

Using CUDA events for timing instead of CPU-based timing with synchronization can significantly reduce overhead, especially in distributed training scenarios with thousands of GPUs.
