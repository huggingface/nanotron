"""Test script for the timer decorator with both CPU and CUDA timer types."""

from nanotron.logging.timers import nanotron_timer, TimerType
import time
import torch

# Enable timers for testing
nanotron_timer.enable()

# Test with default CUDA timing
@nanotron_timer
def test_default_decorator():
    """Test function with default CUDA timing."""
    # Simulate some work
    time.sleep(0.1)
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
    return "Done"

# Test with explicit CUDA timing
@nanotron_timer(timer_type=TimerType.CUDA)
def test_cuda_decorator():
    """Test function with explicit CUDA timing."""
    # Simulate some work
    time.sleep(0.1)
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
    return "Done"

# Test with CPU timing
@nanotron_timer(timer_type=TimerType.CPU)
def test_cpu_decorator():
    """Test function with CPU timing."""
    # Simulate some CPU work
    time.sleep(0.2)
    return "Done"

# Test with custom name
@nanotron_timer("custom_name")
def test_custom_name_decorator():
    """Test function with custom name."""
    # Simulate some work
    time.sleep(0.1)
    return "Done"

if __name__ == "__main__":
    print("Testing timer decorators...")
    
    # Run the test functions
    test_default_decorator()
    test_cuda_decorator()
    test_cpu_decorator()
    test_custom_name_decorator()
    
    # Log all timers
    print("\nTimer results:")
    nanotron_timer.log_all(rank=None)  # Log on all ranks
    
    print("\nTest completed successfully!")
