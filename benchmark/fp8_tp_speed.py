import argparse
import itertools

import pandas as pd
import torch
from nanotron.models.base import init_on_device_and_dtype
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import FP8TensorParallelColumnLinear, TensorParallelColumnLinear
from torch.utils import benchmark

# H100 SXM specs: bottom of https://www.nvidia.com/en-us/data-center/h100/
h100_peak_flops_float32 = 67e12
h100_peak_flops_fp16_tc = 989e12
h100_peak_tops_float8_tc = 1979e12

dtype_to_peak_tops = {
    torch.float32: h100_peak_flops_float32,
    torch.float16: h100_peak_flops_fp16_tc,
    torch.bfloat16: h100_peak_flops_fp16_tc,
    torch.float8_e4m3fn: h100_peak_tops_float8_tc,
    torch.float8_e5m2: h100_peak_tops_float8_tc,
}


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)

    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    measurement = t0.blocked_autorange()
    return measurement.mean


def run_fp8_linear(input, M, N, K, parallel_context, include_backward=False):
    # input = torch.randn(M, K, device="cuda", requires_grad=False)
    column_linear = FP8TensorParallelColumnLinear(
        in_features=K,
        out_features=N,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=False,
    )

    sharded_output = column_linear(input)

    if include_backward is True:
        sharded_output.sum().backward()

    # return sharded_output


def run_linear(input, M, N, K, parallel_context, include_backward=False):
    # input = torch.randn(M, K, device="cuda", requires_grad=False)
    with init_on_device_and_dtype(device="cuda", dtype=torch.bfloat16):
        column_linear = TensorParallelColumnLinear(
            in_features=K,
            out_features=N,
            pg=parallel_context.tp_pg,
            mode=TensorParallelLinearMode.ALL_REDUCE,
            device="cuda",
            async_communication=False,
            bias=False,
        )

    sharded_output = column_linear(input)

    if include_backward is True:
        sharded_output.sum().backward()

    # assert sharded_output.dtype == torch.bfloat16, f"Expected bfloat16, got {sharded_output.dtype}"
    # return sharded_output


def parse_args():
    parser = argparse.ArgumentParser(description="Run profiling experiments with configurable dimensions")
    parser.add_argument("--exp_number", type=str, help="Experiment number")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument(
        "--dimensions",
        type=str,
        default="1024,2048,4096,8192,16384,32768",
        help="Comma-separated list of dimensions to test",
    )
    return parser.parse_args()


def benchmark_linear_operations(M, N, K, parallel_context, include_backward):
    input = torch.randn(M, K, device="cuda", requires_grad=False)
    bfloat16_input = torch.randn(M, K, device="cuda", requires_grad=False, dtype=torch.bfloat16)

    # Benchmark FP8
    fp8_time = benchmark_fn_in_sec(run_fp8_linear, input, M, N, K, parallel_context, include_backward)

    # Benchmark BFloat16
    bfloat16_time = benchmark_fn_in_sec(run_linear, bfloat16_input, M, N, K, parallel_context, include_backward)

    # Calculate FLOPS
    # Each linear operation performs 2*M*N*K FLOPs (multiply-add)
    total_flops = 2 * M * N * K // parallel_context.tensor_parallel_size

    fp8_tflops = (total_flops / fp8_time) / 1e12
    bfloat16_tflops = (total_flops / bfloat16_time) / 1e12

    # Calculate efficiency compared to peak performance
    fp8_efficiency = (fp8_tflops / (h100_peak_tops_float8_tc / 1e12)) * 100
    bfloat16_efficiency = (bfloat16_tflops / (h100_peak_flops_fp16_tc / 1e12)) * 100

    return {
        "M": M,
        "N": N,
        "K": K,
        "Include_Backward": include_backward,
        "FP8_time_ms": fp8_time * 1000,
        "BF16_time_ms": bfloat16_time * 1000,
        "FP8_TFLOPS": fp8_tflops,
        "BF16_TFLOPS": bfloat16_tflops,
        "FP8_efficiency_%": fp8_efficiency,
        "BF16_efficiency_%": bfloat16_efficiency,
        "Speedup": bfloat16_time / fp8_time,
    }


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    dimensions = [int(d.strip()) for d in args.dimensions.split(",")]
    TP_SIZE = args.tp_size
    EXP_NUMBER = args.exp_number

    results = []
    total = len(list(itertools.product(dimensions, dimensions, dimensions)))
    experiment_count = 0
    parallel_context = ParallelContext(data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=TP_SIZE)

    # Run benchmarks and collect results
    results = []
    i = 0
    for M, N, K in itertools.product(dimensions, dimensions, dimensions):
        i += 1
        # result = benchmark_linear_operations(M, N, K, parallel_context)
        # results.append(result)
        # print(f"Experiment {i}/{total} complete")

        # Run forward-only case
        result = benchmark_linear_operations(M, N, K, parallel_context, include_backward=False)
        results.append(result)
        print(f"Experiment {i}/{total} complete (Forward-only)")

        # Run forward+backward case
        result = benchmark_linear_operations(M, N, K, parallel_context, include_backward=True)
        results.append(result)
        print(f"Experiment {i}/{total} complete (Forward+Backward)")

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.round(2)  # Round to 2 decimal places

    # Sort by matrix size for better readability
    df = df.sort_values(by=["M", "N", "K", "Include_Backward"])

    print("\nBenchmark Results: (tp_size={})".format(TP_SIZE))
    print(df.to_string(index=False))
