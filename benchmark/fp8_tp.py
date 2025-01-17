import argparse
import itertools

import pandas as pd
import torch
import torch.distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import FP8TensorParallelColumnLinear
from torch.profiler import ProfilerActivity


def run_experiment(exp_name, M, N, K, TP_SIZE, parallel_context):
    torch.cuda.synchronize()
    input = torch.randn(M, K, device="cuda", requires_grad=True)
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

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{exp_name}"),
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
        with_modules=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        use_cuda=True,
    ) as prof:
        prof.step()
        sharded_output.sum().backward()

    return prof


def print_profiling_table(prof, sort_by="cpu_time_total"):
    print(f"###### sorted by {sort_by} ######")
    print(
        prof.key_averages(group_by_stack_n=100).table(
            sort_by=sort_by,
            row_limit=20,
            top_level_events_only=False,
            # max_src_column_width=2000,  # Increase source column width
            # max_name_column_width=2000,
            # max_shapes_column_width=1000,
            max_src_column_width=100,  # Increase source column width
            max_name_column_width=30,
            max_shapes_column_width=100,
        )
    )


def explore_event_values(event):
    for attr in dir(event):
        if not attr.startswith("_"):  # Skip internal attributes
            try:
                value = getattr(event, attr)
                if callable(value):  # Skip methods
                    continue
                print(f"\n{attr}:")
                print(value)
                print("-" * 50)  # Separator for better readability
            except Exception:
                print(f"{attr}: <error accessing attribute>")


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


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    # Parse dimensions from comma-separated string to list of integers
    dimensions = [int(d.strip()) for d in args.dimensions.split(",")]
    TP_SIZE = args.tp_size
    EXP_NUMBER = args.exp_number

    # dimensions = [1024, 2048, 4096, 8192, 16384]
    # TP_SIZE = 8

    results = []
    total = len(list(itertools.product(dimensions, dimensions, dimensions)))
    experiment_count = 0
    parallel_context = ParallelContext(data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=TP_SIZE)

    for M, N, K in itertools.product(dimensions, dimensions, dimensions):
        exp_name = f"{EXP_NUMBER}_fp8_m{M}_n{N}_k{K}_and_tp{TP_SIZE}"
        total += 1
        print(f"Running experiment with M={M}, N={N}, K={K}, {experiment_count}/{total}")

        prof = run_experiment(exp_name, M, N, K, TP_SIZE=TP_SIZE, parallel_context=parallel_context)

        if dist.get_rank() == 0:
            print_profiling_table(prof, sort_by="cpu_time_total")
            print_profiling_table(prof, sort_by="cuda_time_total")
            print_profiling_table(prof, sort_by="self_cuda_time_total")
            # explore_event_values(table)

            # Get top 5 operations by CPU time
            # sorted_events = prof.key_averages().table(sort_by="cpu_time_total")

            # NOTE: loop through all events and sum up the total time, then calculate the percent
            averages = prof.key_averages(group_by_stack_n=100)
            # NOTE: why sum .self_cpu_time_total instead of .cpu_time_total?
            # source: https://github.com/pytorch/pytorch/blob/f14f245747db2f80e963bd72561f5bd5ed216a4a/torch/autograd/profiler_util.py#L976-L990
            # i test and it matches the torch's generated table
            cpu_time_total_of_all_events = sum([event.self_cpu_time_total for event in averages])
            sorted_events = sorted(averages, key=lambda x: x.cpu_time_total, reverse=True)[:5]

            for event in sorted_events:
                event_cpu_time_percent = (event.cpu_time_total / cpu_time_total_of_all_events) * 100

                results.append(
                    {
                        "M": M,
                        "N": N,
                        "K": K,
                        "Operation": event.key,
                        "CPU Time (ms)": event.cpu_time_total / 1000,  # Convert to milliseconds
                        "CPU Time %": f"{event_cpu_time_percent:.2f}%",
                        "CUDA Time (ms)": event.cuda_time_total / 1000,  # Convert to milliseconds
                        # 'Memory Used (MB)': event.cpu_memory_usage / (1024 * 1024) if event.cpu_memory_usage else 0
                    }
                )

if dist.get_rank() == 0:
    df = pd.DataFrame(results)
    print("\nTop 5 most time-consuming operations for each dimension combination:")
    print(df.to_string())
    df.to_csv(
        f'{EXP_NUMBER}_profiling_results_with_m_n_k_with_cartesian_product_{"_".join(map(str, dimensions))}.csv',
        index=False,
    )
