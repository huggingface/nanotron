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


# def color_scale(val, min_val, max_val):
#     """Generate a color scale from white to dark blue based on value."""
#     if pd.isna(val):
#         return 'background-color: white'
#     normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0
#     return f'background-color: rgba(0, 0, 139, {normalized:.2f}); color: {"white" if normalized > 0.5 else "black"}'


def color_scale(val, min_val, max_val, metric_type="default"):
    """Generate background color based on value and metric type."""
    if pd.isna(val):
        return "background-color: white"

    normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0

    if metric_type == "time":  # Lower is better - red scale
        color = f"background-color: rgba(255, {int(255 * (1-normalized))}, {int(255 * (1-normalized))}, 0.8)"
    elif metric_type == "performance":  # Higher is better - green scale
        color = f"background-color: rgba({int(255 * (1-normalized))}, 255, {int(255 * (1-normalized))}, 0.8)"
    elif metric_type == "efficiency":  # Higher is better - blue scale
        color = f"background-color: rgba({int(255 * (1-normalized))}, {int(255 * (1-normalized))}, 255, 0.8)"
    else:  # Default purple scale
        color = f"background-color: rgba({int(255 * (1-normalized))}, 0, 255, 0.8)"

    text_color = "white" if normalized > 0.5 else "black"
    return f"{color}; color: {text_color}"


# [Previous benchmark_fn_in_sec and run functions remain the same...]

# def create_html_table(df, exp_number, tp_size):
#     # Style the dataframe
#     styled_df = df.style.format({
#         'FP8_time_ms': '{:.2f}',
#         'BF16_time_ms': '{:.2f}',
#         'FP8_TFLOPS': '{:.2f}',
#         'BF16_TFLOPS': '{:.2f}',
#         'FP8_efficiency_%': '{:.2f}',
#         'BF16_efficiency_%': '{:.2f}',
#         'Speedup': '{:.2f}'
#     })

#     # Apply color scaling to specific columns
#     styled_df = styled_df.apply(lambda x: pd.Series([
#         color_scale(v, x.min(), x.max(), 'time') if col.endswith('time_ms')
#         else color_scale(v, x.min(), x.max(), 'performance') if col.endswith('TFLOPS')
#         else color_scale(v, x.min(), x.max(), 'efficiency') if col.endswith('efficiency_%')
#         else color_scale(v, x.min(), x.max()) if col == 'Speedup'
#         else '' for v in x
#     ]), axis=0)

#     # Generate HTML
#     html = f'''
#     <html>
#     <head>
#         <style>
#             table {{ border-collapse: collapse; width: 100%; }}
#             th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
#             th {{ background-color: #4CAF50; color: white; }}
#             tr:nth-child(even) {{ background-color: #f9f9f9; }}
#             .header {{ text-align: center; padding: 20px; }}
#         </style>
#     </head>
#     <body>
#         <div class="header">
#             <h2>Benchmark Results (TP_SIZE={tp_size})</h2>
#             <p>Experiment: {exp_number}</p>
#         </div>
#         {styled_df.to_html()}
#     </body>
#     </html>
#     '''

#     with open(f'{exp_number}_benchmark_results_tp{tp_size}.html', 'w') as f:
#         f.write(html)


def create_html_table(df, exp_number, tp_size):
    def style_df(df):
        # Create an empty DataFrame with the same shape as the input
        styled = pd.DataFrame("", index=df.index, columns=df.columns)

        # Style specific columns
        for column in df.columns:
            if column.endswith("time_ms"):
                styled[column] = df[column].apply(lambda x: color_scale(x, df[column].min(), df[column].max(), "time"))
            elif column.endswith("TFLOPS"):
                styled[column] = df[column].apply(
                    lambda x: color_scale(x, df[column].min(), df[column].max(), "performance")
                )
            elif column.endswith("efficiency_%"):
                styled[column] = df[column].apply(
                    lambda x: color_scale(x, df[column].min(), df[column].max(), "efficiency")
                )
            elif column == "Speedup":
                styled[column] = df[column].apply(lambda x: color_scale(x, df[column].min(), df[column].max()))
        return styled

    # Format numbers and apply styling
    styled_df = df.style.format(
        {
            "FP8_time_ms": "{:.2f}",
            "BF16_time_ms": "{:.2f}",
            "FP8_TFLOPS": "{:.2f}",
            "BF16_TFLOPS": "{:.2f}",
            "FP8_efficiency_%": "{:.2f}",
            "BF16_efficiency_%": "{:.2f}",
            "Speedup": "{:.2f}",
        }
    ).apply(lambda _: style_df(df), axis=None)

    # Generate HTML
    html = f"""
    <html>
    <head>
        <style>
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: right; border: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; text-align: center; }}
            tr:nth-child(even) td:not([style*="background-color"]) {{ background-color: #f9f9f9; }}
            .header {{ text-align: center; padding: 20px; background-color: #f5f5f5; }}
            td[style*="background-color"] {{ transition: all 0.3s ease; }}
            td[style*="background-color"]:hover {{ opacity: 0.8; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Benchmark Results (TP_SIZE={tp_size})</h2>
            <p>Experiment: {exp_number}</p>
        </div>
        {styled_df.to_html(table_id="results")}
    </body>
    </html>
    """

    with open(f"{exp_number}_benchmark_results_tp{tp_size}.html", "w") as f:
        f.write(html)


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)

    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    measurement = t0.blocked_autorange()
    return measurement.mean


def run_fp8_linear(input, M, N, K, parallel_context, include_backward=False):
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


def run_linear(input, M, N, K, parallel_context, include_backward=False):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run profiling experiments with configurable dimensions")
    parser.add_argument("--exp_number", type=str, help="Experiment number")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument(
        "--dimensions",
        type=str,
        default="4096,16384,32768,28672,49152",
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
    parallel_context = ParallelContext(data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=TP_SIZE)

    # Run benchmarks and collect results
    results = []
    i = 0
    for M, N, K in itertools.product(dimensions, dimensions, dimensions):
        i += 1
        # Run forward-only case
        result = benchmark_linear_operations(M, N, K, parallel_context, include_backward=False)
        results.append(result)
        print(f"Experiment {i}/{total} complete (Forward-only)")

        # Run forward+backward case
        result = benchmark_linear_operations(M, N, K, parallel_context, include_backward=True)
        results.append(result)
        print(f"Experiment {i}/{total} complete (Forward+Backward)")

    df = pd.DataFrame(results)
    df = df.round(2)  # Round to 2 decimal places
    df = df.sort_values(by=["M", "N", "K", "Include_Backward"])

    print(df)

    # # Define columns to color and their respective color scales
    # color_columns = {
    #     'FP8_time_ms': 'Reds',
    #     'BF16_time_ms': 'Blues',
    #     'FP8_TFLOPS': 'Greens',
    #     'BF16_TFLOPS': 'Purples',
    #     'FP8_efficiency_%': 'Oranges',
    #     'BF16_efficiency_%': 'Viridis',
    #     'Speedup': 'RdYlBu'
    # }

    # # Create the table
    # fig = go.Figure(data=[go.Table(
    #     header=dict(
    #         values=list(df.columns),
    #         fill_color='lightgrey',
    #         align='left',
    #         font=dict(size=12, color='black')
    #     ),
    #     cells=dict(
    #         values=[df[col] for col in df.columns],
    #         align='left',
    #         font=dict(size=11),
    #         # Format cells with colors for numeric columns
    #         fill_color=[
    #             'white' if col not in color_columns else
    #             [f'rgba({int(255*i)}, {int(255*i)}, {int(255*i)}, 0.5)'
    #              for i in (df[col]-df[col].min())/(df[col].max()-df[col].min())]
    #             for col in df.columns
    #         ]
    #     )
    # )])

    # # Update layout
    # fig.update_layout(
    #     title=f'Benchmark Results (TP_SIZE={TP_SIZE})',
    #     width=1200,
    #     height=800,
    #     margin=dict(l=20, r=20, t=40, b=20)
    # )

    # # Save the interactive HTML file
    # fig.write_html(f'{EXP_NUMBER}_benchmark_results_tp{TP_SIZE}.html')
    create_html_table(df, EXP_NUMBER, TP_SIZE)

    print(f"\nResults have been saved to {EXP_NUMBER}_benchmark_results_tp{TP_SIZE}.html")
