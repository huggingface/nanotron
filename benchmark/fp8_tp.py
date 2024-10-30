# from nanotron import distributed as dist

# import torch.distributed as dist
import torch
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    FP8TensorParallelColumnLinear,
)
from torch.profiler import ProfilerActivity

if __name__ == "__main__":
    # NOTE: divisible by 16 for TP
    in_features = 4096
    out_features = 4096 * 4

    parallel_context = ParallelContext(data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=2)
    # out_features_per_tp_rank = 16
    # out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    batch_size = 128
    seq_len = 8192
    merged_gbs = batch_size * seq_len
    sharded_random_input = torch.randn(batch_size, in_features, device="cuda", requires_grad=True)

    column_linear = FP8TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=False,
    )

    def trace_handler(p):
        output = p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=20)
        print(output)
        p.export_chrome_trace("./trace_" + str(p.step_num) + ".json")

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./log/exp900a01_fp8_tp2_and_mbs_16",
            # use_gzip=True
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        # with_flops=True,
        with_modules=True,
        # on_trace_ready=trace_handler,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        use_cuda=True,
    ) as prof:
        prof.step()
        sharded_output = column_linear(sharded_random_input)
        sharded_output.sum().backward()


# if dist.get_rank() == 0:
# print("sharded_output.dtype: ", sharded_output.dtype)
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20, top_level_events_only=False))

# # Print detailed events with stack traces
# for event in prof.events():
#     if 'aten::copy_' in event.name:
#         print(f"\nOperation: {event.name}")
#         print(f"CUDA time: {event.cuda_time_total/1000:.2f}ms")
#         if event.stack:  # Print full stack trace
#             print("Stack trace:")
#             try:
#                 # Print the raw stack information
#                 print("Raw stack info:", event.stack)

#                 # Safely iterate through stack frames
#                 for frame in event.stack:
#                     if isinstance(frame, (list, tuple)):
#                         print("  ", " - ".join(str(x) for x in frame))
#                     else:
#                         print("  ", frame)
#             except Exception as e:
#                 print(f"Error printing stack: {e}")
#                 print("Raw stack data:", repr(event.stack))
#         print("-" * 80)


print(
    prof.key_averages(group_by_stack_n=1000).table(
        sort_by="self_cuda_time_total",
        row_limit=20,
        max_src_column_width=100,  # Increase source column width
        top_level_events_only=False,
        # max_name_column_width=1000,
        max_name_column_width=120,
    )
)

# print(prof.key_averages(group_by_stack_n=5).table(
#     sort_by="self_cuda_time_total",
#     row_limit=20,
#     max_src_column_width=100,  # Increase source column width
#     top_level_events_only=False
# ))

# prof.export_stacks("_x", "self_cuda_time_total")

# for event in prof.events():
#     print(event.name, event.stack)

# prof.export_chrome_trace("trace.json")
