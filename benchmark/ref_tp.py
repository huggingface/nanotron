# from nanotron import distributed as dist
import nanotron.fp8.distributed as dist

# import torch.distributed as dist
import torch
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import get_data_from_param, get_grad_from_parameter
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    FP8TensorParallelColumnLinear,
)
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from torch import nn

if __name__ == "__main__":
    with_bias = False
    # NOTE: divisible by 16 for TP
    in_features = 32
    out_features_per_tp_rank = 16

    parallel_context = ParallelContext(data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=2)

    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    # Sharded
    column_linear = FP8TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=with_bias,
    )

    # Un-sharded
    reference_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=with_bias, device="cuda")

    # Copy weights/bias from sharded to un-sharded
    with torch.inference_mode():
        # weight = column_linear.weight.data
        # weight = convert_tensor_from_fp8(weight, weight.fp8_meta, torch.bfloat16),
        dist.all_gather(
            tensor_list=list(reference_linear.weight.split(out_features_per_tp_rank, dim=0)),
            # tensor=column_linear.weight.data,
            tensor=get_data_from_param(column_linear.weight),
            group=parallel_context.tp_pg,
        )

        if with_bias is True:
            # TODO(xrsrke): support if bias is in FP8
            # bias = column_linear.bias.data
            bias = get_data_from_param(column_linear.bias)
            bias = bias.to(reference_linear.bias.dtype) if bias.dtype != reference_linear.bias.dtype else bias
            dist.all_gather(
                tensor_list=list(reference_linear.bias.split(out_features_per_tp_rank, dim=0)),
                tensor=bias,
                group=parallel_context.tp_pg,
            )

    # TODO(xrsrke)
    if with_bias is True:
        assert column_linear.bias.requires_grad is (with_bias is True)
        # assert column_linear.bias.data.__class__ == torch.Tensor
        assert get_data_from_param(column_linear.bias).__class__ == nn.Parameter
        # assert column_linear.bias.data.requires_grad is (with_bias is True)

    # Generate random input
    random_input: torch.Tensor
    sharded_random_input: torch.Tensor

    # batch_size = 5
    batch_size = 16
    random_input = torch.randn(batch_size, in_features, device="cuda")
    # synchronize random_input across tp
    dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)
    sharded_random_input = random_input

    dist.barrier()
    assert_tensor_synced_across_pg(random_input, pg=parallel_context.tp_pg)

    # It's important that `random_input` and `sharded_random_input` are two separate tensors with separate storage
    sharded_random_input = sharded_random_input.clone()
    sharded_random_input = sharded_random_input.contiguous()
    random_input.requires_grad = True
    sharded_random_input.requires_grad = True

    # Test that we get the same output after forward pass
    sharded_output = column_linear(sharded_random_input)

    reference_output = reference_linear(random_input)
    # reference_output = ReferenceLinear.apply(random_input, reference_linear.weight, reference_linear.bias)

    # TODO @thomasw21: Tune tolerance
    try:
        torch.testing.assert_close(
            sharded_output,
            # TODO(xrsrke): retrieve accumulation precision from recipe
            # NOTE: before the reference_output.to(torch.bfloat16)
            reference_output[
                :,
                dist.get_rank(parallel_context.tp_pg)
                * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
                * out_features_per_tp_rank,
            ].to(torch.bfloat16),
            rtol=0,
            atol=0.1,
        )
    except BaseException as e:
        print(f"Rank {dist.get_rank(parallel_context.tp_pg)}: FAIL.")
        dist.barrier()
        raise e

    print(f"Rank {dist.get_rank(parallel_context.tp_pg)}: SUCCESS.")
    dist.barrier()

    # Test that we get the same gradient after backward pass
    sharded_output.sum().backward()
    reference_output.sum().backward()
    hidden_dim_slice = slice(
        dist.get_rank(parallel_context.tp_pg) * out_features_per_tp_rank,
        (dist.get_rank(parallel_context.tp_pg) + 1) * out_features_per_tp_rank,
    )

    torch.testing.assert_close(
        # convert_tensor_from_fp8(column_linear.weight.data, column_linear.weight.data.fp8_meta, torch.bfloat16),
        convert_tensor_from_fp8(
            get_data_from_param(column_linear.weight),
            get_data_from_param(column_linear.weight).fp8_meta,
            torch.bfloat16,
        ),
        reference_linear.weight[hidden_dim_slice].to(torch.bfloat16),
        rtol=0.1,
        atol=0.1,
    )

    # TODO(xrsrke): retrieve accumulation precision from recipe
    assert sharded_output.dtype == torch.bfloat16
    # NOTE(xrsrke): we expect the output is a raw torch.Tensor, not FP8Paramter, or NanotronParameter
    # assert isinstance(sharded_output, torch.Tensor)
    assert sharded_output.__class__ == torch.Tensor
    assert sharded_output.requires_grad is True

    torch.testing.assert_close(sharded_random_input.grad, random_input.grad, rtol=0.1, atol=0.1)

    if with_bias is True:
        torch.testing.assert_close(
            column_linear.bias.grad,
            reference_linear.bias.grad[hidden_dim_slice],
        )

    if isinstance(get_data_from_param(column_linear.weight), FP8Tensor):
        # grad = column_linear.weight.data._temp_grad
        # grad = convert_tensor_from_fp8(grad, column_linear.weight.data._temp_grad.fp8_meta, torch.bfloat16)
        grad = get_grad_from_parameter(column_linear.weight)
        grad = convert_tensor_from_fp8(grad, grad.fp8_meta, torch.bfloat16)
    else:
        # grad = column_linear.weight.grad
        grad = get_grad_from_parameter(column_linear.weight)

    torch.testing.assert_close(
        grad,
        reference_linear.weight.grad[hidden_dim_slice].to(torch.bfloat16),
        # rtol=0.1, atol=0.1
        rtol=0.2,
        atol=0.2,
    )

    parallel_context.destroy()
