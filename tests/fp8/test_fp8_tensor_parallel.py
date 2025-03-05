import os

# from nanotron import distributed as dist
import nanotron.fp8.distributed as dist

# import torch.distributed as dist
import pytest
import torch
from nanotron.distributed import get_global_rank
from nanotron.fp8.linear import FP8LinearMeta
from nanotron.fp8.recipe import FP8LinearRecipe
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    FP8TensorParallelColumnLinear,
    FP8TensorParallelRowLinear,
)
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.testing.parallel import init_distributed, rerun_if_address_is_in_use
from torch import nn

# TODO(xrsrke): add test where we test the apis of fp8 parallel linear


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_fp8_column_linear_metadata(
    tp: int,
    dp: int,
    pp: int,
):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_column_linear_metadata)()


def _test_fp8_column_linear_metadata(
    parallel_context: ParallelContext,
):
    # NOTE: divisible by 16 for TP
    in_features = 32
    out_features_per_tp_rank = 16

    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    column_linear = FP8TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=False,
    )

    assert isinstance(column_linear.weight, NanotronParameter)
    # assert isinstance(column_linear.weight.data, FP8Tensor)
    assert isinstance(column_linear.weight.data, FP8Tensor)
    assert isinstance(column_linear.recipe, FP8LinearRecipe)
    assert isinstance(column_linear.metadatas, FP8LinearMeta)

    parallel_context.destroy()


# TODO(xrsrke): support gradient flow to bias
@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@pytest.mark.parametrize("tp_mode", [TensorParallelLinearMode.ALL_REDUCE])
@pytest.mark.parametrize("async_communication", [False])
@pytest.mark.parametrize("with_bias", [False])
@rerun_if_address_is_in_use()
def test_column_linear(
    tp: int,
    dp: int,
    pp: int,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool,
    with_bias: bool,
):
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE and async_communication:
        pytest.skip("ALL_REDUCE mode does not support async communication")

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_column_linear)(
        tp_mode=tp_mode,
        async_communication=async_communication,
        with_bias=with_bias,
    )


def _test_column_linear(
    parallel_context: ParallelContext,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool,
    with_bias: bool,
):
    if async_communication:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # NOTE: divisible by 16 for TP
    in_features = 32
    out_features_per_tp_rank = 16

    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    # Sharded
    column_linear = FP8TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=tp_mode,
        device="cuda",
        async_communication=async_communication,
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
            tensor=column_linear.weight.data,
            group=parallel_context.tp_pg,
        )

        if with_bias is True:
            # TODO(xrsrke): support if bias is in FP8
            # bias = column_linear.bias.data
            bias = column_linear.bias.data
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
        # assert get_data_from_param(column_linear.bias).__class__ == nn.Parameter
        assert isinstance(column_linear.bias, nn.Parameter)
        # assert column_linear.bias.data.requires_grad is (with_bias is True)

    # Generate random input
    random_input: torch.Tensor
    sharded_random_input: torch.Tensor
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        # batch_size = 5
        batch_size = 16
        random_input = torch.randn(batch_size, in_features, device="cuda")
        # synchronize random_input across tp
        dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)
        sharded_random_input = random_input
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        # sharded_batch_size = 5
        sharded_batch_size = 16
        sharded_random_input = torch.randn(sharded_batch_size, in_features, device="cuda")
        if parallel_context.tp_pg.size() > 1:
            random_input = torch.empty(
                sharded_batch_size * parallel_context.tp_pg.size(),
                *(sharded_random_input.shape[1:]),
                device=sharded_random_input.device,
                dtype=sharded_random_input.dtype,
            )
            dist.all_gather_into_tensor(random_input, sharded_random_input, group=parallel_context.tp_pg)
        else:
            random_input = sharded_random_input
    else:
        ValueError(f"Unsupported mode: {tp_mode}")

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

    hidden_dim_slice = slice(
        dist.get_rank(parallel_context.tp_pg) * out_features_per_tp_rank,
        (dist.get_rank(parallel_context.tp_pg) + 1) * out_features_per_tp_rank,
    )

    torch.testing.assert_close(
        # convert_tensor_from_fp8(column_linear.weight.data, column_linear.weight.data.fp8_meta, torch.bfloat16),
        convert_tensor_from_fp8(
            # get_data_from_param(column_linear.weight),
            # get_data_from_param(column_linear.weight).fp8_meta,
            column_linear.weight.data,
            column_linear.weight.data.fp8_meta,
            torch.bfloat16,
        ),
        reference_linear.weight[hidden_dim_slice].to(torch.bfloat16),
        rtol=0.1,
        atol=0.1,
    )

    # reference_output = ReferenceLinear.apply(random_input, reference_linear.weight, reference_linear.bias)

    # TODO @thomasw21: Tune tolerance
    try:
        torch.testing.assert_close(
            sharded_output,
            # TODO(xrsrke): retrieve accumulation precision from recipe
            # NOTE: before the reference_output.to(torch.bfloat16)
            # reference_output[
            #     :,
            #     dist.get_rank(parallel_context.tp_pg)
            #     * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            #     * out_features_per_tp_rank,
            # ].to(torch.bfloat16),
            reference_output[:, hidden_dim_slice].to(torch.bfloat16),
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

    # TODO(xrsrke): retrieve accumulation precision from recipe
    assert sharded_output.dtype == torch.bfloat16
    # NOTE(xrsrke): we expect the output is a raw torch.Tensor, not FP8Paramter, or NanotronParameter
    # assert isinstance(sharded_output, torch.Tensor)
    assert sharded_output.__class__ == torch.Tensor
    assert sharded_output.requires_grad is True

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        torch.testing.assert_close(sharded_random_input.grad, random_input.grad, rtol=0.1, atol=0.1)
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        batch_dim_slice = slice(
            dist.get_rank(parallel_context.tp_pg) * sharded_batch_size,
            (dist.get_rank(parallel_context.tp_pg) + 1) * sharded_batch_size,
        )
        torch.testing.assert_close(
            sharded_random_input.grad,
            random_input.grad[batch_dim_slice],
        )
    else:
        ValueError(f"Unsupported mode: {tp_mode}")

    if with_bias is True:
        torch.testing.assert_close(
            column_linear.bias.grad,
            reference_linear.bias.grad[hidden_dim_slice],
        )

    # if isinstance(column_linear.weight.data, FP8Tensor):
    #     # grad = column_linear.weight.data._temp_grad
    #     # grad = convert_tensor_from_fp8(grad, column_linear.weight.data._temp_grad.fp8_meta, torch.bfloat16)
    #     grad = convert_tensor_from_fp8(column_linear.weight.grad, column_linear.weight.grad.fp8_meta, torch.bfloat16)
    # else:
    #     # grad = column_linear.weight.grad
    #     grad = column_linear.weight.grad
    grad = convert_tensor_from_fp8(column_linear.weight.grad, column_linear.weight.grad.fp8_meta, torch.bfloat16)

    torch.testing.assert_close(
        grad,
        reference_linear.weight.grad[hidden_dim_slice].to(torch.bfloat16),
        # rtol=0.1, atol=0.1
        rtol=0.2,
        atol=0.2,
    )

    parallel_context.destroy()


# TODO(xrsrke): support gradient flow to bias


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@pytest.mark.parametrize("tp_mode", [TensorParallelLinearMode.ALL_REDUCE])
@pytest.mark.parametrize("async_communication", [False])
@pytest.mark.parametrize("with_bias", [False])
@rerun_if_address_is_in_use()
def test_row_linear(
    tp: int, dp: int, pp: int, tp_mode: TensorParallelLinearMode, async_communication: bool, with_bias: bool
):
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE and async_communication:
        pytest.skip("ALL_REDUCE mode does not support async communication")

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_row_linear)(
        tp_mode=tp_mode, async_communication=async_communication, with_bias=with_bias
    )


def _test_row_linear(
    parallel_context: ParallelContext, tp_mode: TensorParallelLinearMode, async_communication: bool, with_bias: bool
):
    if async_communication:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    out_features = 16
    in_features_per_rank = 32

    in_features = parallel_context.tp_pg.size() * in_features_per_rank
    dist.get_rank(parallel_context.tp_pg)

    # Sharded
    row_linear = FP8TensorParallelRowLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=tp_mode,
        device="cuda",
        async_communication=async_communication,
        bias=with_bias,
    )

    # Un-sharded
    reference_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=with_bias, device="cuda")

    # Copy weights/bias from sharded to un-sharded
    # NOTE(xrsrke): dont' use torch.inference_mode because got "Cannot set version_counter for inference tensor"
    # https://github.com/pytorch/pytorch/issues/112024
    dist.all_reduce(tensor=reference_linear.weight, op=dist.ReduceOp.SUM, group=parallel_context.tp_pg)

    sharded_weight = reference_linear.weight[
        :,
        dist.get_rank(parallel_context.tp_pg)
        * in_features_per_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
        * in_features_per_rank,
    ]
    # row_linear.weight.data.set_data(sharded_weight)
    # get_data_from_param(row_linear.weight).set_data(sharded_weight)
    row_linear._set_and_quantize_weights(sharded_weight)

    if with_bias is True:
        # broadcast bias from rank 0, and the other don't have bias
        if dist.get_rank(parallel_context.tp_pg) == 0:
            # row_linear.bias.data.copy_(reference_linear.bias)
            # get_data_from_param(row_linear.bias).copy_(reference_linear.bias)
            row_linear.bias.data.copy_(reference_linear.bias.data)

        dist.broadcast(
            tensor=reference_linear.bias,
            src=get_global_rank(group=parallel_context.tp_pg, group_rank=0),
            group=parallel_context.tp_pg,
        )

    # Generate random input
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        batch_size = 16
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        batch_size = 16 * parallel_context.tp_pg.size()
    else:
        raise ValueError()

    random_input = torch.randn(batch_size, in_features, device="cuda")
    # synchronize random_input across tp
    dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)

    assert_tensor_synced_across_pg(random_input, pg=parallel_context.tp_pg)

    # Row linear receives as input sharded input
    random_sharded_input = random_input[
        :,
        dist.get_rank(parallel_context.tp_pg)
        * in_features_per_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
        * in_features_per_rank,
    ]

    start_idx = dist.get_rank(parallel_context.tp_pg) * in_features_per_rank
    end_idx = (dist.get_rank(parallel_context.tp_pg) + 1) * in_features_per_rank
    sharded_portion = (slice(None), slice(start_idx, end_idx))
    torch.testing.assert_close(
        # convert_tensor_from_fp8(row_linear.weight.data, row_linear.weight.data.fp8_meta, torch.bfloat16),
        convert_tensor_from_fp8(
            # get_data_from_param(row_linear.weight), get_data_from_param(row_linear.weight).fp8_meta, torch.bfloat16
            row_linear.weight.data,
            row_linear.weight.data.fp8_meta,
            torch.bfloat16,
        ),
        reference_linear.weight.to(torch.bfloat16)[sharded_portion],
        rtol=0.1,
        atol=0.1,
    )

    # Test that we get the same output after forward pass
    # TODO @kunhao: We may want to have our custom error type
    reference_output = reference_linear(random_input)
    # reference_output = ReferenceLinear.apply(random_input, reference_linear.weight, reference_linear.bias)
    sharded_output = row_linear(random_sharded_input)

    assert sharded_output.dtype == torch.bfloat16
    # NOTE(xrsrke): we expect the output is a raw torch.Tensor, not FP8Paramter, or NanotronParameter
    assert sharded_output.__class__ == torch.Tensor
    assert sharded_output.requires_grad is True

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        sharded_reference_output = reference_output
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        assert batch_size % parallel_context.tp_pg.size() == 0
        sharded_batch_size = batch_size // parallel_context.tp_pg.size()
        sharded_reference_output = reference_output[
            dist.get_rank(parallel_context.tp_pg)
            * sharded_batch_size : (dist.get_rank(parallel_context.tp_pg) + 1)
            * sharded_batch_size
        ]
    else:
        raise ValueError(f"Unsupported mode: {tp_mode}")

    # TODO @thomasw21: Tune tolerance
    torch.testing.assert_close(sharded_output, sharded_reference_output.to(torch.bfloat16), rtol=0.2, atol=0.2)

    # Test that we get the same gradient after backward pass
    sharded_output.sum().backward()
    reference_output.sum().backward()

    if with_bias is True:
        if dist.get_rank(parallel_context.tp_pg) == 0:
            torch.testing.assert_close(
                row_linear.bias.grad,
                reference_linear.bias.grad,
            )
        else:
            assert row_linear.bias is None

    # if isinstance(row_linear.weight.data, FP8Tensor):
    # if isinstance(get_data_from_param(row_linear.weight), FP8Tensor):
    #     # grad = row_linear.weight.data._temp_grad
    #     # grad = convert_tensor_from_fp8(grad, row_linear.weight.data._temp_grad.fp8_meta, torch.bfloat16)
    #     # grad = get_grad_from_parameter(row_linear.weight)
    #     # grad = row_linear.weight.grad
    #     grad = convert_tensor_from_fp8(row_linear.weight.grad, row_linear.weight.grad.fp8_meta, torch.bfloat16)
    # else:
    #     # grad = row_linear.weight.grad
    #     grad = get_grad_from_parameter(row_linear.weight)
    grad = convert_tensor_from_fp8(row_linear.weight.grad, row_linear.weight.grad.fp8_meta, torch.bfloat16)

    torch.testing.assert_close(
        grad,
        reference_linear.weight.grad[
            :,
            dist.get_rank(parallel_context.tp_pg)
            * in_features_per_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            * in_features_per_rank,
        ].to(torch.bfloat16),
        rtol=0.2,
        atol=0.2,
    )

    parallel_context.destroy()
