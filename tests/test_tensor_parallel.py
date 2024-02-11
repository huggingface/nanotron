import os

import pytest
import torch
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.distributed import get_global_rank
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from torch import nn as torch_nn


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(i, 1, 1) for i in range(1, min(4, available_gpus()) + 1)])
@pytest.mark.parametrize("tp_mode", list(TensorParallelLinearMode))
@pytest.mark.parametrize("async_communication", [False, True])
@rerun_if_address_is_in_use()
def test_column_linear(tp: int, dp: int, pp: int, tp_mode: TensorParallelLinearMode, async_communication: bool):
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE and async_communication:
        pytest.skip("ALL_REDUCE mode does not support async communication")
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_column_linear)(
        tp_mode=tp_mode, async_communication=async_communication
    )


def _test_column_linear(
    parallel_context: ParallelContext, tp_mode: TensorParallelLinearMode, async_communication: bool
):
    if async_communication:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    in_features = 2
    out_features_per_tp_rank = 3
    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    # Sharded
    column_linear = TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=tp_mode,
        device="cuda",
        async_communication=async_communication,
    )

    # Un-sharded
    reference_linear = torch_nn.Linear(in_features=in_features, out_features=out_features, device="cuda")

    # Copy weights/bias from sharded to un-sharded
    with torch.inference_mode():
        dist.all_gather(
            tensor_list=list(reference_linear.weight.split(out_features_per_tp_rank, dim=0)),
            tensor=column_linear.weight,
            group=parallel_context.tp_pg,
        )
        dist.all_gather(
            tensor_list=list(reference_linear.bias.split(out_features_per_tp_rank, dim=0)),
            tensor=column_linear.bias,
            group=parallel_context.tp_pg,
        )

    # Generate random input
    random_input: torch.Tensor
    sharded_random_input: torch.Tensor
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        batch_size = 5
        random_input = torch.randn(batch_size, in_features, device="cuda")
        # synchronize random_input across tp
        dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)
        sharded_random_input = random_input
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        sharded_batch_size = 5
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
    # It's important that `random_input` and `sharded_random_input` are two seperate tensors with seperate storage
    sharded_random_input = sharded_random_input.clone()
    random_input.requires_grad = True
    sharded_random_input.requires_grad = True

    # Test that we get the same output after forward pass
    sharded_output = column_linear(sharded_random_input)
    reference_output = reference_linear(random_input)
    # TODO @thomasw21: Tune tolerance
    try:
        torch.testing.assert_close(
            sharded_output,
            reference_output[
                :,
                dist.get_rank(parallel_context.tp_pg)
                * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
                * out_features_per_tp_rank,
            ],
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
        column_linear.weight.grad,
        reference_linear.weight.grad[hidden_dim_slice],
    )
    torch.testing.assert_close(
        column_linear.bias.grad,
        reference_linear.bias.grad[hidden_dim_slice],
    )
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        torch.testing.assert_close(
            sharded_random_input.grad,
            random_input.grad,
        )
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

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(i, 1, 1) for i in range(1, min(4, available_gpus()) + 1)])
@pytest.mark.parametrize("tp_mode", list(TensorParallelLinearMode))
@pytest.mark.parametrize("async_communication", [False, True])
@rerun_if_address_is_in_use()
def test_row_linear(tp: int, dp: int, pp: int, tp_mode: TensorParallelLinearMode, async_communication: bool):
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE and async_communication:
        pytest.skip("ALL_REDUCE mode does not support async communication")

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_row_linear)(tp_mode=tp_mode, async_communication=async_communication)


def _test_row_linear(parallel_context: ParallelContext, tp_mode: TensorParallelLinearMode, async_communication: bool):
    if async_communication:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    out_features = 3
    in_features_per_rank = 2
    in_features = parallel_context.tp_pg.size() * in_features_per_rank

    # Sharded
    row_linear = TensorParallelRowLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=tp_mode,
        device="cuda",
        async_communication=async_communication,
    )

    # Un-sharded
    reference_linear = torch_nn.Linear(in_features=in_features, out_features=out_features, device="cuda")

    # Copy weights/bias from sharded to un-sharded
    with torch.inference_mode():
        dist.all_reduce(tensor=reference_linear.weight, op=dist.ReduceOp.SUM, group=parallel_context.tp_pg)
        row_linear.weight.copy_(
            reference_linear.weight[
                :,
                dist.get_rank(parallel_context.tp_pg)
                * in_features_per_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
                * in_features_per_rank,
            ]
        )
        # broadcast bias from rank 0, and the other don't have bias
        if dist.get_rank(parallel_context.tp_pg) == 0:
            row_linear.bias.copy_(reference_linear.bias)
        dist.broadcast(
            tensor=reference_linear.bias,
            src=get_global_rank(group=parallel_context.tp_pg, group_rank=0),
            group=parallel_context.tp_pg,
        )

    # Generate random input
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        batch_size = 5
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        batch_size = 5 * parallel_context.tp_pg.size()
    else:
        raise ValueError()
    random_input = torch.randn(batch_size, in_features, device="cuda")
    # synchronize random_input across tp
    dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)

    # Row linear receives as input sharded input
    random_sharded_input = random_input[
        :,
        dist.get_rank(parallel_context.tp_pg)
        * in_features_per_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
        * in_features_per_rank,
    ]

    # Test that we get the same output after forward pass
    # TODO @kunhao: We may want to have our custom error type
    sharded_output = row_linear(random_sharded_input)
    reference_output = reference_linear(random_input)

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
    torch.testing.assert_close(
        sharded_output,
        sharded_reference_output,
    )

    # Test that we get the same gradient after backward pass
    sharded_output.sum().backward()
    reference_output.sum().backward()
    torch.testing.assert_close(
        row_linear.weight.grad,
        reference_linear.weight.grad[
            :,
            dist.get_rank(parallel_context.tp_pg)
            * in_features_per_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            * in_features_per_rank,
        ],
    )
    if dist.get_rank(parallel_context.tp_pg) == 0:
        torch.testing.assert_close(
            row_linear.bias.grad,
            reference_linear.bias.grad,
        )
    else:
        assert row_linear.bias is None

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(i, 1, 1) for i in range(1, min(4, available_gpus()) + 1)])
@pytest.mark.parametrize("tp_mode", list(TensorParallelLinearMode))
@rerun_if_address_is_in_use()
def test_tensor_parallel_embedding(tp: int, dp: int, pp: int, tp_mode: TensorParallelLinearMode):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_tensor_parallel_embedding)(tp_mode=tp_mode)


def _test_tensor_parallel_embedding(parallel_context: ParallelContext, tp_mode: TensorParallelLinearMode):
    num_embeddings_per_rank = 100
    embedding_dim = 3
    num_embeddings = parallel_context.tp_pg.size() * num_embeddings_per_rank

    # Sharded
    sharded_embedding = TensorParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        pg=parallel_context.tp_pg,
        mode=tp_mode,
        device="cuda",
    )

    # Un-sharded
    reference_embedding = torch_nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, device="cuda")

    # Copy weights/bias from sharded to un-sharded
    with torch.inference_mode():
        dist.all_reduce(tensor=reference_embedding.weight, op=dist.ReduceOp.SUM, group=parallel_context.tp_pg)
        sharded_embedding.weight.copy_(
            reference_embedding.weight[
                dist.get_rank(parallel_context.tp_pg)
                * num_embeddings_per_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
                * num_embeddings_per_rank,
                :,
            ]
        )

    # Generate random input
    random_input: torch.Tensor
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        batch_size = 5
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        batch_size = 5 * parallel_context.tp_pg.size()
    else:
        raise ValueError(f"Unsupported mode: {tp_mode}")
    random_input = torch.randint(low=0, high=num_embeddings, size=(batch_size,), device="cuda")
    dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)

    # Test that we get the same output after forward pass
    sharded_output = sharded_embedding(random_input)
    reference_output = reference_embedding(random_input)
    weights = torch.arange(batch_size, device="cuda")[:, None]

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        sharded_reference_output = reference_output
        sharded_weights = weights
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        assert batch_size % parallel_context.tp_pg.size() == 0
        sharded_batch_size = batch_size // parallel_context.tp_pg.size()
        sharded_reference_output = reference_output[
            dist.get_rank(parallel_context.tp_pg)
            * sharded_batch_size : (dist.get_rank(parallel_context.tp_pg) + 1)
            * sharded_batch_size
        ]
        sharded_weights = weights[
            dist.get_rank(parallel_context.tp_pg)
            * sharded_batch_size : (dist.get_rank(parallel_context.tp_pg) + 1)
            * sharded_batch_size
        ]
    else:
        raise ValueError(f"Unsupported mode: {tp_mode}")

    # TODO @thomasw21: Tune tolerance
    torch.testing.assert_close(sharded_output, sharded_reference_output, atol=0, rtol=0)

    # Test that we get the same gradient after backward pass
    (sharded_output * sharded_weights).sum().backward()
    (reference_output * weights).sum().backward()
    torch.testing.assert_close(
        sharded_embedding.weight.grad,
        reference_embedding.weight.grad[
            dist.get_rank(parallel_context.tp_pg)
            * num_embeddings_per_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            * num_embeddings_per_rank,
            :,
        ],
        atol=0,
        rtol=0,
    )

    parallel_context.destroy()
