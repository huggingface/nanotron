from typing import Union

import pytest
import torch
from helpers.dummy import DummyModel, dummy_infinite_data_loader
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.models import init_on_device_and_dtype
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from torch import nn
from torch.nn import functional as F


@pytest.mark.skipif(available_gpus() < 2, reason="Testing build_and_set_rank requires at least 2 gpus")
@rerun_if_address_is_in_use()
def test_build_and_set_rank():
    init_distributed(tp=1, dp=1, pp=2)(_test_build_and_set_rank)()


def _test_build_and_set_rank(parallel_context: ParallelContext):
    device = torch.device("cuda")
    p2p = P2P(pg=parallel_context.pp_pg, device=device)
    model = DummyModel(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size()
    with init_on_device_and_dtype(device):
        for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

    # Check that the ranks are set correctly
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    assert model.mlp[current_pp_rank].linear.rank == current_pp_rank
    assert model.mlp[current_pp_rank].activation.rank == current_pp_rank

    # Check that blocks were built on the correct ranks
    for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
        if pp_rank == current_pp_rank:
            assert hasattr(non_linear.linear, "pp_block")
            assert hasattr(non_linear.activation, "pp_block")
        else:
            assert not hasattr(non_linear.linear, "pp_block")
            assert not hasattr(non_linear.activation, "pp_block")

    parallel_context.destroy()


@pytest.mark.skipif(available_gpus() < 1, reason="Testing test_init_on_device_and_dtype requires at least 1 gpus")
def test_init_on_device_and_dtype():
    device = torch.device(type="cuda", index=0)
    with init_on_device_and_dtype(device=device, dtype=torch.bfloat16):
        model = nn.Linear(10, 10)

    assert model.weight.dtype == torch.bfloat16, "Model weight wasn't initialised with the correct dtype"
    assert model.weight.device == device, "Model weight wasn't initialised with the correct device"


@pytest.mark.skipif(available_gpus() < 2, reason="Testing AFAB requires at least 2 gpus")
@pytest.mark.parametrize(
    "pipeline_engine", [AllForwardAllBackwardPipelineEngine(), OneForwardOneBackwardPipelineEngine()]
)
@pytest.mark.parametrize("pp", list(range(2, min(4, available_gpus()) + 1)))
@rerun_if_address_is_in_use()
def test_pipeline_engine(pipeline_engine: PipelineEngine, pp: int):
    init_distributed(tp=1, dp=1, pp=pp)(_test_pipeline_engine)(pipeline_engine=pipeline_engine)


def _test_pipeline_engine(parallel_context: ParallelContext, pipeline_engine: PipelineEngine):
    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    has_reference_model = dist.get_rank(parallel_context.pp_pg) == reference_rank
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    # spawn model
    model = DummyModel(p2p=p2p)
    if has_reference_model:
        reference_model = DummyModel(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size()
    with init_on_device_and_dtype(device):
        for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for non_linear in reference_model.mlp:
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for pp_rank in range(parallel_context.pp_pg.size()):
                non_linear = model.mlp[pp_rank]
                reference_non_linear = reference_model.mlp[pp_rank]
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    reference_non_linear.linear.pp_block.weight.data.copy_(non_linear.linear.pp_block.weight.data)
                    reference_non_linear.linear.pp_block.bias.data.copy_(non_linear.linear.pp_block.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.linear.pp_block.weight.data.copy_(weight.data)
                reference_non_linear.linear.pp_block.bias.data.copy_(bias.data)
    else:
        p2p.send_tensors(
            [model.mlp[current_pp_rank].linear.pp_block.weight, model.mlp[current_pp_rank].linear.pp_block.bias],
            to_rank=reference_rank,
        )

    # Get infinite dummy data iterator
    data_iterator = dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg)  # First rank receives data

    # Have at least as many microbatches as PP size.
    n_micro_batches_per_batch = parallel_context.pp_pg.size() + 5

    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]
    losses = pipeline_engine.train_batch_iter(
        model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=n_micro_batches_per_batch, grad_accumulator=None
    )

    # Equivalent on the reference model
    if has_reference_model:
        reference_losses = []
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            loss /= n_micro_batches_per_batch
            loss.backward()
            reference_losses.append(loss.detach())

    # Gather loss in reference_rank
    if has_reference_model:
        _losses = []
    for loss in losses:
        if isinstance(loss["loss"], torch.Tensor):
            if has_reference_model:
                _losses.append(loss["loss"])
            else:
                p2p.send_tensors([loss["loss"]], to_rank=reference_rank)
        else:
            assert isinstance(loss["loss"], TensorPointer)
            if not has_reference_model:
                continue
            _losses.append(p2p.recv_tensors(num_tensors=1, from_rank=loss["loss"].group_rank)[0])
    if has_reference_model:
        losses = _losses

    # Check loss are the same as reference
    if has_reference_model:
        for loss, ref_loss in zip(losses, reference_losses):
            torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-7)

    # Check that gradient flows through the entire model
    for param in model.parameters():
        assert param.grad is not None

    # Check that gradient are the same as reference
    if has_reference_model:
        for pp_rank in range(parallel_context.pp_pg.size()):
            non_linear = model.mlp[pp_rank]
            reference_non_linear = reference_model.mlp[pp_rank]
            if pp_rank == current_pp_rank:
                # We already have the weights locally
                torch.testing.assert_close(
                    non_linear.linear.pp_block.weight.grad,
                    reference_non_linear.linear.pp_block.weight.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(
                    non_linear.linear.pp_block.bias.grad,
                    reference_non_linear.linear.pp_block.bias.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(
                weight_grad, reference_non_linear.linear.pp_block.weight.grad, atol=1e-6, rtol=1e-7
            )
            torch.testing.assert_close(bias_grad, reference_non_linear.linear.pp_block.bias.grad, atol=1e-6, rtol=1e-7)
    else:
        p2p.send_tensors(
            [
                model.mlp[current_pp_rank].linear.pp_block.weight.grad,
                model.mlp[current_pp_rank].linear.pp_block.bias.grad,
            ],
            to_rank=reference_rank,
        )

    parallel_context.destroy()


@pytest.mark.skipif(
    available_gpus() < 2,
    reason="Testing `test_pipeline_engine_with_tensor_that_does_not_require_grad` requires at least 2 gpus",
)
@pytest.mark.parametrize(
    "pipeline_engine", [AllForwardAllBackwardPipelineEngine(), OneForwardOneBackwardPipelineEngine()]
)
@pytest.mark.parametrize("pp", list(range(2, min(4, available_gpus()) + 1)))
@rerun_if_address_is_in_use()
def test_pipeline_engine_with_tensor_that_does_not_require_grad(pipeline_engine: PipelineEngine, pp: int):
    init_distributed(pp=pp, dp=1, tp=1)(_test_pipeline_engine_with_tensor_that_does_not_require_grad)(
        pipeline_engine=pipeline_engine
    )


def _test_pipeline_engine_with_tensor_that_does_not_require_grad(
    parallel_context: ParallelContext, pipeline_engine: PipelineEngine
):
    def activation(x: torch.Tensor, y: torch.Tensor):
        return {"output": F.sigmoid(x) * y, "y": y}

    class LinearWithDummyInput(nn.Linear):
        def __init__(self, in_features, out_features):
            super().__init__(in_features=in_features, out_features=out_features)

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            return {"output": super().forward(x), "y": y}

    class DummyModelPassingNonDifferentiableTensor(nn.Module):
        def __init__(
            self,
            p2p: P2P,
        ):
            super().__init__()
            self.p2p = p2p
            self.mlp = nn.Sequential(
                *(
                    nn.ModuleDict(
                        {
                            "linear": PipelineBlock(
                                p2p=p2p,
                                module_builder=LinearWithDummyInput,
                                module_kwargs={"in_features": 10, "out_features": 10},
                                module_input_keys={"x", "y"},
                                module_output_keys={"output", "y"},
                            ),
                            "activation": PipelineBlock(
                                p2p=p2p,
                                module_builder=lambda: activation,
                                module_kwargs={},
                                module_input_keys={"x", "y"},
                                module_output_keys={"output", "y"},
                            ),
                        }
                    )
                    for _ in range(p2p.pg.size() + 1)
                )
            )

            self.loss = PipelineBlock(
                p2p=p2p,
                module_builder=lambda: lambda x: x.sum(),
                module_kwargs={},
                module_input_keys={"x"},
                module_output_keys={"output"},
            )

        def forward(
            self,
            differentiable_tensor: Union[torch.Tensor, TensorPointer],
            non_differentiable_tensor: Union[torch.Tensor, TensorPointer],
        ):
            for non_linear in self.mlp:
                linear_output = non_linear.linear(x=differentiable_tensor, y=non_differentiable_tensor)
                output = non_linear.activation(x=linear_output["output"], y=linear_output["y"])
                differentiable_tensor, non_differentiable_tensor = output["output"], output["y"]

                if isinstance(differentiable_tensor, torch.Tensor):
                    assert differentiable_tensor.requires_grad is True
                if isinstance(non_differentiable_tensor, torch.Tensor):
                    assert non_differentiable_tensor.requires_grad is False

            differentiable_tensor = self.loss(x=differentiable_tensor)["output"]
            return differentiable_tensor

    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    has_reference_model = current_pp_rank == reference_rank

    # spawn model
    model = DummyModelPassingNonDifferentiableTensor(p2p=p2p)
    if has_reference_model:
        reference_model = DummyModelPassingNonDifferentiableTensor(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size() + 1
    # An additional mlp is in the end
    mlp_index_pp_rank = [(i, i) for i in range(parallel_context.pp_pg.size())] + [
        (parallel_context.pp_pg.size(), parallel_context.pp_pg.size() - 1)
    ]

    with init_on_device_and_dtype(device):
        for (mlp_index, pp_rank), non_linear in zip(mlp_index_pp_rank, model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for non_linear in reference_model.mlp:
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for (mlp_index, pp_rank) in mlp_index_pp_rank:
                non_linear = model.mlp[mlp_index]
                reference_non_linear = reference_model.mlp[mlp_index]
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    reference_non_linear.linear.pp_block.weight.data.copy_(non_linear.linear.pp_block.weight.data)
                    reference_non_linear.linear.pp_block.bias.data.copy_(non_linear.linear.pp_block.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.linear.pp_block.weight.data.copy_(weight.data)
                reference_non_linear.linear.pp_block.bias.data.copy_(bias.data)
    else:
        for (mlp_index, pp_rank) in mlp_index_pp_rank:
            if pp_rank == current_pp_rank:
                p2p.send_tensors(
                    [model.mlp[mlp_index].linear.pp_block.weight, model.mlp[mlp_index].linear.pp_block.bias],
                    to_rank=reference_rank,
                )

    # Get infinite dummy data iterator
    def dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg: dist.ProcessGroup, dtype=torch.float, input_pp_rank=0
    ):
        micro_batch_size = 3
        # We assume the first linear is always built on the first rank.
        while True:
            yield {
                "differentiable_tensor": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "non_differentiable_tensor": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
            }

    data_iterator = dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg=parallel_context.pp_pg
    )  # First rank receives data

    # Have at least as many microbatches as PP size.
    n_micro_batches_per_batch = parallel_context.pp_pg.size() + 5

    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]
    losses = pipeline_engine.train_batch_iter(
        model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=n_micro_batches_per_batch, grad_accumulator=None
    )
    # Equivalent on the reference model
    if has_reference_model:
        reference_losses = []
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            loss /= n_micro_batches_per_batch
            loss.backward()
            reference_losses.append(loss.detach())

    # Gather loss in reference_rank
    if has_reference_model:
        _losses = []
    for loss in losses:
        if isinstance(loss["loss"], torch.Tensor):
            if has_reference_model:
                _losses.append(loss["loss"])
            else:
                p2p.send_tensors([loss["loss"]], to_rank=reference_rank)
        else:
            assert isinstance(loss["loss"], TensorPointer)
            if not has_reference_model:
                continue
            _losses.append(p2p.recv_tensors(num_tensors=1, from_rank=loss["loss"].group_rank)[0])
    if has_reference_model:
        losses = _losses

    # Check loss are the same as reference
    if has_reference_model:
        for loss, ref_loss in zip(losses, reference_losses):
            torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-7)

    # Check that gradient flows through the entire model
    for param in model.parameters():
        assert param.grad is not None

    # Check that gradient are the same as reference
    if has_reference_model:
        for (mlp_index, pp_rank) in mlp_index_pp_rank:
            non_linear = model.mlp[mlp_index]
            reference_non_linear = reference_model.mlp[mlp_index]
            if pp_rank == current_pp_rank:
                # We already have the weights locally
                torch.testing.assert_close(
                    non_linear.linear.pp_block.weight.grad,
                    reference_non_linear.linear.pp_block.weight.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(
                    non_linear.linear.pp_block.bias.grad,
                    reference_non_linear.linear.pp_block.bias.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(
                weight_grad, reference_non_linear.linear.pp_block.weight.grad, atol=1e-6, rtol=1e-7
            )
            torch.testing.assert_close(bias_grad, reference_non_linear.linear.pp_block.bias.grad, atol=1e-6, rtol=1e-7)
    else:
        for (mlp_index, pp_rank) in mlp_index_pp_rank:
            if pp_rank == current_pp_rank:
                p2p.send_tensors(
                    [model.mlp[mlp_index].linear.pp_block.weight.grad, model.mlp[mlp_index].linear.pp_block.bias.grad],
                    to_rank=reference_rank,
                )

    parallel_context.destroy()


@pytest.mark.parametrize("pp", list(range(2, min(4, available_gpus()) + 1)))
@rerun_if_address_is_in_use()
def test_pipeline_forward_without_engine(pp: int):
    init_distributed(pp=pp, dp=1, tp=1)(_test_pipeline_forward_without_engine)()


def _test_pipeline_forward_without_engine(parallel_context: ParallelContext):
    def activation(x: torch.Tensor, y: torch.Tensor):
        return {"output": F.sigmoid(x) * y, "y": y}

    class DummyModel(nn.Module):
        def __init__(
            self,
            p2p: P2P,
        ):
            super().__init__()
            self.p2p = p2p
            self.mlp = nn.Sequential(
                *(
                    nn.ModuleDict(
                        {
                            "linear": PipelineBlock(
                                p2p=p2p,
                                module_builder=nn.Linear,
                                module_kwargs={"in_features": 10, "out_features": 10},
                                module_input_keys={"input"},
                                module_output_keys={"output"},
                            ),
                            "activation": PipelineBlock(
                                p2p=p2p,
                                module_builder=lambda: activation,
                                module_kwargs={},
                                module_input_keys={"x", "y"},
                                module_output_keys={"output", "y"},
                            ),
                        }
                    )
                    for _ in range(p2p.pg.size())
                )
            )

            self.loss = PipelineBlock(
                p2p=p2p,
                module_builder=lambda: lambda x: x.sum(),
                module_kwargs={},
                module_input_keys={"x"},
                module_output_keys={"output"},
            )

        def forward(
            self,
            differentiable_tensor: Union[torch.Tensor, TensorPointer],
            non_differentiable_tensor: Union[torch.Tensor, TensorPointer],
        ):
            for non_linear in self.mlp:
                differentiable_tensor = non_linear.linear(input=differentiable_tensor)["output"]
                output = non_linear.activation(x=differentiable_tensor, y=non_differentiable_tensor)
                differentiable_tensor, non_differentiable_tensor = output["output"], output["y"]
            differentiable_tensor = self.loss(x=differentiable_tensor)["output"]
            return differentiable_tensor

    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    has_reference_model = current_pp_rank == reference_rank

    # spawn model
    model = DummyModel(p2p=p2p)
    if has_reference_model:
        reference_model = DummyModel(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size()
    with init_on_device_and_dtype(device):
        for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for non_linear in reference_model.mlp:
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for pp_rank in range(parallel_context.pp_pg.size()):
                non_linear = model.mlp[pp_rank]
                reference_non_linear = reference_model.mlp[pp_rank]
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    reference_non_linear.linear.pp_block.weight.data.copy_(non_linear.linear.pp_block.weight.data)
                    reference_non_linear.linear.pp_block.bias.data.copy_(non_linear.linear.pp_block.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.linear.pp_block.weight.data.copy_(weight.data)
                reference_non_linear.linear.pp_block.bias.data.copy_(bias.data)
    else:
        p2p.send_tensors(
            [model.mlp[current_pp_rank].linear.pp_block.weight, model.mlp[current_pp_rank].linear.pp_block.bias],
            to_rank=reference_rank,
        )

    # Get infinite dummy data iterator
    def dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg: dist.ProcessGroup, dtype=torch.float, input_pp_rank=0
    ):
        micro_batch_size = 3
        # We assume the first linear is always built on the first rank.
        while True:
            yield {
                "differentiable_tensor": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "non_differentiable_tensor": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
            }

    data_iterator = dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg=parallel_context.pp_pg
    )  # First rank receives data

    # Have at least as many microbatches as PP size.
    n_micro_batches_per_batch = parallel_context.pp_pg.size() + 5

    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]

    # Run the model
    losses = []
    for micro_batch in batch:
        with torch.inference_mode():
            loss = model(**micro_batch)
        losses.append(loss)

    # Equivalent on the reference model
    if has_reference_model:
        reference_losses = []
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            reference_losses.append(loss.detach())

    # Gather loss in reference_rank
    if has_reference_model:
        _losses = []
    for loss in losses:
        if isinstance(loss, torch.Tensor):
            if has_reference_model:
                _losses.append(loss)
            else:
                p2p.send_tensors([loss], to_rank=reference_rank)
        else:
            assert isinstance(loss, TensorPointer)
            if not has_reference_model:
                continue
            _losses.append(p2p.recv_tensors(num_tensors=1, from_rank=loss.group_rank)[0])
    if has_reference_model:
        losses = _losses

    # Check loss are the same as reference
    if has_reference_model:
        for loss, ref_loss in zip(losses, reference_losses):
            torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-7)

    parallel_context.destroy()


@pytest.mark.skipif(available_gpus() < 4, reason="Testing `test_pipeline_engine_diamond` requires at least 4 gpus")
@pytest.mark.parametrize(
    "pipeline_engine", [AllForwardAllBackwardPipelineEngine(), OneForwardOneBackwardPipelineEngine()]
)
@rerun_if_address_is_in_use()
def test_pipeline_engine_diamond(pipeline_engine: PipelineEngine):
    init_distributed(pp=4, dp=1, tp=1)(_test_pipeline_engine_diamond)(pipeline_engine=pipeline_engine)
    pass


def _test_pipeline_engine_diamond(parallel_context: ParallelContext, pipeline_engine: PipelineEngine):
    class DiamondModel(nn.Module):
        def __init__(self, p2p: P2P):
            super().__init__()
            self.p2p = p2p
            self.dense_bottom = nn.ModuleDict(
                {
                    "linear": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.Linear,
                        module_kwargs={"in_features": 10, "out_features": 10},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                    "activation": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.ReLU,
                        module_kwargs={},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                }
            )
            self.dense_left = nn.ModuleDict(
                {
                    "linear": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.Linear,
                        module_kwargs={"in_features": 10, "out_features": 10},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                    "activation": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.ReLU,
                        module_kwargs={},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                }
            )
            self.dense_right = nn.ModuleDict(
                {
                    "linear": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.Linear,
                        module_kwargs={"in_features": 10, "out_features": 10},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                    "activation": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.ReLU,
                        module_kwargs={},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                }
            )
            self.dense_top = nn.ModuleDict(
                {
                    "linear": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.Bilinear,
                        module_kwargs={"in1_features": 10, "in2_features": 10, "out_features": 10},
                        module_input_keys={"input1", "input2"},
                        module_output_keys={"output"},
                    ),
                    "activation": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.ReLU,
                        module_kwargs={},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                }
            )

            self.loss = PipelineBlock(
                p2p=p2p,
                module_builder=lambda: lambda x: x.sum(),
                module_kwargs={},
                module_input_keys={"x"},
                module_output_keys={"output"},
            )

        def forward(self, x):
            x = self.dense_bottom.activation(input=self.dense_bottom.linear(input=x)["output"])["output"]
            y = self.dense_left.activation(input=self.dense_left.linear(input=x)["output"])["output"]
            z = self.dense_right.activation(input=self.dense_right.linear(input=x)["output"])["output"]
            out = self.dense_top.activation(input=self.dense_top.linear(input1=y, input2=z)["output"])["output"]
            return self.loss(x=out)["output"]

    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    has_reference_model = current_pp_rank == reference_rank

    # spawn model
    model = DiamondModel(p2p=p2p)
    if has_reference_model:
        reference_model = DiamondModel(p2p=p2p)

    # Set the ranks
    assert parallel_context.pp_pg.size() == len(
        [model.dense_bottom, model.dense_left, model.dense_right, model.dense_top]
    )
    assert parallel_context.pp_pg.size() == 4
    pp_rank_to_dense_name = ["dense_bottom", "dense_left", "dense_right", "dense_top"]
    with init_on_device_and_dtype(device):
        for pp_rank, module_name in enumerate(pp_rank_to_dense_name):
            non_linear = model.get_submodule(module_name)
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for module_name in pp_rank_to_dense_name:
                non_linear = reference_model.get_submodule(module_name)
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for pp_rank, module_name in enumerate(pp_rank_to_dense_name):
                reference_non_linear = reference_model.get_submodule(module_name).linear.pp_block
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    non_linear = model.get_submodule(module_name).linear.pp_block
                    reference_non_linear.weight.data.copy_(non_linear.weight.data)
                    reference_non_linear.bias.data.copy_(non_linear.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.weight.data.copy_(weight.data)
                reference_non_linear.bias.data.copy_(bias.data)
    else:
        non_linear = model.get_submodule(pp_rank_to_dense_name[current_pp_rank]).linear.pp_block
        p2p.send_tensors(
            [non_linear.weight, non_linear.bias],
            to_rank=reference_rank,
        )

    # Get infinite dummy data iterator
    def dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg: dist.ProcessGroup, dtype=torch.float, input_pp_rank=0
    ):
        micro_batch_size = 3
        # We assume the first linear is always built on the first rank.
        while True:
            yield {
                "x": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
            }

    data_iterator = dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg=parallel_context.pp_pg
    )  # First rank receives data

    # Have at least as many microbatches as PP size.
    n_micro_batches_per_batch = parallel_context.pp_pg.size() + 5

    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]
    losses = pipeline_engine.train_batch_iter(
        model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=n_micro_batches_per_batch, grad_accumulator=None
    )

    # Equivalent on the reference model
    if has_reference_model:
        reference_losses = []
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            loss /= n_micro_batches_per_batch
            loss.backward()
            reference_losses.append(loss.detach())

    # Gather loss in reference_rank
    if has_reference_model:
        _losses = []
    for loss in losses:
        if isinstance(loss["loss"], torch.Tensor):
            if has_reference_model:
                _losses.append(loss["loss"])
            else:
                p2p.send_tensors([loss["loss"]], to_rank=reference_rank)
        else:
            assert isinstance(loss["loss"], TensorPointer)
            if not has_reference_model:
                continue
            _losses.append(p2p.recv_tensors(num_tensors=1, from_rank=loss["loss"].group_rank)[0])
    if has_reference_model:
        losses = _losses

    # Check loss are the same as reference
    if has_reference_model:
        for loss, ref_loss in zip(losses, reference_losses):
            torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-7)

    # Check that gradient flows through the entire model
    for param in model.parameters():
        assert param.grad is not None

    # Check that gradient are the same as reference
    if has_reference_model:
        for pp_rank, module_name in enumerate(pp_rank_to_dense_name):
            reference_non_linear = reference_model.get_submodule(module_name).linear.pp_block
            if pp_rank == current_pp_rank:
                # We already have the weights locally
                non_linear = model.get_submodule(module_name).linear.pp_block
                torch.testing.assert_close(
                    non_linear.weight.grad,
                    reference_non_linear.weight.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(
                    non_linear.bias.grad,
                    reference_non_linear.bias.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(weight_grad, reference_non_linear.weight.grad, atol=1e-6, rtol=1e-7)
            torch.testing.assert_close(bias_grad, reference_non_linear.bias.grad, atol=1e-6, rtol=1e-7)
    else:
        non_linear = model.get_submodule(pp_rank_to_dense_name[current_pp_rank]).linear.pp_block
        p2p.send_tensors(
            [non_linear.weight.grad, non_linear.bias.grad],
            to_rank=reference_rank,
        )

    parallel_context.destroy()
