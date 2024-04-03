from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
import torch
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron import distributed as dist
from helpers.dummy import DummyModel, dummy_infinite_data_loader
from nanotron.models import init_on_device_and_dtype
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron import logging
import sys
import logging as lg

logger = logging.get_logger(__name__)


def set_logger_verbosity_format(logging_level: str, parallel_context: "ParallelContext"):
    formatter = lg.Formatter(
        fmt=f"%(asctime)s [%(levelname)s|DP={dist.get_rank(parallel_context.dp_pg)}|PP={dist.get_rank(parallel_context.pp_pg)}|TP={dist.get_rank(parallel_context.tp_pg)}]: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    log_level = logging.log_levels[logging_level]

    # main root logger
    root_logger = logging.get_logger()
    root_logger.setLevel(log_level)
    handler = logging.NewLineStreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Brrr
    logging.set_verbosity(log_level)
    logging.set_formatter(formatter=formatter)

@rerun_if_address_is_in_use()
def test_pipeline_engine(pp: int):
    init_distributed(tp=1, dp=1, pp=pp)(_test_pipeline_engine)(pipeline_engine=AllForwardAllBackwardPipelineEngine())


def _test_pipeline_engine(parallel_context, pipeline_engine):
    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    has_reference_model = dist.get_rank(parallel_context.pp_pg) == reference_rank
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    
    # Set log levels
    # if dist.get_rank(parallel_context.world_pg) == 0:
    #     set_logger_verbosity_format("debug", parallel_context=parallel_context)
    #     set_logger_verbosity_format("debug", parallel_context=parallel_context)
        
    
    set_logger_verbosity_format("debug", parallel_context=parallel_context)
    set_logger_verbosity_format("debug", parallel_context=parallel_context)
   
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

if __name__ == "__main__":
    test_pipeline_engine(pp=2)