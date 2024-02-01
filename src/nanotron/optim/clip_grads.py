from typing import Iterable, Optional, Tuple

import torch

import nanotron.distributed as dist
from nanotron import logging
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.parameters import NanotronParameter

logger = logging.get_logger(__name__)


def clip_grad_norm(
    mp_pg: dist.ProcessGroup,
    named_parameters: Iterable[Tuple[str, NanotronParameter]],
    max_norm: float,
    grad_accumulator: Optional[GradientAccumulator],
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Clips gradients. Adapted from torch.nn.utils.clip_grad_norm_.
    Norms are computed in fp32 precision to retain most accuracy.

    Args:
        mp_pg (dist.ProcessGroup): Process group for model parallel, ie all the ranks part of the same model replica (TP x PP)
        named_parameters (Iterable[(str, Parameter)]): an iterable of named Parameters that will have gradients normalized.
        grad_accumulator (GradientAccumulator): grad accumulator. If not None, in case of Zero1, we need to clip all fp32 grads
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

    .. note:: In case parameters contains tied weights, we keep only a single copy of the gradient, but modify the
        gradient of all tied weights.
    """
    named_parameters = list(named_parameters)
    world_rank = dist.get_rank()

    # assert that all params require grad
    for _, p in named_parameters:
        assert p.requires_grad, "clip_grad_norm_ only supports Tensors that require grad"

    if grad_accumulator is None:
        grads = [
            p.grad for _, p in named_parameters if not p.is_tied or world_rank == p.get_tied_info().global_ranks[0]
        ]
    else:
        # In case of FP32 Grad Accum, We need to clip all fp32 grads
        grads = [
            grad_accumulator.get_grad_buffer(name)
            for name, p in named_parameters
            if not p.is_tied or world_rank == p.get_tied_info().global_ranks[0]
        ]

    # Calculate gradient norm
    if norm_type == torch.inf:
        if len(grads) > 0:
            total_norm = torch.max(
                torch.stack([torch.linalg.vector_norm(g.detach(), ord=torch.inf, dtype=torch.float) for g in grads])
            )
        else:
            total_norm = torch.zeros([], dtype=torch.float, device=torch.device("cuda"))
        dist.all_reduce(total_norm, group=mp_pg, op=dist.ReduceOp.MAX)

    else:
        if len(grads) > 0:
            # TODO @nouamanetazi: Check if we should calculate norm per parameter (remove .pow(norm_type)
            total_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g.detach(), ord=norm_type, dtype=torch.float) for g in grads]),
                ord=norm_type,
                dtype=torch.float,
            ).pow(norm_type)
        else:
            total_norm = torch.zeros([], dtype=torch.float, device=torch.device("cuda"))
        dist.all_reduce(total_norm, group=mp_pg, op=dist.ReduceOp.SUM)
        total_norm.pow_(1.0 / norm_type)

    # Scale gradients
    clip_coef = max_norm / (total_norm + 1.0e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    devices = {
        param.grad.device if grad_accumulator is None else grad_accumulator.get_grad_buffer(name).device
        for name, param in named_parameters
    }
    device_to_clip_coef_clamped = {device: clip_coef_clamped.to(device) for device in devices}

    for name, param in named_parameters:
        if grad_accumulator is None:
            param.grad.detach().mul_(device_to_clip_coef_clamped[param.grad.device])
        else:
            grad_accumulator.get_grad_buffer(name).detach().mul_(
                device_to_clip_coef_clamped[grad_accumulator.get_grad_buffer(name).device]
            )

    return total_norm
