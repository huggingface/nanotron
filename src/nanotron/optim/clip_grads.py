from typing import Iterable, Optional, Tuple

import torch

import nanotron.distributed as dist
from nanotron import logging
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.parameters import NanotronParameter, get_grad_from_parameter

logger = logging.get_logger(__name__)


@torch.no_grad()
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
    assert len(named_parameters) > 0, "There is no parameters to clip. Something has gone wrong."

    world_rank = dist.get_rank()

    # assert that all params require grad
    # for _, p in named_parameters:
    #     assert p.requires_grad, "clip_grad_norm_ only supports Tensors that require grad"

    if grad_accumulator is None:
        # grads = [
        #     p.grad for _, p in named_parameters if not p.is_tied or world_rank == p.get_tied_info().global_ranks[0]
        # ]

        # grads = [
        #     get_grad_from_parameter(p)
        #     for _, p in named_parameters
        #     if not p.is_tied or world_rank == p.get_tied_info().global_ranks[0]
        # ]
        # fp8_param_and_fp32_grads = []
        from nanotron import constants
        from nanotron.fp8.tensor import FP8Tensor
        from nanotron.helpers import get_accum_grad

        grads = []
        for n, p in named_parameters:
            if p.data.__class__ == FP8Tensor:
                from nanotron.fp8.tensor import convert_tensor_from_fp8

                if constants.CONFIG.fp8.is_directly_keep_accum_grad_of_fp8 is True:
                    _g = get_accum_grad(n)
                    g = _g
                else:
                    _g = get_grad_from_parameter(p)
                    g = convert_tensor_from_fp8(_g, _g.fp8_meta, torch.float32)
                grads.append(g)
                # fp8_param_and_fp32_grads.append((p, g))
            else:
                grads.append(get_grad_from_parameter(p))
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
    clip_coef = max_norm / (total_norm + constants.CONFIG.fp8.gradient_clipping_eps)
    # print(f"using constants.CONFIG.fp8.gradient_clipping_eps: {constants.CONFIG.fp8.gradient_clipping_eps}")
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    assert 1 == 1

    # devices = {
    #     param.grad.device if grad_accumulator is None else grad_accumulator.get_grad_buffer(name).device
    #     for name, param in named_parameters
    # }
    # devices = {
    #     get_grad_from_parameter(param).device
    #     if grad_accumulator is None
    #     else grad_accumulator.get_grad_buffer(name).device
    #     for name, param in named_parameters
    # }
    # device_to_clip_coef_clamped = {device: clip_coef_clamped.to(device) for device in devices}

    # for name, param in named_parameters:
    #     if grad_accumulator is None:
    #         # param.grad.detach().mul_(device_to_clip_coef_clamped[param.grad.device])
    #         get_grad_from_parameter(param).mul_(device_to_clip_coef_clamped[get_grad_from_parameter(param).device])
    #         assert 1 == 1
    #     else:
    #         grad_accumulator.get_grad_buffer(name).mul_(
    #             device_to_clip_coef_clamped[grad_accumulator.get_grad_buffer(name).device]
    #         )
    for name, param in named_parameters:
        if grad_accumulator is None:
            # param.grad.detach().mul_(device_to_clip_coef_clamped[param.grad.device])
            # get_grad_from_parameter(param).mul_(clip_coef_clamped)
            if param.data.__class__ == FP8Tensor:
                if constants.CONFIG.fp8.is_directly_keep_accum_grad_of_fp8 is True:
                    get_accum_grad(name).mul_(clip_coef_clamped)
                else:
                    from nanotron.parallel.parameters import get_data_from_param

                    _g = get_grad_from_parameter(param)
                    fp32_grad = convert_tensor_from_fp8(_g, _g.fp8_meta, torch.float32)
                    fp32_grad.mul_(clip_coef_clamped)
                    clipped_fp8_grad = FP8Tensor.from_metadata(fp32_grad, _g.fp8_meta)
                    p_data = get_data_from_param(param)
                    p_data.grad = clipped_fp8_grad

                    assert get_grad_from_parameter(param) is clipped_fp8_grad
                    # get_grad_from_parameter(param)
                    assert 1 == 1
            else:
                get_grad_from_parameter(param).mul_(clip_coef_clamped)
        else:
            raise NotImplementedError
            # grad_accumulator.get_grad_buffer(name).mul_(
            #     device_to_clip_coef_clamped[grad_accumulator.get_grad_buffer(name).device]
            # )

    # NOTE: copy the fp32 grads back to the fp8 grads
    # assert 1 == 1

    return total_norm
