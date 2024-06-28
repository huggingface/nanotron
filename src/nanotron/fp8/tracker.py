from typing import cast

import torch

from nanotron.fp8.linear import FP8Linear, FP8LinearMeta
from nanotron.fp8.tensor import update_scaling_factor


class _ScalingTracker(torch.autograd.Function):
    """A tracker that records metadata of a tensor during training, and dynamically quantizes the tensor based on the metadata."""

    @staticmethod
    def forward(ctx, module: FP8Linear, interval: int) -> FP8Linear:
        assert interval > 0

        # TODO(xrsrke): move this out tracker or not?
        ctx.interval = interval
        ctx.fp8_meta = module.fp8_meta
        return module

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        fp8_meta = cast(ctx.fp8_meta, FP8LinearMeta)
        fp8_meta.input_grad.amax_history.append(fp8_meta.input_grad.amax)
        fp8_meta.weight_grad.amax_history.append(fp8_meta.weight_grad.amax)
        fp8_meta.output_grad.amax_history.append(fp8_meta.output_grad.amax)
        return grad_output


# def track(module: FP8Linear, interval: int) -> FP8Linear:
#     # TODO(xrsrke): add recursive
#     return _ScalingTracker.apply(module, interval)


class DynamicScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module: FP8Linear):
        ctx.fp8_meta = module.fp8_meta
        return _ScalingTracker.apply(module, interval=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # TODO(xrsrke): trigger scaling
        fp8_meta: FP8LinearMeta = cast(ctx.fp8_meta, FP8LinearMeta)

        # TODO(xrsrke): move this to scaling strategy
        amax = torch.max(fp8_meta.input_grad.amax_history, dim=-1)
        prev_scale = fp8_meta.input_grad.scale
        # TODO(xrsrke): retrieve margin from fp8recipe
        scale = update_scaling_factor(amax, prev_scale, fp8_meta.input_grad.dtype)
        fp8_meta.input_grad.scale = scale
        return grad_output
