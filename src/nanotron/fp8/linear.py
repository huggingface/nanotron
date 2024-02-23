from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformer_engine as te  # noqa
from torch import nn

from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor
from nanotron.fp8.constants import FP8LM_RECIPE

import pydevd

class FP8Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Optional[torch.device] = None):
        assert device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"
        super().__init__(in_features, out_features, bias, device)
        # TODO(xrsrke): don't fixed dtype, take it from the FP8 recipe
        # DTypes.FP8E4M3
        self.weight = FP8Parameter(self.weight, dtype=FP8LM_RECIPE.linear.weight.dtype)

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> torch.Tensor:
        # NOTE: only do fp8 kernel if both input and weight are on CUDA device
        if input.device == torch.device("cpu") or self.weight.device == torch.device("cpu"):
            return F.linear(input, self.weight, self.bias)

        # NOTE: just a phony tensor to make pytorch trigger the backward pass
        # because weight and bias's requires_grad are set to False
        # so that we can compute the gradients using the fp8 kernels by ourselves
        phony = torch.empty(0, device=input.device, requires_grad=True)
        output, _ = _FP8Matmul.apply(input, self.weight, phony)

        # TODO(xrsrke): add support for adding bias in fp8
        # TODO(xrsrke): support return an fp8 tensor as output
        # since we will quantize it back to FP8 anyway in the next linear
        output = output if self.bias is None else output + self.bias
        return output


class _FP8Matmul(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, input: Union[FP8Tensor, torch.Tensor], weight: FP8Tensor, phony: torch.Tensor) -> torch.Tensor:
        if type(input) == torch.Tensor:
            input = FP8Tensor(input, dtype=FP8LM_RECIPE.linear.input.dtype)

        ctx.grad_metadata = weight.fp8_grad_meta
        ctx.save_for_backward(input, weight)

        # NOTE: pass FP8Tensor instead of FP8Parameter
        output = fp8_matmul_kernel(
            mat_a=weight.data, transpose_a=True, mat_b=input, transpose_b=False, use_split_accumulator=False
        )

        return output, phony

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        ∂L/∂X = ∂L/∂Y @ Wᵀ
        ∂L/∂W = Xᵀ @ ∂L/∂Y
        Source: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        """
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        
        # TODO(xrsrke): investigate how does grad_output.contiguous() affect the outputs
        input, weight = ctx.saved_tensors

        # TODO(xrsrke): remove fixed grad_output
        if type(grad_output) == torch.Tensor:
            grad_output = torch.ones_like(grad_output)
            grad_output = grad_output.contiguous()
            # DTypes.FP8E5M2
            grad_output = FP8Tensor(grad_output, dtype=FP8LM_RECIPE.linear.output_grad.dtype)

        grad_input = fp8_matmul_kernel(
            mat_a=grad_output, transpose_a=True, mat_b=weight, transpose_b=True, use_split_accumulator=True
        )
        grad_weight = fp8_matmul_kernel(
            mat_a=input, transpose_a=False, mat_b=grad_output, transpose_b=False, use_split_accumulator=True
        )
        weight.grad = FP8Tensor(grad_weight, dtype=FP8LM_RECIPE.linear.weight_grad.dtype)

        return grad_input, None, None
