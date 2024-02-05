from typing import Optional, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
import transformer_engine as te  # noqa
from torch import nn

from nanotron.fp8.constants import DTypes
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor


class FP8LinearMeta(TypedDict):
    """FP8 metadata for FP8Linear."""

    input_grad: FP8Meta
    weight_grad: FP8Meta
    output_grad: FP8Meta


class FP8Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Optional[torch.device] = None):
        super().__init__(in_features, out_features, bias, device)
        # TODO(xrsrke): add device, and 2 fp8 dtypes
        if self.weight.device != torch.device("cpu"):
            self.weight = FP8Parameter(self.weight, dtype=DTypes.FP8E4M3)

            # NOTE: quantization metadata for input gradients, weight gradients
            # TODO(xrsrke): don't fixed this
            self.fp8_meta: FP8LinearMeta = {
                # kfloat8_e4m3
                "input_grad": FP8Meta(amax=1, dtype=DTypes.FP8E4M3, inverse_scale=1),
                "weight_grad": FP8Meta(amax=1, dtype=DTypes.FP8E4M3, inverse_scale=1),
                # kfloat8_e5m2
                "output_grad": FP8Meta(amax=1, dtype=DTypes.FP8E5M2, inverse_scale=1),
            }

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> FP8Tensor:
        # NOTE: only do fp8 kernel if both input and weight are on CUDA device
        if input.device == torch.device("cpu") or self.weight.device == torch.device("cpu"):
            return F.linear(input, self.weight, self.bias)

        # NOTE: just a phony tensor to make pytorch trigger the backward pass
        # because weight and bias's requires grad are set to False
        # so that we can compute the gradients using fp8 kernels by ourselves
        phony = torch.empty(0, device=input.device, requires_grad=True)
        output, _ = _FP8MatMul.apply(input, self.weight, self.fp8_meta, phony)

        output = output if self.bias is None else output + self.bias
        return output


class _FP8MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input: FP8Tensor, weight: FP8Tensor, fp8_meta: FP8LinearMeta, phony: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            if type(input) == torch.Tensor:
                input = FP8Tensor(input, dtype=DTypes.FP8E4M3)
                print(f"_FP8Matmul.forward, after casting input to DTypes.FP8E4M3, input.fp8_meta: {input._fp8_meta}")

        ctx.save_for_backward(input, weight)
        ctx.fp8_meta = fp8_meta

        with torch.no_grad():
            # NOTE: pass FP8Tensor instead of FP8Parameter

            print(f"_FP8Matmul.forward, before matmul, weight.data._fp8_meta: {weight.data._fp8_meta}")

            output = fp8_matmul_kernel(
                mat_a=weight.data, transpose_a=True, mat_b=input, transpose_b=False, use_split_accumulator=False
            )

        return output, phony

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        ∂L/∂X = ∂L/∂Y @ Wᵀ
        ∂L/∂W = Xᵀ @ ∂L/∂Y
        Source: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        """
        # grad_output = grad_output.contiguous()
        input, weight = ctx.saved_tensors

        with torch.no_grad():
            if type(grad_output) == torch.Tensor:
                grad_output = torch.ones_like(grad_output)
                grad_output = grad_output.contiguous()
                grad_output = FP8Tensor(grad_output, dtype=DTypes.FP8E5M2)

        grad_input = fp8_matmul_kernel(
            mat_a=grad_output, transpose_a=True, mat_b=weight, transpose_b=True, use_split_accumulator=True
        )

        grad_weight = fp8_matmul_kernel(
            mat_a=input, transpose_a=False, mat_b=grad_output, transpose_b=False, use_split_accumulator=True
        )

        weight.grad = grad_weight

        return grad_input, None, None, None
