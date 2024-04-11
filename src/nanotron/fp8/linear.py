from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex
from torch import nn
import pydevd

from nanotron.fp8.constants import FP8LM_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor


class FP8Linear(nn.Linear):
    # TODO(xrsrke): qtype isn't the data types of the weight and bias
    # but the accumulation precision dtype
    # chanege it to accum_dtype
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        accum_qtype: DTypes = FP8LM_RECIPE.linear.accum_dtype,
    ):
        """
        Args:
            qtype (DTypes, optional): This is accumulation precision dtype
        """

        assert device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"
        assert accum_qtype in DTypes

        super().__init__(in_features, out_features, bias, device, QTYPE_TO_DTYPE[accum_qtype])
        # TODO(xrsrke): don't fixed dtype, take it from the FP8 recipe
        # DTypes.FP8E4M3
        self.weight = FP8Parameter(
            self.weight, dtype=FP8LM_RECIPE.linear.weight.dtype, interval=FP8LM_RECIPE.linear.weight.interval
        )
        self.accum_qtype = accum_qtype

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> torch.Tensor:
        # NOTE: only do fp8 kernel if both input and weight are on CUDA device
        if input.device == torch.device("cpu") or self.weight.device == torch.device("cpu"):
            # TODO(xrsrke): adjust the accumation precision
            return F.linear(input, self.weight, self.bias)

        # if self.name == "transformer.h.0.self_attention.query_key_value":
        #     assert 1 == 1

        from einops import rearrange

        seq_len = None
        batch_size = None
        is_input_flat = False
        if input.ndim == 3:
            batch_size = input.shape[0]
            seq_len = input.shape[1]
            is_input_flat = True
            input = rearrange(input, "b n h -> (b n) h")
        elif input.ndim > 3:
            raise ValueError(f"Unsupported input shape: {input.shape}")

        # NOTE: just a phony tensor to make pytorch trigger the backward pass
        # because weight and bias's requires_grad are set to False
        # so that we can compute the gradients using the fp8 kernels by ourselves
        phony = torch.empty(0, device=input.device, requires_grad=True)
        # print(f"name={self.name}")
        output, _ = _FP8Matmul.apply(input, self.weight, phony, self.accum_qtype)

        # TODO(xrsrke): add support for adding bias in fp8
        # TODO(xrsrke): support return an fp8 tensor as output
        # since we will quantize it back to FP8 anyway in the next linear
        output = output if self.bias is None else output + self.bias
        output = rearrange(output, "(b n) h -> b n h", n=seq_len, b=batch_size) if is_input_flat is True else output
        return output


class _FP8Matmul(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(
        ctx, input: Union[FP8Tensor, torch.Tensor], weight: FP8Tensor, phony: torch.Tensor, accum_qtype: DTypes
    ) -> torch.Tensor:
        if type(input) == torch.Tensor:
            input = FP8Tensor(
                input,
                dtype=FP8LM_RECIPE.linear.input.dtype,
                interval=FP8LM_RECIPE.linear.input.interval,
                is_dynamic_scaling=FP8LM_RECIPE.linear.input.is_dynamic_scaling,
            )

        ctx.accum_qtype = accum_qtype
        ctx.save_for_backward(input, weight)

        # NOTE: pass FP8Tensor instead of FP8Parameter
        output = fp8_matmul_kernel(
            # NOTE: that works
            mat_a=weight.data,
            transpose_a=True,
            mat_b=input,
            transpose_b=False,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.output,
            accum_qtype=accum_qtype,
        )

        # output = _fp8_matmul_kernel_2(
        #     # NOTE: that works
        #     mat_a=input,
        #     transpose_a=False,
        #     mat_b=weight.data,
        #     transpose_b=True,
        #     use_split_accumulator=False,
        #     accum_qtype=accum_qtype,
        # )

        return output, phony

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        ∂L/∂X = ∂L/∂Y @ Wᵀ
        ∂L/∂W = Xᵀ @ ∂L/∂Y
        Reference: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        """
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input, weight = ctx.saved_tensors
        accum_qtype = ctx.accum_qtype

        if type(grad_output) == torch.Tensor:
            # TODO(xrsrke): investigate how does grad_output.contiguous() affect the outputs
            grad_output = grad_output.contiguous()
            grad_output = FP8Tensor(grad_output, dtype=FP8LM_RECIPE.linear.output_grad.dtype)

        # TODO(xrsrke): extract use_split_accumulator from FP8Recipe
        # grad_input = fp8_matmul_kernel(
        #     mat_a=grad_output,
        #     transpose_a=True,
        #     mat_b=weight,
        #     transpose_b=True,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        #     is_backward=True
        # )
        # grad_weight = fp8_matmul_kernel(
        #     mat_a=input,
        #     transpose_a=False,
        #     mat_b=grad_output,
        #     transpose_b=False,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        #     is_backward=True
        # )

        grad_output_transposed = tex.fp8_transpose(grad_output, grad_output.fp8_meta.te_dtype)
        grad_output_transposed.fp8_meta = grad_output.fp8_meta
        weight_transposed = tex.fp8_transpose(weight, weight.fp8_meta.te_dtype)
        weight_transposed.fp8_meta = weight.fp8_meta

        grad_input = fp8_matmul_kernel(
            mat_a=weight_transposed,
            transpose_a=True,
            mat_b=grad_output,
            transpose_b=False,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.input_grad,
            accum_qtype=accum_qtype,
            # is_backward=True
        )

        input_tranposed = tex.fp8_transpose(input, input.fp8_meta.te_dtype)
        input_tranposed.fp8_meta = input.fp8_meta

        grad_weight = fp8_matmul_kernel(
            mat_a=input_tranposed,
            transpose_a=True,
            mat_b=grad_output_transposed,
            transpose_b=False,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.weight_grad,
            accum_qtype=accum_qtype,
            # is_backward=True
        )

        # grad_input = _fp8_matmul_kernel_2(
        #     mat_a=grad_output,
        #     transpose_a=True,
        #     mat_b=weight,
        #     transpose_b=True,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        # )
        # grad_weight = _fp8_matmul_kernel_2(
        #     mat_a=input,
        #     transpose_a=False,
        #     mat_b=grad_output,
        #     transpose_b=False,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        # )

        # grad_input = _fp8_matmul_kernel(
        #     mat_a=grad_output,
        #     transpose_a=False,
        #     mat_b=weight,
        #     transpose_b=True,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        # )
        # grad_weight = _fp8_matmul_kernel(
        #     mat_a=input,
        #     transpose_a=True,
        #     mat_b=grad_output,
        #     transpose_b=False,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        # )

        assert grad_input.dtype == QTYPE_TO_DTYPE[accum_qtype]
        assert grad_weight.dtype == QTYPE_TO_DTYPE[accum_qtype]
        # TODO(xrsrke): maintain a persistence metadata across training
        weight.grad = FP8Tensor(grad_weight, dtype=FP8LM_RECIPE.linear.weight_grad.dtype)
        # NOTE: sanity check
        assert isinstance(weight.grad, FP8Tensor)
        return grad_input, None, None, None
