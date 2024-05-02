from dataclasses import dataclass
from typing import Optional, Tuple, Union, cast

import torch
import transformer_engine as te  # noqa
from torch import nn

from nanotron.fp8.constants import FP8LM_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor


@dataclass
class FP8LinearMeta:
    """FP8 metadata for FP8Linear."""

    input: Optional[FP8Meta] = None
    weight: Optional[FP8Meta] = None
    input_grad: Optional[FP8Meta] = None
    weight_grad: Optional[FP8Meta] = None
    # output_grad: Optional[FP8Meta] = None


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
        self.metadatas = FP8LinearMeta()
        self.accum_qtype = accum_qtype

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> torch.Tensor:
        assert input.device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"

        # # NOTE: only do fp8 kernel if both input and weight are on CUDA device
        # if input.device == torch.device("cpu") or self.weight.device == torch.device("cpu"):
        #     # TODO(xrsrke): adjust the accumation precision
        #     return F.linear(input, self.weight, self.bias)

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
        output, _ = _FP8Matmul.apply(input, self.weight, phony, self.metadatas, self.accum_qtype)

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
        ctx,
        input: Union[FP8Tensor, torch.Tensor],
        weight: FP8Tensor,
        phony: torch.Tensor,
        metadatas: FP8LinearMeta,
        accum_qtype: DTypes,
    ) -> torch.Tensor:
        assert not isinstance(input, FP8Tensor)

        if metadatas.input is None:
            fp8_input = FP8Tensor(
                input,
                dtype=FP8LM_RECIPE.linear.input.dtype,
                interval=FP8LM_RECIPE.linear.input.interval,
                # is_delayed_scaling=FP8LM_RECIPE.linear.input.is_delayed_scaling,
            )
            metadatas.input = fp8_input.fp8_meta
        else:
            fp8_input = FP8Tensor.from_metadata(input, metadatas.input)

        ctx.accum_qtype = accum_qtype
        ctx.metadatas = metadatas
        ctx.save_for_backward(fp8_input, weight.data)

        # NOTE: pass FP8Tensor instead of FP8Parameter
        output = fp8_matmul_kernel(
            # NOTE: that works
            mat_a=weight.data,
            transpose_a=True,
            mat_b=fp8_input,
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
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        fp8_input, fp8_weight = ctx.saved_tensors
        accum_qtype = ctx.accum_qtype

        fp8_input = cast(FP8Tensor, fp8_input)
        fp8_weight = cast(FP8Tensor, fp8_weight)
        grad_output = grad_output.contiguous()

        ctx.metadatas = cast(FP8LinearMeta, ctx.metadatas)
        if ctx.metadatas.input_grad is None:
            fp8_grad_output = FP8Tensor(
                grad_output,
                dtype=FP8LM_RECIPE.linear.input_grad.dtype,
                interval=FP8LM_RECIPE.linear.input_grad.interval,
            )
            ctx.metadatas.input_grad = fp8_grad_output.fp8_meta
        else:
            fp8_grad_output = FP8Tensor.from_metadata(grad_output, ctx.metadatas.input_grad)

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

        # fp8_weight_transposed = tex.fp8_transpose(fp8_weight, fp8_weight.fp8_meta.te_dtype)
        # fp8_weight_transposed.fp8_meta = fp8_weight.fp8_meta
        transposed_fp8_weight = fp8_weight.transpose_fp8()

        grad_input = fp8_matmul_kernel(
            mat_a=transposed_fp8_weight,
            transpose_a=True,
            mat_b=fp8_grad_output,
            transpose_b=False,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.input_grad,
            accum_qtype=accum_qtype,
            is_backward=True,
        )

        # fp8_grad_output_transposed = tex.fp8_transpose(fp8_grad_output, fp8_grad_output.fp8_meta.te_dtype)
        # fp8_grad_output_transposed.fp8_meta = fp8_grad_output.fp8_meta
        # fp8_input_tranposed = tex.fp8_transpose(fp8_input, fp8_input.fp8_meta.te_dtype)
        # fp8_input_tranposed.fp8_meta = fp8_input.fp8_meta

        transposed_fp8_grad_output = fp8_grad_output.transpose_fp8()
        transposed_fp8_input = fp8_input.transpose_fp8()

        grad_weight = fp8_matmul_kernel(
            mat_a=transposed_fp8_input,
            transpose_a=True,
            mat_b=transposed_fp8_grad_output,
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

        if ctx.metadatas.weight_grad is None:
            fp8_weight_grad = FP8Tensor(
                grad_weight,
                dtype=FP8LM_RECIPE.linear.weight_grad.dtype,
                interval=FP8LM_RECIPE.linear.weight_grad.interval,
            )
            ctx.metadatas.weight_grad = fp8_weight_grad.fp8_meta
        else:
            fp8_weight_grad = FP8Tensor.from_metadata(grad_weight, ctx.metadatas.weight_grad)

        fp8_weight.grad = fp8_weight_grad
        # NOTE: sanity check
        assert isinstance(fp8_weight.grad, FP8Tensor)
        return grad_input, None, None, None, None
