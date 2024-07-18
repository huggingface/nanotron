from dataclasses import dataclass
from typing import Optional, Tuple, Union, cast

import torch
import transformer_engine as te  # noqa
from torch import nn

from nanotron.fp8.constants import FP8LM_LINEAR_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.recipe import FP8LinearRecipe
from nanotron.fp8.tensor import FP8Tensor
from nanotron.parallel.parameters import get_data_from_param


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
        # accum_qtype: DTypes = FP8LM_RECIPE.linear.accum_dtype,
        recipe: FP8LinearRecipe = FP8LM_LINEAR_RECIPE,
        # NOTE: placeholder for dtype in torch's nn.Linear
        # TODO(xrsrke): remove this shit
        **kwargs,
    ):
        """
        Args:
            qtype (DTypes, optional): This is accumulation precision dtype
        """

        assert device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"
        # assert accum_qtype in DTypes

        # TODO(xrsrke): take initialization dtype from recipe
        super().__init__(in_features, out_features, bias, device, dtype=QTYPE_TO_DTYPE[recipe.accum_dtype])
        # TODO(xrsrke): don't fixed dtype, take it from the FP8 recipe
        # DTypes.FP8E4M3
        weight_data = self.weight.data
        orig_w_shape = weight_data.shape
        weight_data = weight_data.contiguous().view(-1).contiguous().reshape(orig_w_shape)
        quant_w = FP8Parameter(weight_data, dtype=recipe.weight.dtype, interval=recipe.weight.interval)
        assert quant_w.dtype in [torch.uint8, torch.int8], f"got {self.weight.data.dtype}"
        self.weight = quant_w
        # assert self.weight.data.orig_data.abs().max() == quant_w.fp8_meta.amax

        assert self.weight.data.dtype in [torch.uint8, torch.int8], f"got {self.weight.data.dtype}"
        self.metadatas = FP8LinearMeta()
        # self.accum_qtype = accum_qtype
        self.recipe = recipe

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> torch.Tensor:
        import nanotron.fp8.functional as F

        # return F.linear(
        #     input=input, weight=self.weight.data, bias=self.bias, metadatas=self.metadatas, recipe=self.recipe
        # )
        return F.linear(
            input=input,
            weight=get_data_from_param(self.weight),
            bias=get_data_from_param(self.bias),
            metadatas=self.metadatas,
            recipe=self.recipe,
        )

    def __repr__(self) -> str:
        return f"FP8{super().__repr__()}"


# NOTE: original version
class _FP8Matmul(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        input: Union[FP8Tensor, torch.Tensor],
        weight: FP8Tensor,
        output: torch.Tensor,
        phony: torch.Tensor,
        metadatas: FP8LinearMeta,
        # accum_qtype: DTypes,
        recipe: FP8LinearRecipe,
        name,
    ) -> torch.Tensor:
        assert not isinstance(input, FP8Tensor)

        orig_input_shape = input.shape
        input = input.contiguous().view(-1).contiguous().view(orig_input_shape)

        if metadatas.input is None:
            fp8_input = FP8Tensor(
                input,
                dtype=recipe.input.dtype,
                interval=recipe.input.interval,
            )
            metadatas.input = fp8_input.fp8_meta
        else:
            fp8_input = FP8Tensor.from_metadata(input, metadatas.input)

        # ctx.accum_qtype = accum_qtype
        ctx.save_for_backward(fp8_input, weight)
        ctx.metadatas = metadatas
        ctx.name = name
        ctx.recipe = recipe

        accum_output = output.contiguous()
        # accum_output = torch.zeros(output.shape, dtype=torch.float16, device="cuda")

        assert fp8_input.data.is_contiguous()
        assert weight.data.is_contiguous()
        assert accum_output.is_contiguous()

        output = fp8_matmul_kernel(
            # NOTE: that works
            mat_a=weight,
            transpose_a=True,
            mat_b=fp8_input,
            transpose_b=False,
            output=accum_output,
            use_split_accumulator=recipe.split_accumulator.output,
            accumulate=recipe.accumulate.output,
            accum_qtype=recipe.accum_dtype,
            recipe=recipe,
        )
        return output, phony

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        ∂L/∂X = ∂L/∂Y @ Wᵀ
        ∂L/∂W = Xᵀ @ ∂L/∂Y
        Reference: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        """
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        fp8_input, fp8_weight = ctx.saved_tensors
        recipe = ctx.recipe
        recipe = cast(FP8LinearRecipe, recipe)
        # accum_qtype = ctx.accum_qtype

        fp8_input = cast(FP8Tensor, fp8_input)
        fp8_weight = cast(FP8Tensor, fp8_weight)
        grad_output = grad_output.contiguous()

        ctx.metadatas = cast(FP8LinearMeta, ctx.metadatas)
        if ctx.metadatas.input_grad is None:
            fp8_grad_output = FP8Tensor(
                grad_output,
                dtype=recipe.input_grad.dtype,
                interval=recipe.input_grad.interval,
            )
            ctx.metadatas.input_grad = fp8_grad_output.fp8_meta
        else:
            fp8_grad_output = FP8Tensor.from_metadata(grad_output, ctx.metadatas.input_grad)

        transposed_fp8_weight = fp8_weight.transpose_fp8()

        grad_input_temp = torch.zeros(
            fp8_grad_output.shape[0],
            transposed_fp8_weight.shape[0],
            device="cuda",
            dtype=QTYPE_TO_DTYPE[recipe.accum_dtype],
        )
        grad_input = fp8_matmul_kernel(
            mat_a=transposed_fp8_weight,
            transpose_a=True,
            mat_b=fp8_grad_output,
            transpose_b=False,
            output=grad_input_temp,
            use_split_accumulator=recipe.split_accumulator.input_grad,
            accum_qtype=recipe.accum_dtype,
            accumulate=recipe.accumulate.input_grad,
            is_backward=True,
            recipe=recipe,
        )

        transposed_fp8_grad_output = fp8_grad_output.transpose_fp8()
        transposed_fp8_input = fp8_input.transpose_fp8()

        grad_weight_temp = torch.zeros(
            transposed_fp8_input.shape[0],
            transposed_fp8_grad_output.shape[0],
            device="cuda",
            dtype=QTYPE_TO_DTYPE[recipe.accum_dtype],
        )
        grad_weight = fp8_matmul_kernel(
            mat_a=transposed_fp8_input,
            transpose_a=True,
            mat_b=transposed_fp8_grad_output,
            transpose_b=False,
            output=grad_weight_temp,
            use_split_accumulator=recipe.split_accumulator.weight_grad,
            accumulate=recipe.accumulate.weight_grad,
            accum_qtype=recipe.accum_dtype,
            # is_backward=True
            recipe=recipe,
        )

        assert grad_input.dtype == QTYPE_TO_DTYPE[recipe.accum_dtype]
        assert grad_weight.dtype == QTYPE_TO_DTYPE[recipe.accum_dtype]
        # TODO(xrsrke): maintain a persistence metadata across training

        grad_weight = grad_weight.T.contiguous()
        orig_shape = grad_weight.shape
        grad_weight = grad_weight.contiguous().t().contiguous().view(-1).contiguous().reshape(orig_shape)

        if ctx.metadatas.weight_grad is None:
            fp8_weight_grad = FP8Tensor(
                grad_weight,
                dtype=recipe.weight_grad.dtype,
                interval=recipe.weight_grad.interval,
            )
            ctx.metadatas.weight_grad = fp8_weight_grad.fp8_meta
        else:
            fp8_weight_grad = FP8Tensor.from_metadata(grad_weight, ctx.metadatas.weight_grad)

        fp8_weight.grad = fp8_weight_grad

        # NOTE: sanity check
        assert isinstance(fp8_weight.grad, FP8Tensor)
        return grad_input.contiguous(), None, None, None, None, None, None
