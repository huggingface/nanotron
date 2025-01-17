import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, cast

import torch
from torch import nn

from nanotron.fp8.constants import FP8LM_LINEAR_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.recipe import FP8LinearRecipe
from nanotron.fp8.tensor import FP8Tensor
from nanotron.parallel.parameters import NanotronParameter

try:
    import transformer_engine as te  # noqa
    import transformer_engine_torch as tex  # noqa
except ImportError:
    warnings.warn("Please install Transformer engine for FP8 training!")


@dataclass
class FP8LinearMeta:
    """FP8 metadata for FP8Linear."""

    input: Optional[FP8Meta] = None
    weight: Optional[FP8Meta] = None
    input_grad: Optional[FP8Meta] = None
    weight_grad: Optional[FP8Meta] = None


class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        recipe: FP8LinearRecipe = FP8LM_LINEAR_RECIPE,
    ):
        """
        Args:
            qtype (DTypes, optional): This is accumulation precision dtype
        """
        assert device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"

        # TODO(xrsrke): take initialization dtype from recipe
        # NOTE: initialize in float32
        assert dtype == torch.float32, f"FP8 linear recommends to initialize in float32, but got: {dtype}"
        super().__init__(in_features, out_features, bias, device, dtype=dtype)
        self._set_and_quantize_weights(self.weight.data, recipe)

    def _set_and_quantize_weights(self, data: torch.Tensor, recipe: FP8LinearRecipe = FP8LM_LINEAR_RECIPE):
        """
        data: if set to None, then we quantize the module's current weights, otherwise, we quantize
        the provided tensor
        """
        assert data is None or isinstance(data, torch.Tensor)
        quant_w = FP8Tensor(data, dtype=recipe.weight.dtype, interval=recipe.weight.interval)

        # NOTE: if we create a new parameter, then we can have that new quantized parameter
        # in [torch.int8, torch.uint8] dtype, then we can assign int|uint8 gradient to it
        # TODO(xrsrke): keep the metadata of the original NanotronParameter
        new_param = NanotronParameter.create_param_that_share_metadata(quant_w, param=self.weight)
        setattr(self, "weight", new_param)

        # NOTE: assume each time we requantize the weights, we reset the metadata
        self.metadatas = FP8LinearMeta()
        self.recipe = recipe

        if self.bias is not None:
            self.bias.data = self.bias.data.to(QTYPE_TO_DTYPE[recipe.accum_dtype])

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> torch.Tensor:
        import nanotron.fp8.functional as F

        return F.linear(
            input=input,
            weight=self.weight,
            bias=self.bias,
            metadatas=self.metadatas,
            recipe=self.recipe,
        )

    # def __repr__(self) -> str:
    #     return f"FP8{super().__repr__()}"


class _FP8Matmul(torch.autograd.Function):
    @staticmethod
    # @torch.no_grad()
    def forward(
        ctx,
        input: Union[FP8Tensor, torch.Tensor],
        weight: NanotronParameter,
        output: torch.Tensor,
        phony: torch.Tensor,
        metadatas: FP8LinearMeta,
        recipe: FP8LinearRecipe,
        name,
    ) -> torch.Tensor:
        assert not isinstance(input, FP8Tensor)
        assert isinstance(weight, NanotronParameter)

        from nanotron import constants
        from nanotron.config.fp8_config import FP8Args

        if constants.CONFIG is None:
            fp8_config = FP8Args()
        else:
            fp8_config = cast(FP8Args, constants.CONFIG.fp8)

        sync_amax_in_input = fp8_config.sync_amax_in_input

        if metadatas.input is None:
            fp8_input = FP8Tensor(
                input, dtype=recipe.input.dtype, interval=recipe.input.interval, sync=sync_amax_in_input
            )
            metadatas.input = fp8_input.fp8_meta
        else:
            fp8_input = FP8Tensor.from_metadata(input, metadatas.input, sync=sync_amax_in_input)

        ctx.save_for_backward(fp8_input, weight)
        ctx.is_input_require_grad = input.requires_grad
        ctx.metadatas = metadatas
        ctx.name = name
        ctx.recipe = recipe

        accum_output = output

        output = fp8_matmul_kernel(
            # NOTE: that works
            mat_a=weight.data,
            mat_b=fp8_input,
            output=accum_output,
            use_split_accumulator=recipe.split_accumulator.output,
            accumulate=recipe.accumulate.output,
            accum_qtype=recipe.accum_dtype,
        )
        return output, phony

    @staticmethod
    # @torch.no_grad()  # NOTE: drop 5% speed up in fwd only, and add 2% speed up in fwd+bwd
    def backward(ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        ∂L/∂X = ∂L/∂Y @ Wᵀ
        ∂L/∂W = Xᵀ @ ∂L/∂Y
        Reference: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        """
        from typing import cast

        from nanotron import constants
        from nanotron.config.fp8_config import FP8Args
        from nanotron.fp8.utils import is_overflow_underflow_nan

        if constants.CONFIG is None:
            fp8_config = FP8Args()
        else:
            fp8_config = cast(FP8Args, constants.CONFIG.fp8)

        sync_amax_in_igrad = fp8_config.sync_amax_in_igrad
        sync_amax_in_wgrad = fp8_config.sync_amax_in_wgrad

        fp8_input, fp8_weight_param = ctx.saved_tensors
        fp8_weight = fp8_weight_param.data
        recipe = ctx.recipe
        recipe = cast(FP8LinearRecipe, recipe)

        fp8_input = cast(FP8Tensor, fp8_input)
        fp8_weight = cast(FP8Tensor, fp8_weight)

        assert is_overflow_underflow_nan(grad_output) is False, f"name: {ctx.name}"

        ctx.metadatas = cast(FP8LinearMeta, ctx.metadatas)
        if ctx.metadatas.input_grad is None:
            fp8_grad_output = FP8Tensor(
                grad_output,
                dtype=recipe.input_grad.dtype,
                interval=recipe.input_grad.interval,
                sync=sync_amax_in_igrad,
            )
            ctx.metadatas.input_grad = fp8_grad_output.fp8_meta
        else:
            fp8_grad_output = FP8Tensor.from_metadata(grad_output, ctx.metadatas.input_grad, sync=sync_amax_in_igrad)

        if ctx.is_input_require_grad:
            transposed_fp8_weight = fp8_weight.transpose_fp8()
            # NOTE: same reason as output buffer in .forward
            grad_input_temp = torch.zeros(
                fp8_grad_output.shape[0],
                transposed_fp8_weight.shape[0],
                device="cuda",
                dtype=recipe.accum_dtype,
            )
            grad_input = fp8_matmul_kernel(
                mat_a=transposed_fp8_weight,
                mat_b=fp8_grad_output,
                output=grad_input_temp,
                use_split_accumulator=recipe.split_accumulator.input_grad,
                accum_qtype=recipe.accum_dtype,
                accumulate=recipe.accumulate.input_grad,
            )
            grad_input.__debug_is_from_fp8 = True
        else:
            grad_input = None

        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        assert is_overflow_underflow_nan(grad_input) is False if grad_input is not None else True

        # TODO(xrsrke): fuse cast and transpose
        transposed_fp8_grad_output = fp8_grad_output.transpose_fp8()
        transposed_fp8_input = fp8_input.transpose_fp8()

        # NOTE: same reason as output buffer in .forward
        grad_weight_temp = torch.zeros(
            transposed_fp8_input.shape[0],
            transposed_fp8_grad_output.shape[0],
            device="cuda",
            dtype=recipe.accum_dtype,
        )
        grad_weight = fp8_matmul_kernel(
            mat_a=transposed_fp8_input,
            mat_b=transposed_fp8_grad_output,
            output=grad_weight_temp,
            use_split_accumulator=recipe.split_accumulator.weight_grad,
            accumulate=recipe.accumulate.weight_grad,
            accum_qtype=recipe.accum_dtype,
        )
        assert is_overflow_underflow_nan(grad_weight) is False

        if ctx.is_input_require_grad:
            assert grad_input.dtype == recipe.accum_dtype

        assert grad_weight.dtype == recipe.accum_dtype
        # TODO(xrsrke): maintain a persistence metadata across training

        grad_weight = grad_weight.reshape(grad_weight.shape[::-1])

        if ctx.metadatas.weight_grad is None:
            fp8_weight_grad = FP8Tensor(
                grad_weight,
                dtype=recipe.weight_grad.dtype,
                interval=recipe.weight_grad.interval,
                sync=sync_amax_in_wgrad,
            )
            ctx.metadatas.weight_grad = fp8_weight_grad.fp8_meta
        else:
            fp8_weight_grad = FP8Tensor.from_metadata(grad_weight, ctx.metadatas.weight_grad, sync=sync_amax_in_wgrad)

        fp8_weight_param.grad = fp8_weight_grad

        # NOTE: sanity check
        assert isinstance(fp8_weight_param.grad, FP8Tensor)
        return grad_input, None, None, None, None, None, None
