from dataclasses import dataclass
from typing import Optional, Tuple, Union, cast

import pydevd
import torch
import transformer_engine as te  # noqa
from torch import nn

from nanotron.fp8.constants import FP8LM_LINEAR_RECIPE
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.recipe import FP8LinearRecipe
from nanotron.fp8.tensor import FP8Tensor
from nanotron.parallel.parameters import NanotronParameter


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
        # recipe: FP8LinearRecipe = FP8LM_LINEAR_RECIPE,
        # NOTE: placeholder for dtype in torch's nn.Linear
        # TODO(xrsrke): remove this shit
        **kwargs,
    ):
        """
        Args:
            qtype (DTypes, optional): This is accumulation precision dtype
        """
        assert device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"

        # TODO(xrsrke): take initialization dtype from recipe
        # NOTE: initialize in float32
        super().__init__(in_features, out_features, bias, device, dtype=torch.float32)
        self._set_and_quantize_weights(self.weight.data)

        # assert self.bias is None
        # if self.bias is not None:
        #     self.bias = nn.Parameter(self.bias.to(recipe.accum_dtype))
        #     assert self.bias.dtype == recipe.accum_dtype

        # self.metadatas = FP8LinearMeta()
        # self.recipe = recipe

    def _set_and_quantize_weights(self, data: Optional[torch.Tensor], recipe: FP8LinearRecipe = FP8LM_LINEAR_RECIPE):
        """
        data: if set to None, then we quantize the module's current weights, otherwise, we quantize
        the provided tensor
        """
        # quant_w = FP8Parameter(self.weight.data, dtype=recipe.weight.dtype, interval=recipe.weight.interval)
        quant_w = FP8Tensor(data, dtype=recipe.weight.dtype, interval=recipe.weight.interval)

        # assert quant_w.dtype in [torch.uint8, torch.int8], f"got {self.weight.data.dtype}"
        # self.weight = quant_w
        # setattr(self.weight, "data", quant_w)
        # NOTE: if we create a new parameter, then we can have that new quantized parameter
        # in [torch.int8, torch.uint8] dtype, then we can assign int|uint8 gradient to it
        # TODO(xrsrke): keep the metadata of the original NanotronParameter
        # setattr(self, "weight", NanotronParameter(tensor=quant_w))
        setattr(self, "weight", NanotronParameter.create_param_that_share_metadata(quant_w, self.weight))

        # if self.name == "model.decoder.0.attention.qkv_proj":
        #     assert 1 == 1

        # NOTE: assume each time we requantize the weights, we reset the metadata
        self.metadatas = FP8LinearMeta()
        self.recipe = recipe

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> torch.Tensor:
        import nanotron.fp8.functional as F

        return F.linear(
            input=input,
            # weight=get_data_from_param(self.weight),
            # bias=None if self.bias is None else get_data_from_param(self.bias),
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
            # mat_a=weight, # i used weight before removing get_data_from_param
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

        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        if (
            constants.CONFIG is not None
            and constants.CONFIG.fp8 is not None
            and constants.CONFIG.fp8.is_debugging is True
        ):
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        # dist.monitored_barrier(wait_all_ranks=True)

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
            grad_input_temp = torch.empty(
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

        # TODO(xrsrke): fuse cast and transpose
        transposed_fp8_grad_output = fp8_grad_output.transpose_fp8()
        transposed_fp8_input = fp8_input.transpose_fp8()

        grad_weight_temp = torch.empty(
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

        if ctx.is_input_require_grad:
            assert grad_input.dtype == recipe.accum_dtype

        assert grad_weight.dtype == recipe.accum_dtype
        # TODO(xrsrke): maintain a persistence metadata across training

        grad_weight = grad_weight.reshape(grad_weight.shape[::-1])

        # NOTE: if use gradient accumulation, then directly keep the high precision weights for later accumulate
        if constants.CONFIG is not None and (
            constants.CONFIG.tokens.batch_accumulation_per_replica > 1
            or constants.CONFIG.fp8.is_directly_keep_accum_grad_of_fp8 is True
        ):
            from nanotron.helpers import set_accum_grad

            # NOTE: if do fp8_weight.grad = grad_weight, then the following error will be raised:
            # Traceback (most recent call last):
            #   File "<string>", line 1, in <module>
            #   File "/fsx/phuc/temp/temp3_env_for_fp8/env/lib/python3.10/site-packages/torch/_tensor.py", line 1386, in __torch_function__
            #     ret = func(*args, **kwargs)
            # RuntimeError: attempting to assign a gradient with dtype 'c10::BFloat16' to a tensor with dtype 'unsigned char'. Please ensure that the gradient and the tensor have the same dtype
            fp8_weight.__accum_grad = grad_weight
            assert fp8_weight.__accum_grad.dtype in [torch.float16, torch.bfloat16, torch.float32]
            # constants.ACCUM_GRADS[ctx.name] = grad_weight
            set_accum_grad(ctx.name, grad_weight)
        else:
            if ctx.metadatas.weight_grad is None:
                fp8_weight_grad = FP8Tensor(
                    grad_weight,
                    dtype=recipe.weight_grad.dtype,
                    interval=recipe.weight_grad.interval,
                    sync=sync_amax_in_wgrad,
                )
                ctx.metadatas.weight_grad = fp8_weight_grad.fp8_meta
            else:
                fp8_weight_grad = FP8Tensor.from_metadata(
                    grad_weight, ctx.metadatas.weight_grad, sync=sync_amax_in_wgrad
                )

            fp8_weight_param.grad = fp8_weight_grad

            # NOTE: sanity check
            assert isinstance(fp8_weight_param.grad, FP8Tensor)

        return grad_input, fp8_weight_grad, None, None, None, None, None
