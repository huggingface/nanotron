from typing import Tuple, Union, cast

import torch
import transformer_engine as te  # noqa  # noqa
import transformer_engine_extensions as tex

from nanotron import constants
from nanotron.fp8.constants import FP8LM_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8LinearMeta

# from nanotron.fp8.kernel import fp8_matmul_kernel, correct_fp8_matmul_kernel
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor


@torch.no_grad()
def fp8_matmul_kernel(
    mat_a: FP8Tensor,
    transpose_a: bool,
    mat_b: FP8Tensor,
    transpose_b: bool,
    output,
    use_split_accumulator: bool,
    accum_qtype: DTypes,
    # TODO(xrsrke): remove this flag
    is_backward: bool = False,
) -> torch.Tensor:
    assert (
        mat_a.device != "cpu" and mat_b.device != "cpu"
    ), "The tensors must be on a CUDA device in order to use the FP8 kernel!!"
    assert isinstance(accum_qtype, DTypes)

    device = mat_a.device

    # NOTE: this is the accumulation precision dtype
    # TODO(xrsrke): move this mapping to constants
    if accum_qtype == DTypes.KFLOAT32:
        out_dtype = getattr(tex.DType, "kFloat32")
        out_torch_dtype = torch.float32
    elif accum_qtype == DTypes.KFLOAT16:
        out_dtype = getattr(tex.DType, "kFloat16")
        out_torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported accumulation dtype: {accum_qtype}")

    _empty_tensor = torch.Tensor()

    # # output = torch.empty(mat_a.shape[0], mat_b.shape[1], device=device, dtype=torch.float32)
    # # output = torch.empty(mat_a.shape[0], mat_b.shape[-1], device=device, dtype=out_torch_dtype)
    # # NOTE: mat_a = weight, mat_b = input
    # output = torch.empty(mat_b.shape[0], mat_a.shape[0], device=device, dtype=out_torch_dtype)

    workspace = torch.empty(33_554_432, dtype=torch.int8, device=device)
    accumulate = False

    # NOTE: currently TE don't support adding bias in FP8
    # along with matmul, it only takes an empty bias
    # bias = torch.tensor([], dtype=torch.float32)
    bias = torch.tensor([], dtype=out_torch_dtype)
    TE_CONFIG_TRANSPOSE_BIAS = False

    mat_a_fp8_meta: FP8Meta = mat_a.fp8_meta
    mat_b_fp8_meta: FP8Meta = mat_b.fp8_meta

    # NOTE: these are the fixed configs that TE only takes
    # so we have to TE the A and B matrix to match these configs
    TE_CONFIG_TRANSPOSE_A = True
    TE_CONFIG_TRANSPOSE_B = False
    SCALE = AMAX = _empty_tensor

    if is_backward is False:
        # orig_mat_a_shape = deepcopy(mat_a.shape)
        # orig_mat_b_shape = deepcopy(mat_b.shape)

        mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype) if transpose_a is False else mat_a
        mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype) if transpose_b is True else mat_b

        # if transpose_a is True:
        #     mat_a = mat_a
        # elif transpose_a is False:
        #     mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype)

        # if transpose_b is False:
        #     mat_b = mat_b
        # elif transpose_b is True:
        #     mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype)

        # if transpose_a is False:
        #     mat_a = mat_a
        # elif transpose_a is True:
        #     mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype)

        # if transpose_b is True:
        #     mat_b = mat_b
        # elif transpose_b is False:
        #     mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype)

        # output = torch.empty(mat_a.T.shape[0], mat_b.shape[-1], device=device, dtype=out_torch_dtype)

    # if is_backward is False:
    #     output = torch.empty(mat_b.shape[0], mat_a.shape[0], device=device, dtype=out_torch_dtype)
    #     # output = torch.empty(mat_b.shape[-1], mat_a.shape[-1], device=device, dtype=out_torch_dtype)
    # else:
    #     output = torch.empty(mat_b.shape[0], mat_a.shape[0], device=device, dtype=out_torch_dtype)
    #     # output = torch.empty(mat_a.shape[-1], mat_b.shape[-1], device=device, dtype=out_torch_dtype)

    tex.te_gemm(
        mat_a,
        mat_a_fp8_meta.inverse_scale,
        mat_a_fp8_meta.te_dtype,
        TE_CONFIG_TRANSPOSE_A,
        mat_b,
        mat_b_fp8_meta.inverse_scale,
        mat_b_fp8_meta.te_dtype,
        TE_CONFIG_TRANSPOSE_B,
        output,
        SCALE,
        out_dtype,
        AMAX,
        bias,
        out_dtype,
        _empty_tensor,
        TE_CONFIG_TRANSPOSE_BIAS,
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
        0,
    )

    return output


# NOTE: not original version
# class _FP8Matmul(torch.autograd.Function):
#     @staticmethod
#     @torch.no_grad()
#     def forward(
#         ctx,
#         input: Union[FP8Tensor, torch.Tensor],
#         weight: FP8Tensor,
#         output: torch.Tensor,
#         phony: torch.Tensor,
#         metadatas: FP8LinearMeta,
#         accum_qtype: DTypes,
#         name: Optional[str] = None  # TODO(xrsrke): remove this shit after debugging
#         # is_weight_transposed: bool = False,
#     ) -> torch.Tensor:
#         assert not isinstance(input, FP8Tensor)

#         # NOTE: pad input shape to disibile by 16
#         # input = F.pad(input, (0, 0, 0, 16 - input.shape[1] % 16))

#         if metadatas.input is None:
#             fp8_input = FP8Tensor(
#                 input,
#                 dtype=FP8LM_RECIPE.linear.input.dtype,
#                 interval=FP8LM_RECIPE.linear.input.interval,
#                 # is_delayed_scaling=FP8LM_RECIPE.linear.input.is_delayed_scaling,
#             )
#             metadatas.input = fp8_input.fp8_meta
#         else:
#             fp8_input = FP8Tensor.from_metadata(input, metadatas.input)

#         ctx.accum_qtype = accum_qtype
#         ctx.metadatas = metadatas
#         # ctx.is_weight_transposed = is_weight_transposed
#         ctx.save_for_backward(fp8_input, weight)
#         ctx.name = name

#         output = fp8_matmul_kernel(
#             # NOTE: that works
#             mat_a=weight,
#             transpose_a=True,
#             mat_b=fp8_input,
#             transpose_b=False,
#             output=output,
#             use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.output,
#             accum_qtype=accum_qtype,
#         )

#         return output, phony

#     @staticmethod
#     @torch.no_grad()
#     def backward(ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
#         """
#         ∂L/∂X = ∂L/∂Y @ Wᵀ
#         ∂L/∂W = Xᵀ @ ∂L/∂Y
#         Reference: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
#         """
#         # import pydevd
#         # pydevd.settrace(suspend=False, trace_only_current_thread=True)
#         # assert isinstance(grad_output, torch.Tensor)
#         assert grad_output.__class__ == torch.Tensor
#         assert grad_output.dtype == torch.float16

#         if ctx.name == "mlp.down_proj":
#             assert 1 == 1

#         if ctx.name == "lm_head":
#             assert 1 == 1

#         fp8_input, fp8_weight = ctx.saved_tensors
#         accum_qtype = ctx.accum_qtype

#         fp8_input = cast(FP8Tensor, fp8_input)
#         fp8_weight = cast(FP8Tensor, fp8_weight)
#         grad_output = grad_output.contiguous()

#         ctx.metadatas = cast(FP8LinearMeta, ctx.metadatas)
#         if ctx.metadatas.input_grad is None:
#             fp8_grad_output = FP8Tensor(
#                 grad_output,
#                 dtype=FP8LM_RECIPE.linear.input_grad.dtype,
#                 interval=FP8LM_RECIPE.linear.input_grad.interval,
#             )

#             ctx.metadatas.input_grad = fp8_grad_output.fp8_meta
#         else:
#             fp8_grad_output = FP8Tensor.from_metadata(grad_output, ctx.metadatas.input_grad)

#         transposed_fp8_weight = fp8_weight.transpose_fp8()

#         grad_input_temp = torch.zeros(
#             fp8_grad_output.shape[0], transposed_fp8_weight.shape[0], device="cuda", dtype=QTYPE_TO_DTYPE[accum_qtype]
#         )
#         grad_input = fp8_matmul_kernel(
#             mat_a=transposed_fp8_weight,
#             transpose_a=True,
#             mat_b=fp8_grad_output,
#             transpose_b=False,
#             output=grad_input_temp,
#             use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.input_grad,
#             accum_qtype=accum_qtype,
#             is_backward=True,
#         )

#         transposed_fp8_grad_output = fp8_grad_output.transpose_fp8()
#         transposed_fp8_input = fp8_input.transpose_fp8()

#         grad_weight_temp = torch.zeros(
#             transposed_fp8_input.shape[0],
#             transposed_fp8_grad_output.shape[0],
#             device="cuda",
#             dtype=QTYPE_TO_DTYPE[accum_qtype],
#         )
#         grad_weight = fp8_matmul_kernel(
#             mat_a=transposed_fp8_input,
#             transpose_a=True,
#             mat_b=transposed_fp8_grad_output,
#             transpose_b=False,
#             output=grad_weight_temp,
#             use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.weight_grad,
#             accum_qtype=accum_qtype,
#             # is_backward=True,
#         )
#         assert grad_input.dtype == QTYPE_TO_DTYPE[accum_qtype]
#         assert grad_weight.dtype == QTYPE_TO_DTYPE[accum_qtype]
#         # TODO(xrsrke): maintain a persistence metadata across training

#         # if fp8_weight.shape != grad_weight.shape:
#         #     grad_weight = grad_weight.T

#         grad_weight = grad_weight.T

#         if ctx.metadatas.weight_grad is None:
#             # NOTE: this is weird, i only add this when work with TP
#             # didn't encount the mismatch shape problem with non-tp
#             fp8_weight_grad = FP8Tensor(
#                 grad_weight,
#                 dtype=FP8LM_RECIPE.linear.weight_grad.dtype,
#                 interval=FP8LM_RECIPE.linear.weight_grad.interval,
#             )
#             ctx.metadatas.weight_grad = fp8_weight_grad.fp8_meta
#         else:
#             fp8_weight_grad = FP8Tensor.from_metadata(grad_weight, ctx.metadatas.weight_grad)


#         fp8_weight.grad = fp8_weight_grad
#         fp8_weight._temp_grad = fp8_weight_grad

#         assert isinstance(fp8_weight.grad, FP8Tensor)

#         # assert isinstance(grad_input, torch.Tensor)
#         assert grad_output.__class__ == torch.Tensor
#         assert grad_input.dtype == torch.float16

#         return grad_input, None, None, None, None, None, None


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
        accum_qtype: DTypes,
        name,
    ) -> torch.Tensor:
        assert not isinstance(input, FP8Tensor)

        constants.DEBUG_FP8_INPUT = input

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
        ctx.save_for_backward(fp8_input, weight)
        ctx.name = name

        constants.DEBUG_FP8_WEIGHT = weight

        # NOTE: pass FP8Tensor instead of FP8Parameter
        output = fp8_matmul_kernel(
            # NOTE: that works
            mat_a=weight,
            transpose_a=True,
            mat_b=fp8_input,
            transpose_b=False,
            output=output,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.output,
            accum_qtype=accum_qtype,
        )

        constants.DEBUG_FP8_OUTPUT = output

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

        constants.DEBUG_FP8_GRAD_OUTPUT = grad_output

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

        grad_input_temp = torch.zeros(
            fp8_grad_output.shape[0], transposed_fp8_weight.shape[0], device="cuda", dtype=QTYPE_TO_DTYPE[accum_qtype]
        )
        # grad_input_temp = torch.zeros(
        #     32,
        #     16,
        #     device="cuda",
        #     dtype=QTYPE_TO_DTYPE[accum_qtype],
        # )
        grad_input = fp8_matmul_kernel(
            mat_a=transposed_fp8_weight,
            transpose_a=True,
            mat_b=fp8_grad_output,
            transpose_b=False,
            output=grad_input_temp,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.input_grad,
            accum_qtype=accum_qtype,
            is_backward=True,
        )
        constants.DEBUG_FP8_GRAD_INPUT = grad_input

        # fp8_grad_output_transposed = tex.fp8_transpose(fp8_grad_output, fp8_grad_output.fp8_meta.te_dtype)
        # fp8_grad_output_transposed.fp8_meta = fp8_grad_output.fp8_meta
        # fp8_input_tranposed = tex.fp8_transpose(fp8_input, fp8_input.fp8_meta.te_dtype)
        # fp8_input_tranposed.fp8_meta = fp8_input.fp8_meta

        transposed_fp8_grad_output = fp8_grad_output.transpose_fp8()
        transposed_fp8_input = fp8_input.transpose_fp8()

        # grad_weight_temp = torch.zeros(
        #     32,
        #     16,
        #     device="cuda",
        #     dtype=QTYPE_TO_DTYPE[accum_qtype],
        # )
        grad_weight_temp = torch.zeros(
            transposed_fp8_input.shape[0],
            transposed_fp8_grad_output.shape[0],
            device="cuda",
            dtype=QTYPE_TO_DTYPE[accum_qtype],
        )
        grad_weight = fp8_matmul_kernel(
            mat_a=transposed_fp8_input,
            transpose_a=True,
            mat_b=transposed_fp8_grad_output,
            transpose_b=False,
            output=grad_weight_temp,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.weight_grad,
            accum_qtype=accum_qtype,
            # is_backward=True
        )

        constants.DEBUG_FP8_GRAD_WEIGHT = grad_weight

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

        grad_weight = grad_weight.T.contiguous()
        constants.DEBUG_FP8_GRAD_WEIGHT_BEFORE_RESHAPE = grad_weight

        orig_shape = grad_weight.shape
        grad_weight = grad_weight.contiguous().t().contiguous().view(-1).contiguous().reshape(orig_shape)

        constants.DEBUG_FP8_GRAD_WEIGHT_AFTER_RESHAPE = grad_weight

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
        fp8_weight._temp_grad = fp8_weight_grad

        # NOTE: sanity check
        assert isinstance(fp8_weight.grad, FP8Tensor)
        return grad_input.contiguous(), None, None, None, None, None, None
