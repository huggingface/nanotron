import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

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
    accumulate: bool,
    accum_qtype: torch.dtype,
    # TODO(xrsrke): remove this flag
    is_backward: bool = False,
    recipe=None,
) -> torch.Tensor:
    assert (
        mat_a.device != "cpu" and mat_b.device != "cpu"
    ), "The tensors must be on a CUDA device in order to use the FP8 kernel!!"
    # assert isinstance(accum_qtype, DTypes)
    assert isinstance(accum_qtype, torch.dtype)

    device = mat_a.device

    # NOTE: this is the accumulation precision dtype
    # TODO(xrsrke): move this mapping to constants
    # if accum_qtype == DTypes.KFLOAT32:
    #     out_dtype = getattr(tex.DType, "kFloat32")
    #     out_torch_dtype = torch.float32
    # elif accum_qtype == DTypes.KFLOAT16:
    #     out_dtype = getattr(tex.DType, "kFloat16")
    #     out_torch_dtype = torch.float16
    # else:
    #     raise ValueError(f"Unsupported accumulation dtype: {accum_qtype}")

    if accum_qtype == torch.float32:
        out_dtype = getattr(tex.DType, "kFloat32")
        out_torch_dtype = torch.float32
    elif accum_qtype == torch.float16:
        out_dtype = getattr(tex.DType, "kFloat16")
        out_torch_dtype = torch.float16
    elif accum_qtype == torch.bfloat16:
        out_dtype = getattr(tex.DType, "kBFloat16")
        out_torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported accumulation dtype: {accum_qtype}")

    _empty_tensor = torch.Tensor()

    # # output = torch.empty(mat_a.shape[0], mat_b.shape[1], device=device, dtype=torch.float32)
    # # output = torch.empty(mat_a.shape[0], mat_b.shape[-1], device=device, dtype=out_torch_dtype)
    # # NOTE: mat_a = weight, mat_b = input
    # output = torch.empty(mat_b.shape[0], mat_a.shape[0], device=device, dtype=out_torch_dtype)

    workspace = torch.empty(33_554_432, dtype=torch.int8, device=device)
    # accumulate = False

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

    try:
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
    except RuntimeError:
        raise RuntimeError(
            f"mat_a_fp8_meta.te_dtype: {mat_a_fp8_meta.te_dtype}, mat_b_fp8_meta.te_dtype: {mat_b_fp8_meta.te_dtype}, out_dtype: {out_dtype}, recipe: {recipe}"
        )

    return output
