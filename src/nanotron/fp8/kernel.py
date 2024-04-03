import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor


@torch.no_grad()
def fp8_matmul_kernel(
    mat_a: FP8Tensor,
    transpose_a: bool,
    mat_b: FP8Tensor,
    transpose_b: bool,
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

    # orig_mat_a_shape = deepcopy(mat_a.shape)
    # orig_mat_b_shape = deepcopy(mat_b.shape)

    mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype) if transpose_a is False else mat_a
    mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype) if transpose_b is True else mat_b

    # output = torch.empty(mat_a.T.shape[0], mat_b.shape[-1], device=device, dtype=out_torch_dtype)
    if is_backward is False:
        output = torch.empty(mat_b.shape[0], mat_a.shape[0], device=device, dtype=out_torch_dtype)
    else:
        output = torch.empty(mat_b.shape[0], mat_a.shape[0], device=device, dtype=out_torch_dtype)

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


@torch.no_grad()
def _fp8_matmul_kernel(
    mat_a: FP8Tensor,
    transpose_a: bool,
    mat_b: FP8Tensor,
    transpose_b: bool,
    use_split_accumulator: bool,
    accum_qtype: DTypes,
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

    # output = torch.empty(mat_a.shape[0], mat_b.shape[1], device=device, dtype=torch.float32)
    # output = torch.empty(mat_a.shape[0], mat_b.shape[-1], device=device, dtype=out_torch_dtype)
    # NOTE: mat_a = weight, mat_b = input

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
    # TE_CONFIG_TRANSPOSE_A = True
    # TE_CONFIG_TRANSPOSE_B = False
    SCALE = AMAX = _empty_tensor

    # orig_mat_a_shape = deepcopy(mat_a.shape)
    # orig_mat_b_shape = deepcopy(mat_b.shape)

    # mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype) if transpose_a is False else mat_a
    # mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype) if transpose_b is True else mat_b

    _mat_a_shape = (mat_a.T if transpose_a else mat_a).shape
    _mat_b_shape = (mat_b.T if transpose_b else mat_b).shape
    # output = torch.empty(_mat_a_shape[0], _mat_b_shape[-1], device=device, dtype=out_torch_dtype)
    output = torch.empty(_mat_a_shape[0], _mat_b_shape[-1], device=device, dtype=out_torch_dtype)

    tex.te_gemm(
        mat_a,
        mat_a_fp8_meta.inverse_scale,
        mat_a_fp8_meta.te_dtype,
        transpose_a,
        mat_b,
        mat_b_fp8_meta.inverse_scale,
        mat_b_fp8_meta.te_dtype,
        transpose_b,
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


@torch.no_grad()
def _fp8_matmul_kernel_2(
    mat_a: FP8Tensor,
    transpose_a: bool,
    mat_b: FP8Tensor,
    transpose_b: bool,
    use_split_accumulator: bool,
    accum_qtype: DTypes,
    # TODO(xrsrke): remove this flag
    # is_backward: bool = False
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

    # orig_mat_a_shape = deepcopy(mat_a.shape)
    # orig_mat_b_shape = deepcopy(mat_b.shape)

    mat_a_new_shape = mat_a.T.shape if transpose_a is True and TE_CONFIG_TRANSPOSE_A is True else mat_a.T.shape
    mat_b_new_shape = mat_b.T.shape if transpose_b is True and TE_CONFIG_TRANSPOSE_B is True else mat_b.T.shape

    mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype) if transpose_a is False else mat_a
    mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype) if transpose_b is True else mat_b

    # output = torch.empty(mat_a.T.shape[0], mat_b.shape[-1], device=device, dtype=out_torch_dtype)
    output = torch.empty(mat_a_new_shape[0], mat_b_new_shape[-1], device=device, dtype=out_torch_dtype)

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
