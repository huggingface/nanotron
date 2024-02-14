import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.tensor import FP8Tensor
from nanotron.fp8.meta import FP8Meta


@torch.no_grad()
def fp8_matmul_kernel(
    mat_a: FP8Tensor,
    transpose_a: bool,
    mat_b: FP8Tensor,
    transpose_b: bool,
    use_split_accumulator: bool,
) -> torch.Tensor:
    assert (
        mat_a.device != "cpu" and mat_b.device != "cpu"
    ), "The tensors must be on a CUDA device in order to use the FP8 kernel!!"

    device = mat_a.device

    _empty_tensor = torch.Tensor()
    output = torch.empty(mat_a.shape[0], mat_b.shape[1], device=device, dtype=torch.float32)
    workspace = torch.empty(33_554_432, dtype=torch.int8, device=device)
    accumulate = False

    out_dtype = getattr(tex.DType, "kFloat32")
    # NOTE: currently TE don't support adding bias in FP8
    # along with matmul, it only takes an empty bias
    bias = torch.tensor([], dtype=torch.float32)
    TE_CONFIG_TRANSPOSE_BIAS = False

    mat_a_fp8_meta: FP8Meta = mat_a.fp8_meta
    mat_b_fp8_meta: FP8Meta = mat_b.fp8_meta

    # NOTE: these are the fixed configs that TE only takes
    # so we have to TE the A and B matrix to match these configs
    TE_CONFIG_TRANSPOSE_A = True
    TE_CONFIG_TRANSPOSE_B = False
    SCALE = AMAX = _empty_tensor

    mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype) if transpose_a is False else mat_a
    mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype) if transpose_b is True else mat_b

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
