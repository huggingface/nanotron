import warnings

import torch

from nanotron.fp8.tensor import FP8Tensor

try:
    import transformer_engine as te  # noqa
    import transformer_engine_extensions as tex
except ImportError:
    warnings.warn("Please install Transformer engine for FP8 training.")


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
    transpose_bias = False

    mat_a_fp8_meta = mat_a.fp8_meta
    mat_b_fp8_meta = mat_b.fp8_meta

    if transpose_a is False:
        mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype)

    if transpose_b is True:
        mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype)

    tex.te_gemm(
        mat_a,
        mat_a_fp8_meta.inverse_scale,
        mat_a_fp8_meta.te_dtype,
        True,  # transa, default True
        mat_b,
        mat_b_fp8_meta.inverse_scale,
        mat_b_fp8_meta.te_dtype,
        False,  # transb, default False
        output,
        _empty_tensor,  # scale
        out_dtype,
        _empty_tensor,  # amax
        bias,
        out_dtype,
        _empty_tensor,
        transpose_bias,  # grad, defualt False
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
        0,
    )

    return output
