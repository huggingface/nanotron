import torch
import transformer_engine as te  # noqa
import transformer_engine_torch as tex

from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor


@torch.no_grad()
def fp8_matmul_kernel(
    mat_a: FP8Tensor,
    mat_b: FP8Tensor,
    output,
    use_split_accumulator: bool,
    accumulate: bool,
    accum_qtype: torch.dtype,
    # TODO(xrsrke): remove this flag
) -> torch.Tensor:
    # from nanotron.fp8.constants import _empty_tensor, workspace

    assert (
        mat_a.device != "cpu" and mat_b.device != "cpu"
    ), "The tensors must be on a CUDA device in order to use FP8!!"
    # assert isinstance(accum_qtype, DTypes)
    assert isinstance(accum_qtype, torch.dtype)

    device = mat_a.device

    # NOTE: this is the accumulation precision dtype
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
    workspace = torch.empty(33_554_432, dtype=torch.int8, device=device)

    # NOTE: currently TE don't support adding bias in FP8
    # along with matmul, it only takes an empty bias
    bias = torch.tensor([], dtype=out_torch_dtype)
    TE_CONFIG_TRANSPOSE_BIAS = False

    mat_a_fp8_meta: FP8Meta = mat_a.fp8_meta
    mat_b_fp8_meta: FP8Meta = mat_b.fp8_meta

    # NOTE: these are the fixed configs that TE only takes
    # so we have to TE the A and B matrix to match these configs
    TE_CONFIG_TRANSPOSE_A = True
    TE_CONFIG_TRANSPOSE_B = False
    SCALE = AMAX = _empty_tensor

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
