import torch
import transformer_engine.pytorch.cpp_extensions as texcpp

# from transformer_engine.pytorch.module import get_workspace
# import transformer_engine_extensions as tex
import transformer_engine_torch as tex

scale = 1.0

meta = tex.FP8TensorMeta()
meta.scale = torch.ones(1, dtype=torch.float32, device="cuda") * scale
meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / scale
meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")


def cast_to_fp8(x, qtype):
    ret = texcpp.cast_to_fp8(x, meta, tex.FP8FwdTensors.GEMM1_INPUT, qtype)
    ret._fp8_qtype = qtype
    return ret


def cast_from_fp8(x, qtype):
    ret = texcpp.cast_from_fp8(x, meta, tex.FP8FwdTensors.GEMM1_INPUT, x._fp8_qtype, qtype)
    ret._fp8_qtype = qtype
    return ret


one_scale_inv = torch.ones(1, dtype=torch.float32, device="cuda")
empty_tensor = torch.Tensor()
# workspace = get_workspace()
workspace = torch.empty(33_554_432, dtype=torch.int8, device="cuda")
assert workspace.is_cuda


# PT_DType = dict([(v, k) for k, v in texcpp.TE_DType.items()])
# PT_DType[tex.DType.kFloat8E4M3] = torch.uint8
# PT_DType[tex.DType.kFloat8E5M2] = torch.uint8


def convert_torch_dtype_to_te_dtype(dtype: torch.dtype) -> tex.DType:
    # NOTE: transformer engine maintains it own dtype mapping
    # so we need to manually map torch dtypes to TE dtypes
    TORCH_DTYPE_TE_DTYPE_NAME_MAPPING = {
        torch.int32: "kInt32",
        torch.float32: "kFloat32",
        torch.float16: "kFloat16",
        torch.bfloat16: "kBFloat16",
        # DTypes.FP8E4M3: "kFloat8E4M3",
        # DTypes.FP8E5M2: "kFloat8E5M2",
        # DTypes.KFLOAT16: "kFloat16",
    }
    return getattr(tex.DType, TORCH_DTYPE_TE_DTYPE_NAME_MAPPING[dtype])


def fp8_gemm(fa, fb, trans_a, trans_b, bias=None, qtype=tex.DType.kFloat32):
    """
    # te_gemm

    input_A: (A_row, A_col)
    input_B: (B_row, B_col)

    when transa, transb = True, False
    m, k, n = A_row, A_col, B_row
    lda, ldb, ldd = A_col, A_col, A_row
    output_D: (B_row, A_row)

    when transa, transb = False, False
    m, k, n = A_col, A_row, B_row
    lda, ldb, ldd = A_col, A_row, A_col
    output_D: (B_row, A_col)

    when transa, transb = False, True
    m, k, n = A_col, A_row, B_col
    lda, ldb, ldd = A_col, B_col, A_col
    output_D: (B_col, A_col)
    """
    assert fa.is_cuda and fb.is_cuda
    assert fa.is_contiguous()
    assert fb.is_contiguous()
    device = fa.device
    fa_qtype, fb_qtype = fa._fp8_qtype, fb._fp8_qtype
    A_row, A_col = fa.shape
    B_row, B_col = fb.shape
    if trans_a and not trans_b:
        assert A_col == B_col
        C_row, C_col = B_row, A_row
    elif not trans_a and not trans_b:
        assert A_row == B_col
        C_row, C_col = B_row, A_col
    elif not trans_a and trans_b:
        assert A_row == B_row
        C_row, C_col = B_col, A_col
    out_shape = (C_row, C_col)

    # dtype = PT_DType[qtype]
    if qtype == tex.DType.kFloat32:
        dtype = torch.float32
    elif qtype == tex.DType.kFloat16:
        dtype = torch.float16

    out = torch.empty(out_shape, dtype=dtype, device=device)
    # te_gemm is column-order.

    # tex.te_gemm(
    #     fa, one_scale_inv, fa_qtype, trans_a,
    #     fb, one_scale_inv, fb_qtype, trans_b,
    #     out, qtype,
    #     bias or empty_tensor, empty_tensor, False,
    #     workspace, workspace.shape[0],
    #     False, True,
    # )

    _empty_tensor = torch.Tensor()
    SCALE = AMAX = _empty_tensor
    TE_CONFIG_TRANSPOSE_BIAS = False

    tex.te_gemm(
        fa,
        one_scale_inv,
        fa_qtype,
        trans_a,
        fb,
        one_scale_inv,
        fb_qtype,
        trans_b,
        # out, SCALE, qtype, AMAX,
        # bias or empty_tensor, qtype, False,
        # workspace, workspace.shape[0],
        # False, True,
        out,
        SCALE,
        qtype,
        AMAX,
        torch.tensor([], dtype=dtype),
        qtype,
        _empty_tensor,
        TE_CONFIG_TRANSPOSE_BIAS,
        workspace,
        workspace.shape[0],
        False,
        True,
        0,
    )

    out._fp8_qtype = qtype
    return out


def fp8_matmul(fa, fb, bias=None, qtype=tex.DType.kFloat32):
    # trans_a = False and trans_b = False is not implemented.
    fb_qtype = fb._fp8_qtype
    fb = fb.T.contiguous()
    fb._fp8_qtype = fb_qtype
    return fp8_gemm(fb, fa, trans_a=True, trans_b=False, bias=bias, qtype=qtype)


h100_peak_flops_float32 = 67e12
h100_peak_flops_fp16_tc = 989e12
h100_peak_tops_float8_tc = 1979e12

dtype_to_peak_tops = {
    torch.float32: h100_peak_flops_float32,
    torch.float16: h100_peak_flops_fp16_tc,
    torch.bfloat16: h100_peak_flops_fp16_tc,
    torch.float8_e4m3fn: h100_peak_tops_float8_tc,
    torch.float8_e5m2: h100_peak_tops_float8_tc,
}

from torch.utils import benchmark


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)

    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    measurement = t0.blocked_autorange()
    return measurement.mean


def run_fp8(a, b):
    fa = cast_to_fp8(a, tex.DType.kFloat8E4M3)
    fb = cast_to_fp8(b, tex.DType.kFloat8E4M3)
    fp8_matmul(fa, fb, qtype=tex.DType.kFloat16)


def run_bfloat16(a, b):
    a = a.to(torch.bfloat16)
    b = b.to(torch.bfloat16)
    torch.matmul(a, b)


def benchmark_linear_operations(a, b):
    M, K = a.shape
    N, _ = b.shape

    # Benchmark FP8
    fp8_time = benchmark_fn_in_sec(run_fp8, a, b)

    # Benchmark BFloat16
    bfloat16_time = benchmark_fn_in_sec(run_bfloat16, a, b)

    # Calculate FLOPS
    # Each linear operation performs 2*M*N*K FLOPs (multiply-add)
    total_flops = 2 * M * N * K

    fp8_tflops = (total_flops / fp8_time) / 1e12
    bfloat16_tflops = (total_flops / bfloat16_time) / 1e12

    # Calculate efficiency compared to peak performance
    fp8_efficiency = (fp8_tflops / (h100_peak_tops_float8_tc / 1e12)) * 100
    bfloat16_efficiency = (bfloat16_tflops / (h100_peak_flops_fp16_tc / 1e12)) * 100

    return {
        "M": M,
        "N": N,
        "K": K,
        "FP8_time_ms": fp8_time * 1000,
        "BF16_time_ms": bfloat16_time * 1000,
        "FP8_TFLOPS": fp8_tflops,
        "BF16_TFLOPS": bfloat16_tflops,
        "FP8_eff%": fp8_efficiency,
        "BF16_eff%": bfloat16_efficiency,
        "Speedup": bfloat16_time / fp8_time,
    }


if __name__ == "__main__":
    # a = torch.randn(128, 128).cuda()
    # b = torch.randn(128, 128).cuda()
    # qa = cast_from_fp8(fa, tex.DType.kFloat32)
    # qb = cast_from_fp8(fb, tex.DType.kFloat32)
    # qc = torch.matmul(qa, qb)

    # E4M3/E5M2 @ E4M3/E5M2 = FP16/FP32
    # print(qc, qc2)

    import pandas as pd

    def create_benchmark_table(sizes):
        results = []
        for size in sizes:
            a = torch.randn(size, size).cuda()
            b = torch.randn(size, size).cuda()
            result = benchmark_linear_operations(a, b)
            results.append(result)

        df = pd.DataFrame(results)
        df = df.round(2)  # Round to 2 decimal places
        return df

    # Example usage:
    sizes = [4096, 16384, 32768, 28672, 49152]
    benchmark_table = create_benchmark_table(sizes)
    print(benchmark_table)
