import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.tensor import FP8Tensor


@torch.no_grad()
def fp8_matmul_kernel(
    mat_a: FP8Tensor,
    transpose_a: bool,
    mat_b: FP8Tensor,
    transpose_b: bool,
    bias: FP8Tensor = None,
    transpose_bias: bool = False,
    use_split_accumulator: bool = None,
) -> torch.Tensor:
    assert use_split_accumulator is not None
    assert (
        mat_a.device != "cpu" and mat_b.device != "cpu"
    ), "The tensors must be on a CUDA device in order to use the FP8 kernel!!"

    device = mat_a.device

    _empty_tensor = torch.Tensor()
    output = torch.empty(mat_a.shape[0], mat_b.shape[1], device=device, dtype=torch.float32)
    workspace = torch.empty(33_554_432, dtype=torch.int8, device=device)
    accumulate = False
    # use_split_accumulator = False

    out_dtype = getattr(tex.DType, "kFloat32")
    # TODO(xrsrke): add support for adding bias in fp8
    bias = torch.tensor([], dtype=torch.float32)

    print("---------------------------------------------------------------- \n \n")
    print(
        f"pipegoose._fp8_kernel.before_matmul, mat_a[0, :10]: {mat_a[0, :3]}, mat_a.fp8_meta.inverse_scale: {mat_a.fp8_meta.inverse_scale}, mat_a.fp8_meta.te_dtype: {mat_a.fp8_meta.te_dtype} \n \n"
    )
    print(
        f"pipegoose._fp8_kernel.before_matmul, mat_b[0, :10]: {mat_b[0, :3]}, mat_b.fp8_meta.inverse_scale: {mat_b.fp8_meta.inverse_scale}, mat_b.fp8_meta.te_dtype: {mat_b.fp8_meta.te_dtype} \n \n"
    )
    print(f"pipegoose._fp8_kernel.before_matmul, output[0, :10]: {output[0, :3]}, out_dtype: {out_dtype} \n \n")
    print(
        f"pipegoose._fp8_kernel.before_matmul: _empty_tensor: {_empty_tensor}, bias: {bias}, workspace[:5]: {workspace[:5]}, accumulate: {accumulate}, use_split_accumulator: {use_split_accumulator} \n \n"
    )

    # print(f"pipegoose._fp8_kernel.before_matmul, mat_a: {mat_a}, mat_a.fp8_meta.inverse_scale: {mat_a.fp8_meta.inverse_scale}, mat_a.fp8_meta.te_dtype: {mat_a.fp8_meta.te_dtype} \n \n")
    print(
        f"pipegoose._fp8_kernel.before_matmul, mat_a.fp8_meta.inverse_scale: {mat_a.fp8_meta.inverse_scale}, mat_a.fp8_meta.te_dtype: {mat_a.fp8_meta.te_dtype} \n \n"
    )
    try:
        # print(f"pipegoose._fp8_kernel.before_matmul, mat_b: {mat_b}, mat_b.fp8_meta.inverse_scale: {mat_b.fp8_meta.inverse_scale}, mat_b.fp8_meta.te_dtype: {mat_b.fp8_meta.te_dtype} \n \n")
        print(
            f"pipegoose._fp8_kernel.before_matmul, mat_b.fp8_meta.inverse_scale: {mat_b.fp8_meta.inverse_scale}, mat_b.fp8_meta.te_dtype: {mat_b.fp8_meta.te_dtype} \n \n"
        )
    except AttributeError:
        raise TypeError(f"mat_b.none: {mat_b is None}, mat_b.type: {type(mat_b)}, {mat_b.fp8_meta}")

    # print(f"pipegoose._fp8_kernel.before_matmul, output: {output}, out_dtype: {out_dtype} \n \n")
    print(
        f"pipegoose._fp8_kernel.before_matmul: _empty_tensor: {_empty_tensor}, bias: {bias}, workspace[:5]: {workspace[:5]}, accumulate: {accumulate}, use_split_accumulator: {use_split_accumulator} \n \n"
    )

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

    # print(f"pipegoose._fp8_kernel.after_matmul, mat_a: {mat_a}, mat_a.fp8_meta.inverse_scale: {mat_a.fp8_meta.inverse_scale}, mat_a.fp8_meta.te_dtype: {mat_a.fp8_meta.te_dtype} \n \n")
    # print(f"pipegoose._fp8_kernel.after_matmul, mat_b: {mat_b}, mat_b.fp8_meta.inverse_scale: {mat_b.fp8_meta.inverse_scale}, mat_b.fp8_meta.te_dtype: {mat_b.fp8_meta.te_dtype} \n \n")
    # print(f"pipegoose._fp8_kernel.after_matmul, output: {output}, out_dtype: {out_dtype} \n \n")
    # print(f"pipegoose._fp8_kernel.after_matmul: _empty_tensor: {_empty_tensor}, bias: {bias}, workspace[:5]: {workspace[:5]}, accumulate: {accumulate}, use_split_accumulator: {use_split_accumulator} \n \n")

    # print(f"---------------------------------------------------------------- \n \n")

    return output


@torch.no_grad()
def _fp8_matmul_kernel(
    mat_a: FP8Tensor,
    transpose_a: bool,
    mat_b: FP8Tensor,
    transpose_b: bool,
    bias: FP8Tensor = None,
    transpose_bias: bool = False,
    use_split_accumulator: bool = None,
) -> torch.Tensor:
    assert use_split_accumulator is not None
    assert (
        mat_a.device != "cpu" and mat_b.device != "cpu"
    ), "The tensors must be on a CUDA device in order to use the FP8 kernel!!"

    device = mat_a.device

    _empty_tensor = torch.Tensor()
    output = torch.empty(mat_a.shape[0], mat_b.shape[1], device=device, dtype=torch.float32)
    workspace = torch.empty(33_554_432, dtype=torch.int8, device=device)
    accumulate = False
    # use_split_accumulator = False

    out_dtype = getattr(tex.DType, "kFloat32")
    # TODO(xrsrke): add support for adding bias in fp8
    bias = torch.tensor([], dtype=torch.float32)

    print("---------------------------------------------------------------- \n \n")
    print(
        f"pipegoose._fp8_kernel.before_matmul, mat_a[0, :10]: {mat_a[0, :3]}, mat_a.fp8_meta.inverse_scale: {mat_a.fp8_meta.inverse_scale}, mat_a.fp8_meta.te_dtype: {mat_a.fp8_meta.te_dtype} \n \n"
    )
    print(
        f"pipegoose._fp8_kernel.before_matmul, mat_b[0, :10]: {mat_b[0, :3]}, mat_b.fp8_meta.inverse_scale: {mat_b.fp8_meta.inverse_scale}, mat_b.fp8_meta.te_dtype: {mat_b.fp8_meta.te_dtype} \n \n"
    )
    print(f"pipegoose._fp8_kernel.before_matmul, output[0, :10]: {output[0, :3]}, out_dtype: {out_dtype} \n \n")
    print(
        f"pipegoose._fp8_kernel.before_matmul: _empty_tensor: {_empty_tensor}, bias: {bias}, workspace[:5]: {workspace[:5]}, accumulate: {accumulate}, use_split_accumulator: {use_split_accumulator} \n \n"
    )

    # print(f"pipegoose._fp8_kernel.before_matmul, mat_a: {mat_a}, mat_a.fp8_meta.inverse_scale: {mat_a.fp8_meta.inverse_scale}, mat_a.fp8_meta.te_dtype: {mat_a.fp8_meta.te_dtype} \n \n")
    print(
        f"pipegoose._fp8_kernel.before_matmul, mat_a.fp8_meta.inverse_scale: {mat_a.fp8_meta.inverse_scale}, mat_a.fp8_meta.te_dtype: {mat_a.fp8_meta.te_dtype} \n \n"
    )
    try:
        # print(f"pipegoose._fp8_kernel.before_matmul, mat_b: {mat_b}, mat_b.fp8_meta.inverse_scale: {mat_b.fp8_meta.inverse_scale}, mat_b.fp8_meta.te_dtype: {mat_b.fp8_meta.te_dtype} \n \n")
        print(
            f"pipegoose._fp8_kernel.before_matmul, mat_b.fp8_meta.inverse_scale: {mat_b.fp8_meta.inverse_scale}, mat_b.fp8_meta.te_dtype: {mat_b.fp8_meta.te_dtype} \n \n"
        )
    except AttributeError:
        raise TypeError(f"mat_b.none: {mat_b is None}, mat_b.type: {type(mat_b)}, {mat_b.fp8_meta}")

    # print(f"pipegoose._fp8_kernel.before_matmul, output: {output}, out_dtype: {out_dtype} \n \n")
    print(
        f"pipegoose._fp8_kernel.before_matmul: _empty_tensor: {_empty_tensor}, bias: {bias}, workspace[:5]: {workspace[:5]}, accumulate: {accumulate}, use_split_accumulator: {use_split_accumulator} \n \n"
    )

    mat_a_fp8_meta = mat_a.fp8_meta
    mat_b_fp8_meta = mat_b.fp8_meta

    if transpose_a is True:
        mat_a = tex.fp8_transpose(mat_a, mat_a_fp8_meta.te_dtype)

    if transpose_b is False:
        mat_b = tex.fp8_transpose(mat_b, mat_b_fp8_meta.te_dtype)

    tex.te_gemm(
        mat_b,
        mat_b_fp8_meta.inverse_scale,
        mat_b_fp8_meta.te_dtype,
        True,  # transa, default True
        mat_a,
        mat_a_fp8_meta.inverse_scale,
        mat_a_fp8_meta.te_dtype,
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

    # print(f"pipegoose._fp8_kernel.after_matmul, mat_a: {mat_a}, mat_a.fp8_meta.inverse_scale: {mat_a.fp8_meta.inverse_scale}, mat_a.fp8_meta.te_dtype: {mat_a.fp8_meta.te_dtype} \n \n")
    # print(f"pipegoose._fp8_kernel.after_matmul, mat_b: {mat_b}, mat_b.fp8_meta.inverse_scale: {mat_b.fp8_meta.inverse_scale}, mat_b.fp8_meta.te_dtype: {mat_b.fp8_meta.te_dtype} \n \n")
    # print(f"pipegoose._fp8_kernel.after_matmul, output: {output}, out_dtype: {out_dtype} \n \n")
    # print(f"pipegoose._fp8_kernel.after_matmul: _empty_tensor: {_empty_tensor}, bias: {bias}, workspace[:5]: {workspace[:5]}, accumulate: {accumulate}, use_split_accumulator: {use_split_accumulator} \n \n")

    # print(f"---------------------------------------------------------------- \n \n")

    return output
