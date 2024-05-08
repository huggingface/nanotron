from typing import Optional, Union

import torch

from nanotron.fp8.constants import QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8LinearMeta
from nanotron.fp8.tensor import FP8Tensor


def mm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    accum_qtype: DTypes,
    metadatas: FP8LinearMeta,
    out: torch.Tensor,
):
    """
    It would be nicer to use output as argument name, but pytorch use "out", so to consistent with pytorch APIs, we use "out" here!
    NOTE: we assume that mat2 is transposed, yea this is weird, will replace this with a triton kernel.
    """
    from einops import rearrange

    from nanotron.fp8.linear import _FP8Matmul

    seq_len = None
    batch_size = None
    is_input_flat = False
    if input.ndim == 3:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        is_input_flat = True
        input = rearrange(input, "b n h -> (b n) h")
    elif input.ndim > 3:
        raise ValueError(f"Unsupported input shape: {input.shape}")

    # NOTE: just a phony tensor to make pytorch trigger the backward pass
    # because weight and bias's requires_grad are set to False
    # so that we can compute the gradients using the fp8 kernels by ourselves
    phony = torch.empty(0, device=input.device, requires_grad=True)
    output, _ = _FP8Matmul.apply(input, mat2, out, phony, metadatas, accum_qtype)

    # TODO(xrsrke): add support for adding bias in fp8
    # TODO(xrsrke): support return an fp8 tensor as output
    # since we will quantize it back to FP8 anyway in the next linear
    output = rearrange(output, "(b n) h -> b n h", n=seq_len, b=batch_size) if is_input_flat is True else output
    return output


def addmm(
    input,
    mat1,
    mat2,
    output: torch.Tensor,
    accum_qtype: DTypes,
    metadatas: FP8LinearMeta,
    beta: Union[float, int] = 1,
    alpha: Union[float, int] = 1,
):
    """
    NOTE: we assume that mat2 is transposed, yea this is weird, will replace this with a triton kernel.
    """
    assert beta == 1.0, "Currently only support beta=1."
    assert alpha == 1.0, "Currently only support alpha=1."

    output = mm(input=mat1, mat2=mat2, out=output, accum_qtype=accum_qtype, metadatas=metadatas)
    output = output if input is None else output + input
    return output


def linear(
    input: torch.Tensor,
    weight: FP8Tensor,
    bias: Optional[torch.Tensor] = None,
    accum_qtype: DTypes = None,
    metadatas: FP8LinearMeta = None,
):
    assert accum_qtype is not None, "accum_qtype must be specified"
    assert metadatas is not None, "metadatas must be specified"
    assert input.device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"
    # return addmm(input=bias, mat1=input, mat2=weight.transpose_fp8(), output=output, accum_qtype=accum_qtype, metadatas=metadatas)

    # TODO(xrsrke): refactor this out, don't duplicate the code
    from einops import rearrange

    from nanotron.fp8.linear import _FP8Matmul

    seq_len = None
    batch_size = None
    is_input_flat = False
    if input.ndim == 3:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        is_input_flat = True
        input = rearrange(input, "b n h -> (b n) h")
    elif input.ndim > 3:
        raise ValueError(f"Unsupported input shape: {input.shape}")

    # NOTE: just a phony tensor to make pytorch trigger the backward pass
    # because weight and bias's requires_grad are set to False
    # so that we can compute the gradients using the fp8 kernels by ourselves
    phony = torch.empty(0, device=input.device, requires_grad=True)
    output = torch.zeros(input.shape[0], weight.shape[0], device="cuda", dtype=QTYPE_TO_DTYPE[accum_qtype])
    output, _ = _FP8Matmul.apply(input, weight, output, phony, metadatas, accum_qtype)

    # TODO(xrsrke): add support for adding bias in fp8
    # TODO(xrsrke): support return an fp8 tensor as output
    # since we will quantize it back to FP8 anyway in the next linear
    output = rearrange(output, "(b n) h -> b n h", n=seq_len, b=batch_size) if is_input_flat is True else output
    output = output if bias is None else output + bias
    return output

    # output = torch.zeros(input.shape[0], weight.shape[1], device="cuda", dtype=QTYPE_TO_DTYPE[accum_qtype])
    # output = addmm(input=bias, mat1=input, mat2=weight, output=output, accum_qtype=accum_qtype, metadatas=metadatas)
    # return output
