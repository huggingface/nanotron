from typing import Optional, Union

import torch

from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8LinearMeta
from nanotron.fp8.recipe import FP8LinearRecipe
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
    out: torch.Tensor,
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

    out = mm(input=mat1, mat2=mat2, out=out, accum_qtype=accum_qtype, metadatas=metadatas)
    out = out if input is None else out + input
    return out


def smooth_quant(input, weight, alpha=0.5):
    # Compute smoothing factor
    # s = torch.max(torch.abs(X), dim=-1)[0].pow(alpha) / torch.max(torch.abs(W), dim=-1)[0].pow(1-alpha)
    input_s = torch.amax(torch.abs(input), dim=(0, 1), keepdim=True)
    w_s = torch.amax(torch.abs(weight._orig_data_after_set_data), dim=0)

    s = input_s.squeeze().pow(0.5) / w_s.pow(1 - 0.5)

    # Apply smoothing
    X_smoothed = input.detach() / s.unsqueeze(dim=0).unsqueeze(dim=0)
    # X_smoothed.requires_grad = input.requires_grad
    X_smoothed.requires_grad_()

    with torch.no_grad():
        W_smoothed = weight._orig_data_after_set_data * s.unsqueeze(0)
    weight.set_data(W_smoothed)

    return X_smoothed, weight


def linear(
    input: torch.Tensor,
    weight: FP8Tensor,
    bias: Optional[torch.Tensor] = None,
    # accum_qtype: DTypes = None,
    metadatas: FP8LinearMeta = None,
    recipe: FP8LinearRecipe = None,
    name: Optional[str] = None,
):
    if recipe.actsmooth is True:
        _orig_input = input
        _orig_weight = weight
        input, weight = smooth_quant(input, weight)

        # input.requires_grad = _orig_input.requires_grad
        # if input.requires_grad:
        #     input.grad_fn = _orig_input.grad_fn

    # assert accum_qtype is not None, "accum_qtype must be specified"
    assert metadatas is not None, "metadatas must be specified"
    assert recipe is not None, "recipe must be specified"
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
    output = torch.zeros(input.shape[0], weight.shape[0], device="cuda", dtype=recipe.accum_dtype)
    output, _ = _FP8Matmul.apply(input, weight, output, phony, metadatas, recipe, name)

    # TODO(xrsrke): add support for adding bias in fp8
    # TODO(xrsrke): support return an fp8 tensor as output
    # since we will quantize it back to FP8 anyway in the next linear
    output = rearrange(output, "(b n) h -> b n h", n=seq_len, b=batch_size) if is_input_flat is True else output
    output = output if bias is None else output + bias
    return output
