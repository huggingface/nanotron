from typing import Optional

import torch

from nanotron.fp8.linear import FP8LinearMeta
from nanotron.fp8.recipe import FP8LinearRecipe
from nanotron.parallel.parameters import NanotronParameter


def linear(
    input: torch.Tensor,
    weight: NanotronParameter,
    bias: Optional[torch.Tensor] = None,
    metadatas: FP8LinearMeta = None,
    recipe: FP8LinearRecipe = None,
    name: Optional[str] = None,
):
    assert isinstance(weight, NanotronParameter)
    from typing import cast

    from nanotron import constants
    from nanotron.config.fp8_config import FP8Args

    assert metadatas is not None, "metadatas must be specified"
    assert recipe is not None, "recipe must be specified"
    assert input.device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"

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
    # NOTE: interesting that if i initialize the output buffer as torch.empty
    # it leads to nan matmul, so i do torch.zeros instead
    # output = torch.empty(input.shape[0], weight.shape[0], device="cuda", dtype=recipe.accum_dtype)
    output = torch.zeros(input.shape[0], weight.shape[0], device="cuda", dtype=recipe.accum_dtype)
    output, _ = _FP8Matmul.apply(input, weight, output, phony, metadatas, recipe, name)

    # TODO(xrsrke): add support for adding bias in fp8
    # TODO(xrsrke): support return an fp8 tensor as output
    # since we will quantize it back to FP8 anyway in the next linear
    output = rearrange(output, "(b n) h -> b n h", n=seq_len, b=batch_size) if is_input_flat is True else output
    output = output if bias is None else output + bias
    return output
