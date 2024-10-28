from typing import Optional, Tuple

import torch

from nanotron.fp8.linear import FP8LinearMeta
from nanotron.fp8.recipe import FP8LinearRecipe
from nanotron.fp8.tensor import FP8Tensor


def smooth_quant(input: torch.Tensor, weight: FP8Tensor, alpha: float) -> Tuple[torch.Tensor, FP8Tensor]:
    """
    An implementation of SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
    https://arxiv.org/abs/2211.10438
    """
    # Compute smoothing factor
    input_s = torch.amax(torch.abs(input), dim=(0, 1), keepdim=True)
    w_s = torch.amax(torch.abs(weight._orig_data_after_set_data), dim=0)

    s = input_s.squeeze().pow(alpha) / w_s.pow(1 - alpha)

    # NOTE: create a smoothed tensor without adding the smoothing operations
    # to computational graph, and keep the original computational graph
    X_smoothed = input.detach() / s.unsqueeze(dim=0).unsqueeze(dim=0)
    X_smoothed.requires_grad_()

    with torch.no_grad():
        W_smoothed = weight._orig_data_after_set_data * s.unsqueeze(0)
    weight.set_data(W_smoothed)

    return X_smoothed, weight


def linear(
    input: torch.Tensor,
    weight: FP8Tensor,
    bias: Optional[torch.Tensor] = None,
    metadatas: FP8LinearMeta = None,
    recipe: FP8LinearRecipe = None,
    name: Optional[str] = None,
):
    from typing import cast

    from nanotron import constants
    from nanotron.config.fp8_config import FP8Args

    if recipe.smooth_quant is True:
        fp8_config = cast(FP8Args, constants.CONFIG.fp8)
        migration_strength = fp8_config.smooth_quant_migration_strength
        input, weight = smooth_quant(input, weight, alpha=migration_strength)

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
    output = torch.zeros(input.shape[0], weight.shape[0], device="cuda", dtype=recipe.accum_dtype)
    output, _ = _FP8Matmul.apply(input, weight, output, phony, metadatas, recipe, name)

    # TODO(xrsrke): add support for adding bias in fp8
    # TODO(xrsrke): support return an fp8 tensor as output
    # since we will quantize it back to FP8 anyway in the next linear
    output = rearrange(output, "(b n) h -> b n h", n=seq_len, b=batch_size) if is_input_flat is True else output
    output = output if bias is None else output + bias
    return output
