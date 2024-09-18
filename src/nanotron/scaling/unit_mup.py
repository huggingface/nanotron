from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

from unit_scaling.constraints import apply_constraint
from unit_scaling.scale import scale_fwd, scale_bwd
from unit_scaling.core.functional import logarithmic_interpolation
from math import log

def linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    orig_shape: Tuple[int, int],
    scale_power,
    # constraint: Optional[str] = "to_output_scale",
    constraint: Optional[str],
    # scale_power: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tensor:
    # fan_out, fan_in = weight.shape
    fan_in, fan_out = orig_shape
    
    # TODO(xrsrke): use the actual batch size in distributed training
    from nanotron.constants import CONFIG
    # batch_size = input.numel() // fan_in
    batch_size = CONFIG.tokens.micro_batch_size

    output_scale = 1 / fan_in ** scale_power[0]
    grad_input_scale = 1 / fan_out ** scale_power[1]
    grad_weight_scale = grad_bias_scale = 1 / batch_size ** scale_power[2]

    if constraint is None:
        assert 1 == 1

    output_scale, grad_input_scale = apply_constraint(
        constraint, output_scale, grad_input_scale
    )

    input = scale_bwd(input, grad_input_scale)
    weight = scale_bwd(weight, grad_weight_scale)
    bias = scale_bwd(bias, grad_bias_scale) if bias is not None else None
    output = F.linear(input, weight, bias)
    output = scale_fwd(output, output_scale)
    return output


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    mult: float = 1.0,
) -> Tensor:
    *_, seq_len, d_head = value.shape
    # Empirical model of attention output std given mult and seq_len
    scale = (1 - dropout_p) ** 0.5 / logarithmic_interpolation(
        alpha=1 / (1 + 4 * d_head / mult**2),  # = sigmoid(log(mult**2 / (4 * d_head)))
        lower=((log(seq_len) if is_causal else 1) / seq_len) ** 0.5,
        upper=1.0,
    )
    query, key, value = (scale_bwd(t, scale) for t in (query, key, value))
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=mult / d_head,
    )
    return scale_fwd(out, scale)
