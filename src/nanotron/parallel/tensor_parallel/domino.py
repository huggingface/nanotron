"""
Implementation of communication overlapping
in the paper "Domino: Eliminating Communication in LLM Training via
Generic Tensor Slicing and Overlapping"
https://arxiv.org/abs/2409.15241
"""

import re
import threading
from typing import Optional

FWD_MLP_OP_NAME = "fwd.layer_mlp_{}_batch_{}"
FWD_ATTN_OP_NAME = "fwd.layer_attn_{}_batch_{}"
BWD_ATTN_OP_NAME = "bwd.layer_attn_{}_batch_{}"
BWD_MLP_OP_NAME = "bwd.layer_mlp_{}_batch_{}"

_operation_context = threading.local()


def is_domino_async_comm(x: str) -> bool:
    """
    Determine whether a module (e.g., mlp, attention)
    runs all-reduce asynchronously in tensor parallelism
    based on its module name.

    Currently support intra-layer communication overlapping
    as described in domino's input splitting approach.

    How do we determine it?
    + In the forward pass: We run all the forward pass's communication asynchronously
    diagram: https://imgur.com/a/g5Ou2iZ

    + In the backward pass: We run all backward pass's communication asynchronously
    except for the first batch's attention module.
    https://imgur.com/a/MrZb57a
    """
    NON_ASYNC_HANDLE_IDX = [
        "bwd.layer_attn_{}_batch_0",
    ]

    patterns = [p.replace("{}", r"\d+") for p in NON_ASYNC_HANDLE_IDX]  # Replace {} with regex for numbers
    regex = re.compile("^(" + "|".join(patterns) + ")$")  # Combine patterns into a single regex
    not_async = bool(regex.match(x))
    return not not_async


class OpNameContext:
    """Track both forward and backward operation names"""

    def __init__(self, fwd_op_name: str, bwd_op_name: str):
        self.fwd_op_name = fwd_op_name
        self.bwd_op_name = bwd_op_name
        self.prev_fwd = None
        self.prev_bwd = None

    def __enter__(self):
        self.prev_fwd = getattr(_operation_context, "current_fwd_op_name", None)
        self.prev_bwd = getattr(_operation_context, "current_bwd_op_name", None)
        _operation_context.current_fwd_op_name = self.fwd_op_name
        _operation_context.current_bwd_op_name = self.bwd_op_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _operation_context.current_fwd_op_name = self.prev_fwd
        _operation_context.current_bwd_op_name = self.prev_bwd


def get_current_fwd_op_name() -> Optional[str]:
    return getattr(_operation_context, "current_fwd_op_name", None)


def get_current_bwd_op_name() -> Optional[str]:
    return getattr(_operation_context, "current_bwd_op_name", None)
