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
    performs all-reduce asynchronously in tensor parallelism
    """
    NON_ASYNC_HANDLE_IDX = [
        # "fwd.layer_mlp_{}_batch_1",
        "bwd.layer_attn_{}_batch_0",
    ]

    patterns = [p.replace("{}", r"\d+") for p in NON_ASYNC_HANDLE_IDX]  # Replace {} with regex for numbers
    regex = re.compile("^(" + "|".join(patterns) + ")$")  # Combine patterns into a single regex
    not_async = bool(regex.match(x))
    return not not_async


class OpNameContext:
    """
    A context manager to set the name of a module operation
    """

    def __init__(self, op_name: str):
        # TODO: support passing stream_manager as a part of an operation context
        self.op_name = op_name
        self.previous_op_name = None

    def __enter__(self):
        if not hasattr(_operation_context, "current_op_name"):
            _operation_context.current_op_name = None
        self.previous_op_name = _operation_context.current_op_name
        _operation_context.current_op_name = self.op_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _operation_context.current_op_name = self.previous_op_name


def get_op_name() -> Optional[str]:
    """
    Get the name of the current operation.
    """
    return getattr(_operation_context, "current_op_name", None)
