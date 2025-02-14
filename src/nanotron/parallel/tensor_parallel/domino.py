import re

FWD_MLP_HANDLE_IDX = "fwd.layer_mlp_{}_batch_{}"
FWD_ATTN_HANDLE_IDX = "fwd.layer_attn_{}_batch_{}"
BWD_ATTN_HANDLE_IDX = "bwd.layer_attn_{}_batch_{}"
BWD_MLP_HANDLE_IDX = "bwd.layer_mlp_{}_batch_{}"


def is_async_comm(op_name: str):
    """
    There are two operations that we can't overlap
    for the forward pass: the last micro-batch of the mlp layer
    for the backward pass: the first micro-batch of the attention layer
    """
    NON_ASYNC_HANDLE_IDX = [
        "fwd.layer_mlp_{}_batch_1",
        "bwd.layer_attn_{}_batch_0",
    ]

    patterns = [p.replace("{}", r"\d+") for p in NON_ASYNC_HANDLE_IDX]  # Replace {} with regex for numbers
    regex = re.compile("^(" + "|".join(patterns) + ")$")  # Combine patterns into a single regex
    not_async = bool(regex.match(op_name))
    return not not_async
