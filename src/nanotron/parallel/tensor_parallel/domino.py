import re

FWD_MLP_HANDLE_IDX = "fwd.layer_mlp_{}_batch_{}"
FWD_ATTN_HANDLE_IDX = "fwd.layer_attn_{}_batch_{}"
BWD_ATTN_HANDLE_IDX = "bwd.layer_attn_{}_batch_{}"
BWD_MLP_HANDLE_IDX = "bwd.layer_mlp_{}_batch_{}"


def is_async_comm(x):
    NON_ASYNC_HANDLE_IDX = [
        # NOTE: execute all fwd's comm in async
        # "fwd.layer_mlp_{}_batch_1",
        # "bwd.layer_mlp_{}_batch_1",
        "bwd.layer_attn_{}_batch_0",
    ]
    # assert "fwd." not in x

    patterns = [p.replace("{}", r"\d+") for p in NON_ASYNC_HANDLE_IDX]  # Replace {} with regex for numbers
    regex = re.compile("^(" + "|".join(patterns) + ")$")  # Combine patterns into a single regex
    not_async = bool(regex.match(x))
    return not not_async
