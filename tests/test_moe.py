import torch
from helpers.qwen_helper import TINY_MOE_QWEN_CONFIG
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.models.base import init_on_device_and_dtype
from nanotron.nn.moe import GroupedMLP


def test_grouped_mlp():
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        expert_parallel_size=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )
    num_tokens_per_experts = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    NUM_TOKENS = num_tokens_per_experts.sum()
    NUM_EXPERTS = TINY_MOE_QWEN_CONFIG.moe_config.num_experts
    HIDDEN_SIZE = TINY_MOE_QWEN_CONFIG.hidden_size
    permuted_hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    assert len(num_tokens_per_experts) == NUM_EXPERTS

    with init_on_device_and_dtype(device=torch.device("cuda"), dtype=torch.bfloat16):
        grouped_mlp = GroupedMLP(config=TINY_MOE_QWEN_CONFIG, parallel_config=parallel_config)

    output = grouped_mlp(permuted_hidden_states, num_tokens_per_experts)

    assert output["hidden_states"].shape == (NUM_TOKENS, HIDDEN_SIZE)
    assert output["hidden_states"].dtype == torch.bfloat16
    assert output["hidden_states"].device.type == "cuda"


if __name__ == "__main__":
    test_grouped_mlp()
