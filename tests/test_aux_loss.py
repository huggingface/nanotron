import torch
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import MoEConfig, Qwen2Config
from nanotron.nn.load_balancing_loss import switch_aux_loss
from nanotron.nn.moe import Router
from torch.nn.functional import softmax


def test_switch_aux_loss_basic():
    probs = softmax(torch.randn(10, 4), dim=-1)  # 10 tokens, 4 experts
    tokens_per_expert = torch.tensor([3, 3, 2, 2], dtype=torch.float32)
    aux_loss_coeff = 0.1
    top_k = 1

    loss = switch_aux_loss(probs, tokens_per_expert, aux_loss_coeff, top_k)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar tensor
    assert loss.item() >= 0  # loss should be non-negative


def test_switch_aux_loss():
    num_tokens = 4
    num_experts = 6
    top_k = 3
    hidden_size = 8
    router = _create_router(num_experts, top_k, hidden_size)

    logits = torch.randn(num_tokens, num_experts, device="cuda")

    # copy logics from apply_aux_loss
    # need to pay attention to the top k
    _, routing_indices = router.top_k_softmax(logits)
    tokens_per_expert = routing_indices.sum(dim=0)
    total_tokens = num_tokens * top_k
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32)

    # not use the top k version
    assert scores.shape == (num_tokens, num_experts)
    assert routing_indices.shape == (num_tokens, num_experts)
    assert tokens_per_expert.shape == (num_experts,)
    assert tokens_per_expert.sum() == total_tokens

    # copy logics aux loss
    total_num_tokens = scores.shape[0]
    experts_num = scores.shape[1]
    assert total_num_tokens == num_tokens
    assert experts_num == num_experts
    assert scores.dtype == torch.float32
    assert torch.allclose(scores.sum(dim=1), torch.ones(num_tokens, device=scores.device))

    aggregated_probs_per_expert = scores.sum(dim=0)
    assert aggregated_probs_per_expert.shape == (num_experts,)

    aux_loss_coeff = 1
    aux_loss = (
        aux_loss_coeff
        * num_experts
        * torch.sum(aggregated_probs_per_expert * tokens_per_expert)
        / (total_num_tokens**2 * top_k)
    )
    assert aux_loss.shape == ()
    assert aux_loss.item() >= 1.0


def _create_router(num_experts, top_k, hidden_size):
    config = Qwen2Config(
        bos_token_id=0,
        eos_token_id=1,
        hidden_act="silu",
        hidden_size=hidden_size,
        initializer_range=0.02,
        intermediate_size=16,
        is_qwen2_config=True,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=1,
        pad_token_id=None,
        attention_bias=True,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        rope_theta=10000.0,
        rope_interleaved=False,
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=100,
        moe_config=MoEConfig(
            top_k=top_k,
            num_experts=num_experts,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=32,
            aux_loss_coeff=0.01,
            enable_shared_expert=True,
            z_loss_coeff=1e-5,
            layers=[-1],
        ),
    )

    parallel_config = ParallelismArgs(
        dp=1,
        tp=1,
        pp=1,
    )

    router = Router(config=config, parallel_config=parallel_config, tp_pg=None, layer_idx=0)
    return router


def test_router():
    num_tokens = 2
    num_experts = 6
    top_k = 2
    hidden_size = 8
    router = _create_router(num_experts, top_k, hidden_size)

    x = torch.randn(num_tokens, hidden_size, device="cuda")

    routing_weights, routing_indices = router(x)

    # Assertions
    assert routing_weights.shape == (num_tokens, num_experts)
    assert routing_indices.shape == (num_tokens, num_experts)
    assert torch.all(routing_weights >= 0)
    assert torch.all(torch.sum(routing_weights, dim=1) < 1)
    assert torch.all(routing_indices >= 0)
    assert torch.all(routing_indices < num_experts)
    assert routing_weights.dtype == torch.float32


if __name__ == "__main__":
    test_switch_aux_loss_basic()
    test_router()
    test_switch_aux_loss()
