""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os

from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    MoEConfig,
    OptimizerArgs,
    ParallelismArgs,
    Qwen2Config,
    RandomInit,
    SFTDatasetsArgs,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

# Create a MoE-enabled Qwen2 model with 4 experts
model_config = Qwen2Config(
    # Config for a tiny model model with MoE layers
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=192,
    initializer_range=0.02,
    intermediate_size=768,
    max_position_embeddings=256,
    num_attention_heads=4,
    num_hidden_layers=2,
    num_key_value_heads=4,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=256,
    is_qwen2_config=True,
    pad_token_id=None,
    # MoE configuration using the new dataclass
    moe_config=MoEConfig(
        num_experts=4,  # Total number of experts
        top_k=2,  # Number of experts to route each token to
        layers=[1],  # The second layer (index 1) will be a MoE layer
        enable_shared_expert=True,  # Use a shared expert alongside specialized experts
        token_dispatcher_type="alltoall",  # Communication pattern for distributed experts
    ),
)

# Calculate rough parameter count - now with MoE consideration
moe_layer_count = len(model_config.moe_config.layers) if model_config.moe_config else 0
dense_layer_count = model_config.num_hidden_layers - moe_layer_count

# Base parameters (embeddings)
base_params = model_config.vocab_size * model_config.hidden_size * 2

# Dense FFN parameters
dense_ffn_params = dense_layer_count * (3 * model_config.hidden_size * model_config.intermediate_size)

# MoE FFN parameters (experts × parameters per expert)
moe_ffn_params = 0
shared_expert_params = 0
router_params = 0

if model_config.moe_config:
    # MoE FFN parameters (experts × parameters per expert)
    moe_ffn_params = (
        moe_layer_count
        * model_config.moe_config.num_experts
        * (3 * model_config.hidden_size * model_config.intermediate_size)
    )

    # Shared expert parameters if enabled
    shared_expert_params = (
        moe_layer_count * (3 * model_config.hidden_size * model_config.intermediate_size)
        if model_config.moe_config.enable_shared_expert
        else 0
    )

    # Router parameters
    router_params = moe_layer_count * (model_config.hidden_size * model_config.moe_config.num_experts)

# Attention parameters (same for both dense and MoE layers)
attn_params = model_config.num_hidden_layers * (4 * model_config.hidden_size * model_config.hidden_size)

# Total parameters
total_params = base_params + dense_ffn_params + moe_ffn_params + shared_expert_params + router_params + attn_params

num_params = human_format(total_params).replace(".", "p")

print(f"Model has {num_params} parameters")
if model_config.moe_config:
    print(f"MoE layers: {model_config.moe_config.layers}")
    print(f"Number of experts: {model_config.moe_config.num_experts}")

seed = 42

learning_rate = LRSchedulerArgs(
    learning_rate=3e-4, lr_warmup_steps=2, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
)

optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=True,
    learning_rate_scheduler=learning_rate,
    optimizer_factory=AdamWOptimizerArgs(
        adam_eps=1e-08,
        adam_beta1=0.9,
        adam_beta2=0.95,
        torch_adam_is_fused=True,
    ),
)

parallelism = ParallelismArgs(
    dp=2,
    pp=1,
    tp=1,
    context_parallel_size=1,
    expert_parallel_size=1,
    pp_engine="1f1b",
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
    moe_layer_recompute=True,  # Add recompute option for MoE layers
)

tokens = TokensArgs(sequence_length=256, train_steps=15, micro_batch_size=2, batch_accumulation_per_replica=1)

data_stages = [
    DatasetStageArgs(
        name="Stable Training Stage",
        start_training_step=1,
        data=DataArgs(
            # For pretraining:
            # dataset=PretrainDatasetsArgs(
            #     hf_dataset_or_datasets="trl-lib/tldr",
            #     text_column_name="text",
            # ),
            # For SFT (uncomment to use):
            dataset=SFTDatasetsArgs(
                hf_dataset_or_datasets="trl-lib/tldr",
                hf_dataset_splits="train",
                sft_dataloader=True,
                debug_max_samples=1000,
            ),
            seed=seed,
        ),
    ),
]

checkpoints_path = "./checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

# Determine run name based on whether MoE is enabled
run_name = "tiny_qwen_moe_%date_%jobid" if model_config.is_moe_model else "tiny_qwen_%date_%jobid"

config = Config(
    general=GeneralArgs(project="debug", run=run_name, seed=seed, ignore_sanity_checks=False),
    checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=10),
    parallelism=parallelism,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
    tokenizer=TokenizerArgs("robot-test/dummy-tokenizer-wordlevel"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data_stages=data_stages,
    # profiler=ProfilerArgs(profiler_export_path="./tb_logs"),
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file - name depends on whether MoE is enabled
    config_filename = "config_tiny_qwen_moe.yaml" if model_config.is_moe_model else "config_tiny_qwen.yaml"
    config.save_as_yaml(f"{dir}/{config_filename}")

    # You can now train a model with this config using `/run_train.py`
