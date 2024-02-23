""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information.

Usage:
```
python config_tiny_mixtral.py
```
"""
import os

from config_mixtral import MixtralConfig, get_num_params
from nanotron.config import (
    CheckpointsArgs,
    Config,
    DataArgs,
    GeneralArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

MODEL_CONFIG = MixtralConfig(
    # Config for Mixtral 7B
    attn_pdrop=0.0,
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=1024,
    initializer_range=0.02,
    intermediate_size=3584,
    max_position_embeddings=131072,
    num_attention_heads=32,
    num_hidden_layers=2,
    num_key_value_heads=8,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_theta=10000.0,
    sliding_window_size=4096,
    tie_word_embeddings=False,
    use_cache=True,
    vocab_size=32000,
    # MoE specific config
    num_experts_per_tok=2,
    moe_num_experts=2,
)

num_params = human_format(get_num_params(MODEL_CONFIG)).replace(".", "p")

print(f"Model has {num_params} parameters")

PARALLELISM = ParallelismArgs(
    dp=1,
    pp=1,
    tp=2,
    expert_parallel_size=2,
    pp_engine="1f1b",
    tp_mode="ALL_REDUCE",
    tp_linear_async_communication=False,
)

tokens = TokensArgs(sequence_length=256, train_steps=1918, micro_batch_size=2, batch_accumulation_per_replica=1)
checkpoints = CheckpointsArgs(checkpoints_path="checkpoints/", checkpoint_interval=100000)
optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=True,
    adam_eps=1e-08,
    adam_beta1=0.9,
    adam_beta2=0.95,
    torch_adam_is_fused=True,
    learning_rate_scheduler=LRSchedulerArgs(
        learning_rate=3e-4, lr_warmup_steps=100, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
    ),
)
data = DataArgs(
    seed=0,
    num_loading_workers=1,
    dataset=None
    # dataset=PretrainDatasetsArgs(
    #     hf_dataset_or_datasets="roneneldan/TinyStories",
    #     hf_dataset_splits="train",
    #     text_column_name="text",
    #     dataset_processing_num_proc_per_process=12,
    # ),
)


CONFIG = Config(
    general=GeneralArgs(project="mixtralai", run="Mixtral-7B-v0.1", seed=42, step=0),
    parallelism=PARALLELISM,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=MODEL_CONFIG),
    tokenizer=TokenizerArgs("mixtralai/Mixtral-7B-v0.1"),
    checkpoints=checkpoints,
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data=data,
    profiler=None,
    lighteval=None,
)

if __name__ == "__main__":
    file_path = os.path.abspath(__file__)

    file_path = file_path.replace(".py", ".yaml").replace("_tiny", "")
    # Save config as YAML file
    CONFIG.save_as_yaml(file_path)

    # You can now train a model with this config using `/run_train.py`
