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
    OptimizerArgs,
    ParallelismArgs,
    Qwen2Config,
    RandomInit,
    SFTDatasetsArgs,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

model_config = Qwen2Config(
    # Config for a tiny model model with 1.62M parameters
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
)

num_params = human_format(
    model_config.vocab_size * model_config.hidden_size * 2
    + model_config.num_hidden_layers
    * (
        3 * model_config.hidden_size * model_config.intermediate_size
        + 4 * model_config.hidden_size * model_config.hidden_size
    )
).replace(".", "p")

print(f"Model has {num_params} parameters")

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

config = Config(
    general=GeneralArgs(project="debug", run="tiny_qwen_%date_%jobid", seed=seed, ignore_sanity_checks=False),
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

    # Save config as YAML file
    config.save_as_yaml(f"{dir}/config_tiny_qwen.yaml")

    # You can now train a model with this config using `/run_train.py`
