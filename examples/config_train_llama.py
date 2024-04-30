""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os

from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LlamaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

model_config = LlamaConfig(
    # Config for a tiny model model with 1.62M parameters
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    max_position_embeddings=512,
    num_attention_heads=16,
    num_hidden_layers=12,
    num_key_value_heads=16,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=50272,
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
    tp=2,
    pp_engine="1f1b",
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
)

# a global batch-size of 1M tokens.  micro_batch_size * dp * sequence_length * batch_accumulation_per_replica
tokens = TokensArgs(sequence_length=512, train_steps=200, micro_batch_size=128, batch_accumulation_per_replica=8)

checkpoints_path = os.path.dirname(os.path.dirname(__file__)) + "/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = Config(
    general=GeneralArgs(project="debug", run="tiny_llama_%date_%jobid", seed=seed),
    checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=10),
    parallelism=parallelism,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
    tokenizer=TokenizerArgs("gpt2"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data_stages=[
        DatasetStageArgs(
            name="Stable Training Stage",
            start_training_step=1,
            data=DataArgs(
                dataset=PretrainDatasetsArgs(hf_dataset_or_datasets="roneneldan/TinyStories", text_column_name="text"),
                seed=seed,
            ),
        )
    ],
    profiler=None,
)


if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    config.save_as_yaml(f"{dir}/config_train_llama.yaml")

    # You can now train a model with this config using `/run_train.py`
