"""
Benchmarking script for the Llama-2-7b model
"""

import os

from nanotron.config import (
    CheckpointsArgs,
    Config,
    DataArgs,
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

# Config for a llama model with 6.74M parameters
model_config = LlamaConfig()

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
    adam_eps=1e-08,
    adam_beta1=0.9,
    adam_beta2=0.95,
    torch_adam_is_fused=True,
    learning_rate_scheduler=learning_rate,
)

parallelism = ParallelismArgs(
    dp=2,
    pp=1,
    tp=4,
    pp_engine="1f1b",
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
)

tokens = TokensArgs(sequence_length=8192, train_steps=5, micro_batch_size=1, batch_accumulation_per_replica=8)

dataset = PretrainDatasetsArgs(hf_dataset_or_datasets="stas/openwebtext-10k", text_column_name="text")

checkpoints_path = os.path.dirname(os.path.dirname(__file__)) + "/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = Config(
    general=GeneralArgs(project="bench", run="llama", seed=seed),
    checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=1000),
    parallelism=parallelism,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
    tokenizer=TokenizerArgs("meta-llama/Llama-2-7b-hf"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data=DataArgs(dataset=dataset, seed=seed),
    profiler=None,
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    config.save_as_yaml(f"{dir}/config_llama.yaml")

    # Launch training
    os.system("export CUDA_DEVICE_MAX_CONNECTIONS=1")
    gpus = config.parallelism.dp * config.parallelism.pp * config.parallelism.tp
    os.system(f"torchrun --nproc_per_node={gpus} run_train.py --config-file {dir}/config_llama.yaml")
