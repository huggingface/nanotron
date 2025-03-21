""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information.

Usage:
python examples/config_resume_training.py
"""
import json
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
    NanosetDatasetsArgs,
    OptimizerArgs,
    ParallelismArgs,
    Qwen2Config,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

# Path to the converted SmolLM2-135M checkpoint. See `examples/llama/convert_hf_to_nanotron.py` for more information.
CHECKPOINT_PATH = "./checkpoints/smollm2-135m-nanotron"
TOKENIZER_PATH = "HuggingFaceTB/SmolLM2-135M"

# load from CHECKPOINT_PATH/model_config.json
model_config_dict = json.load(open(f"{CHECKPOINT_PATH}/model_config.json"))
model_config_dict.pop("is_llama_config", None)
model_config = Qwen2Config(**model_config_dict)

# Calculate rough parameter count
dense_layer_count = model_config.num_hidden_layers

# Base parameters (embeddings)
base_params = model_config.vocab_size * model_config.hidden_size * 2

# Dense FFN parameters
dense_ffn_params = dense_layer_count * (3 * model_config.hidden_size * model_config.intermediate_size)

# Attention parameters (same for both dense and MoE layers)
attn_params = model_config.num_hidden_layers * (4 * model_config.hidden_size * model_config.hidden_size)

# Total parameters
total_params = base_params + dense_ffn_params + attn_params

num_params = human_format(total_params).replace(".", "p")

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
            # dataset=PretrainDatasetsArgs(
            #     hf_dataset_or_datasets="HuggingFaceTB/SmolLM2-135M",
            #     text_column_name="text",
            # ),
            dataset=NanosetDatasetsArgs(
                dataset_folder="/fsx/loubna/tokenized_for_exps/mcf-dataset",  # 1.4T tokens
            ),
            # TokenizedBytesDatasetFolderArgs(
            #     folder="/fsx/loubna/tokenized_for_exps/fineweb-edu-400B", # 1.4T tokens
            #     filename_pattern=r".*\.ds$",
            #     shuffle=True,
            #     seed=SEED,
            # ),
            # For SFT (uncomment to use):
            # dataset=SFTDatasetsArgs(
            #     hf_dataset_or_datasets="trl-lib/tldr",
            #     hf_dataset_splits="train",
            #     debug_max_samples=1000,
            # ),
            seed=seed,
        ),
    ),
]

checkpoints_path = "./checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

run_name = "resume_training_%date_%jobid"

config = Config(
    general=GeneralArgs(project="resume_training", run=run_name, seed=seed, ignore_sanity_checks=False),
    checkpoints=CheckpointsArgs(
        checkpoints_path=checkpoints_path,
        checkpoint_interval=10,
        resume_checkpoint_path=CHECKPOINT_PATH,
        load_lr_scheduler=False,
        load_optimizer=False,
    ),
    parallelism=parallelism,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
    tokenizer=TokenizerArgs(tokenizer_name_or_path=TOKENIZER_PATH),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data_stages=data_stages,
    # profiler=ProfilerArgs(profiler_export_path="./tb_logs"),
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    config_filename = "config_resume_training.yaml"
    config.save_as_yaml(f"{dir}/{config_filename}")
    print(f"Config saved to {dir}/{config_filename}")

    # You can now train a model with this config using `/run_train.py`
