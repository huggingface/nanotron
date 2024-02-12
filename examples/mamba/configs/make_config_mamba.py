""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os
import torch

from nanotron.config import (
    CheckpointsArgs,
    Config,
    DataArgs,
    GeneralArgs,
    MambaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    MambaInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

model_config = MambaConfig(
    d_model=256,
    num_hidden_layers=1,
    vocab_size=50277,
    ssm_cfg={},
    rms_norm=True,
    fused_add_norm=True,
    residual_in_fp32=True,
    pad_vocab_size_multiple=8,
    # Custom
    dtype=torch.float32,
    rms_norm_eps=1e-5,
)


#TODO(fmom): do something similar
# num_params = human_format(
#     model_config.vocab_size * model_config.d_model * 2
#     + model_config.num_hidden_layers
#     * (
#         3 * model_config.d_model * model_config.intermediate_size
#         + 4 * model_config.d_model * model_config.d_model
#     )
# ).replace(".", "p")

# print(f"Model has {num_params} parameters")

seed = 42

learning_rate = LRSchedulerArgs(
    learning_rate=3e-4, lr_warmup_steps=2, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
)

optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=False, #NOTE(fmom): because we are using PP=TP=DP=1
    adam_eps=1e-08,
    adam_beta1=0.9,
    adam_beta2=0.95,
    torch_adam_is_fused=True,
    learning_rate_scheduler=learning_rate,
)

parallelism = ParallelismArgs(
    dp=1,
    pp=1,
    tp=1,
    pp_engine="1f1b",
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
    recompute_granularity="selective",
)

tokens = TokensArgs(sequence_length=1024, train_steps=40, micro_batch_size=2, batch_accumulation_per_replica=1)

dataset = PretrainDatasetsArgs(
    hf_dataset_or_datasets="stas/openwebtext-10k", text_column_name="text"
)

checkpoints_path = os.path.dirname(os.path.dirname(__file__)) + "/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = Config(
    general=GeneralArgs(project="test", run="mamba", seed=seed),
    checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=243232232232323332),
    parallelism=parallelism,
    model=ModelArgs(init_method=MambaInit(initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1), model_config=model_config),
    tokenizer=TokenizerArgs("gpt2"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data=DataArgs(dataset=dataset, seed=seed),
    profiler=None,
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    config.save_as_yaml(f"{dir}/config_mamba.yaml")

    # You can now train a model with this config using `/run_train.py`
