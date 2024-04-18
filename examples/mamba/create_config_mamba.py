""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import math
import os
import uuid

from config import MambaConfig, MambaInit, MambaModelConfig
from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

new_job_id = uuid.uuid4()
job_id = str(new_job_id)[:8]
seed = 42

ssm_cfg_dtype = "bfloat16"
ssm_cfg = {
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "dt_rank": "auto",
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init": "random",
    "dt_scale": 1.0,
    "dt_init_floor": 1e-4,
    "conv_bias": True,
    "bias": False,
    "use_fast_path": True,
}
# https://huggingface.co/state-spaces/mamba-790m/blob/main/config.json
model_config = MambaModelConfig(
    d_model=1024,
    num_hidden_layers=2,
    vocab_size=50278,
    ssm_cfg=ssm_cfg,
    rms_norm=True,
    fused_add_norm=True,
    residual_in_fp32=True,
    pad_vocab_size_multiple=8,
    # Custom
    dtype=ssm_cfg_dtype,
    rms_norm_eps=1e-5,
)

# NOTE: vocab_size is normally round up to the nearest multiple of 10. But here, we don't really care
tie_embedding = model_config.vocab_size * model_config.d_model  # model_config.vocab_size * model_config.d_model
expand = 2 if ("expand" not in ssm_cfg) else ssm_cfg["expand"]
ngroups = 1 if ("ngroups" not in ssm_cfg) else ssm_cfg["ngroups"]
d_state = 16 if ("d_state" not in ssm_cfg) else ssm_cfg["d_state"]
d_conv = 4 if ("d_conv" not in ssm_cfg) else ssm_cfg["d_conv"]
dt_rank = (
    math.ceil(model_config.d_model / 16)
    if ("dt_rank" not in ssm_cfg or ssm_cfg["dt_rank"] == "auto")
    else ssm_cfg["dt_rank"]
)

d_inner = int(expand * model_config.d_model)
in_proj = model_config.d_model * d_inner * 2

# conv1d.weight = out_channels * (in_channels // groups) * kernel_size
# conv1d.bias = out_channels
conv1d = d_inner * int(d_inner / d_inner) * d_conv + d_inner
# linear.weight = out_features * in_features
in_proj = model_config.d_model * d_inner * 2 + 0
x_proj = d_inner * (dt_rank + d_state * 2) + 0
out_proj = d_inner * model_config.d_model + 0
dt_proj = dt_rank * d_inner + d_inner
A_log = d_inner * d_state
D = d_inner
norm = model_config.d_model
norm_f = model_config.d_model

num_params = human_format(
    (
        tie_embedding
        + model_config.num_hidden_layers * (A_log + D + in_proj + conv1d + x_proj + dt_proj + out_proj + norm + norm_f)
    )
).replace(".", "p")

print(f"Model has {num_params} parameters")

seed = 42


optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=True,  # NOTE(fmom): because we are using PP=TP=DP=1
    learning_rate_scheduler=LRSchedulerArgs(
        learning_rate=0.0015,
        lr_warmup_steps=30,
        lr_warmup_style="linear",
        lr_decay_style="cosine",
        min_decay_lr=0.00015,
    ),
    optimizer_factory=AdamWOptimizerArgs(
        adam_eps=1e-08,
        adam_beta1=0.9,
        adam_beta2=0.95,
        torch_adam_is_fused=True,
    ),
)


parallelism = ParallelismArgs(
    dp=2,
    pp=2,
    tp=2,
    pp_engine="1f1b",
    tp_mode="ALL_REDUCE",
    tp_linear_async_communication=False,
)

tokens = TokensArgs(sequence_length=2048, train_steps=300, micro_batch_size=8, batch_accumulation_per_replica=1)

data_stages = [
    DatasetStageArgs(
        name="Stable Training Stage",
        start_training_step=1,
        data=DataArgs(
            dataset=PretrainDatasetsArgs(hf_dataset_or_datasets="roneneldan/TinyStories", text_column_name="text"),
            seed=seed,
        ),
    )
]

model = ModelArgs(
    init_method=MambaInit(initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1),
    model_config=model_config,
)

checkpoints_path = os.path.dirname(os.path.dirname(__file__)) + "/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = MambaConfig(
    general=GeneralArgs(project="test", run="mamba", seed=seed, ignore_sanity_checks=True),
    checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=100),
    parallelism=parallelism,
    model=model,
    tokenizer=TokenizerArgs("gpt2"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data_stages=data_stages,
    profiler=None,
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    config.save_as_yaml(f"{dir}/config_mamba.yaml")
