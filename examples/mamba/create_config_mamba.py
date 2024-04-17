""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import math
import os
import pprint
import uuid

import wandb
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

# optimizer = OptimizerArgs(
#     zero_stage=0,
#     weight_decay=0.01,
#     clip_grad=1.0,
#     accumulate_grad_in_fp32=True,  # NOTE(fmom): because we are using PP=TP=DP=1
#     learning_rate_scheduler=LRSchedulerArgs(
#         learning_rate=0.0015,
#         lr_warmup_steps=30,
#         lr_warmup_style="linear",
#         lr_decay_style="cosine",
#         min_decay_lr=0.00015,
#     ),
#     optimizer_factory=SGDOptimizerArgs(),
# )

parallelism = ParallelismArgs(
    dp=1,
    pp=1,
    tp=1,
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
    import argparse
    from dataclasses import fields, is_dataclass

    from nanotron.config import get_config_from_file

    def print_differences(target, updates):
        if not is_dataclass(target) or not is_dataclass(updates):
            raise ValueError("Both target and updates should be dataclass instances")

        for field in fields(target):
            update_value = getattr(updates, field.name)

            if update_value is not None:
                if is_dataclass(update_value):
                    print_differences(getattr(target, field.name), update_value)
                else:
                    target_value = getattr(target, field.name)
                    if update_value != target_value:
                        if update_value.__class__.__module__ != "builtins":
                            continue
                        print(f"{field.name}: {target_value} -> {update_value}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="Output directory for yaml", type=str)
    parser.add_argument("--wandb-username", required=True, help="Specific wandb username", type=str)
    parser.add_argument("--wandb-project", required=True, help="Specific wandb project name", type=str)
    parser.add_argument("--wandb-run", required=True, help="Specific name for this run", type=str)

    args = parser.parse_args()

    config.general.project = args.wandb_project
    config.general.run = f"{args.wandb_run}_{job_id}"

    api = wandb.Api()
    projects = api.projects(entity=args.wandb_username)
    project_exists = any(project.name == args.wandb_project for project in projects)

    if not project_exists:
        raise ValueError(
            f"Project '{args.wandb_project}' does not exist. You should create the project first at entity {config.experiment_logger.wandb_logger.wandb_entity}"
        )

    directories = []

    experiment_path = f"{args.out_dir}/{config.general.project}/{config.general.run}"
    directories.append(experiment_path)

    config.checkpoints.checkpoints_path = f"{experiment_path}/checkpoints"
    config.checkpoints.resume_checkpoint_path = f"{experiment_path}/checkpoints"
    directories.append(config.checkpoints.checkpoints_path)
    directories.append(config.checkpoints.resume_checkpoint_path)

    # if config.lighteval is not None:
    # config.lighteval.slurm_script_dir = f"{experiment_path}/lighteval/slurm_scripts"
    # config.lighteval.slurm_template = f"{experiment_path}/run_eval.slurm.jinja"
    # config.lighteval.logging.local_output_path = f"{experiment_path}/logs"

    # directories.append(config.lighteval.slurm_script_dir)
    # directories.append(config.lighteval.logging.local_output_path)

    # if config.s3_upload is not None:
    #     config.s3_upload.upload_s3_path = f"s3://huggingface-brrr-us-east-1/fmom/checkpoints/{args.wandb_run}_{job_id}"
    #     directories.append(config.s3_upload.upload_s3_path)

    # if config.profiler is not None:
    #     config.profiler.profiler_export_path = f"{experiment_path}/logs"

    directories.append(f"{experiment_path}/logs")

    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    pprint.pprint(f"Dataset name: {config.data_stages}")
    print("Parallelism")
    print("\tdp", config.parallelism.dp)
    print("\tpp", config.parallelism.pp)
    print("\ttp", config.parallelism.tp)
    if config.lighteval is not None:
        print("Parallelism LightEval")
        print("\tdp", config.lighteval.parallelism.dp)
        print("\tpp", config.lighteval.parallelism.pp)
        print("\ttp", config.lighteval.parallelism.tp)

    yaml_path = f"{experiment_path}/{config.general.run}.yaml"
    # Sanity check that we can load, save to YAML and reload the config
    config.save_as_yaml(yaml_path)
    config2 = get_config_from_file(yaml_path, config_class=MambaConfig)
    print_differences(config, config2)

    print("Save at", yaml_path)
