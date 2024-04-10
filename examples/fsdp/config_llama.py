""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os
import pprint
import uuid

import wandb
from nanotron.config import (
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

new_job_id = uuid.uuid4()
job_id = str(new_job_id)[:8]
seed = 42

general = GeneralArgs(
    project=None, run=None, seed=seed  # NOTE(fmom): defined with argparse  # NOTE(fmom): defined with argparse
)

checkpoints = CheckpointsArgs(
    checkpoints_path=None,  # NOTE(fmom): defined with argparse
    resume_checkpoint_path=None,  # NOTE(fmom): defined with argparse
    checkpoint_interval=10,
)

parallelism = ParallelismArgs(
    dp=1,
    pp=1,
    tp=1,
    pp_engine="afab",
    tp_mode="ALL_REDUCE",
    tp_linear_async_communication=False,
)

model = ModelArgs(
    init_method=RandomInit(std=0.025),
    model_config=LlamaConfig(
        # Config for a tiny model model with 1.62M parameters
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=16,
        initializer_range=0.02,
        intermediate_size=64,
        max_position_embeddings=50277,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=50277,
    ),
)

tokenizer = TokenizerArgs("gpt2")

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
        learning_rate=3e-4, lr_warmup_steps=2, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
    ),
)

logging = LoggingArgs(
    # 'debug', 'info', 'warning', 'error', 'critical' and 'passive'
    log_level="info",
    log_level_replica="info",
    iteration_step_info_interval=1,
)

tokens = TokensArgs(sequence_length=32, train_steps=10, micro_batch_size=2, batch_accumulation_per_replica=1)

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

profiler = None
lighteval = None

config = Config(
    general=general,
    checkpoints=checkpoints,
    parallelism=parallelism,
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    logging=logging,
    tokens=tokens,
    data_stages=data_stages,
    profiler=profiler,
    lighteval=lighteval,
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

    if config.lighteval is not None:
        config.lighteval.slurm_script_dir = f"{experiment_path}/lighteval/slurm_scripts"
        config.lighteval.slurm_template = f"{experiment_path}/run_eval.slurm.jinja"
        config.lighteval.logging.local_output_path = f"{experiment_path}/logs"

        directories.append(config.lighteval.slurm_script_dir)
        directories.append(config.lighteval.logging.local_output_path)
        directories.append(config.lighteval.s3_tmp_dir)

    # if config.s3_upload is not None:
    #     config.s3_upload.upload_s3_path = f"s3://huggingface-brrr-us-east-1/fmom/checkpoints/{args.wandb_run}_{job_id}"
    #     directories.append(config.s3_upload.upload_s3_path)

    if config.profiler is not None:
        config.profiler.profiler_export_path = f"{experiment_path}/logs"
        directories.append(config.profiler.profiler_export_path)

    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    num_params = human_format(
        model.model_config.vocab_size * model.model_config.hidden_size * 2
        + model.model_config.num_hidden_layers
        * (
            3 * model.model_config.hidden_size * model.model_config.intermediate_size
            + 4 * model.model_config.hidden_size * model.model_config.hidden_size
        )
    ).replace(".", "p")

    pprint.pprint(f"Model has {num_params} parameters")
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
    config2 = get_config_from_file(yaml_path)
    print_differences(config, config2)

    print("Save at", yaml_path)
