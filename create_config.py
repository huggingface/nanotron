import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
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
    PretrainDatasetsArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.models.llama import LlamaConfig

if __name__ == "__main__":
    ###########################################
    ## ADAPT TO YOUR ENVIRONMENT (toy example of smollm-135M on 1 GPU)

    HF_USER_OR_ORG = None
    TRAIN_STEPS = 100
    CHECKPOINT_INTERVAL = 200
    SAVE_NAME = "smollm-135M-1gpu-toy"

    ###########################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", help="path to save the configuration file", type=str, default="yaml")
    parser.add_argument("--seed", help="seed", type=int, default=8)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    general = GeneralArgs(
        project="smollm",
        run="toy-smollm",
        seed=args.seed,
        temp_dir="temp",
    )

    model_config = LlamaConfig(
        bos_token_id=0,
        eos_token_id=0,
        hidden_act="silu",
        hidden_size=576,
        initializer_range=0.02,
        intermediate_size=1536,
        max_position_embeddings=2048,
        num_attention_heads=9,
        num_hidden_layers=30,
        num_key_value_heads=3,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=49152,
    )

    # Uncomment to evaluate the model on a set of tasks with lighteval during the training.
    # lighteval = LightEvalConfig(
    #     tasks=LightEvalTasksArgs(
    #         tasks="early-signal",  # "generatives", "all"
    #         custom_tasks="nanotron.lighteval.evaluation_tasks",
    #         max_samples=1000,
    #         dataset_loading_processes=8,
    #     ),
    #     parallelism=ParallelismArgs(
    #         dp=8,
    #         pp=1,
    #         tp=1,
    #         pp_engine="1f1b",
    #         tp_mode="ALL_REDUCE",
    #         # recompute_granularity="selective",
    #         tp_linear_async_communication=False,
    #     ),
    #     batch_size=16,
    #     logging=LightEvalLoggingArgs(
    #         output_dir=None,
    #         push_to_hub=True,
    #         push_to_tensorboard=True,
    #         public_run=False,
    #         results_org=HF_USER_OR_ORG,
    #         tensorboard_metric_prefix="eval",
    #     ),
    # )

    lighteval = None

    checkpoints = CheckpointsArgs(
        # checkpoints_path="checkpoints",
        checkpoints_path_is_shared_file_system=False,
        # resume_checkpoint_path="local_path/to/checkpoint" or s3_path,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        save_initial_state=False,
    )

    parallelism = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )

    tokens = TokensArgs(
        batch_accumulation_per_replica=1,
        micro_batch_size=8,
        sequence_length=2048,
        train_steps=TRAIN_STEPS,
        val_check_interval=-1,
    )

    model = ModelArgs(
        model_config=model_config,
        init_method=RandomInit(
            std=1 / math.sqrt(model_config.hidden_size),
        ),
        dtype=torch.bfloat16,
    )

    logging = LoggingArgs(
        # 'debug', 'info', 'warning', 'error', 'critical' and 'passive'
        log_level="info",
        log_level_replica="info",
        iteration_step_info_interval=1,
    )

    learning_rate_scheduler = LRSchedulerArgs(
        learning_rate=3e-3,
        lr_warmup_steps=10,
        lr_warmup_style="linear",
        lr_decay_style="linear",
        lr_decay_steps=20,
        lr_decay_starting_step=80,
        min_decay_lr=0,
    )

    optimizer = OptimizerArgs(
        zero_stage=0,
        weight_decay=0.01,
        clip_grad=1.0,
        accumulate_grad_in_fp32=True,
        learning_rate_scheduler=learning_rate_scheduler,
        optimizer_factory=AdamWOptimizerArgs(
            adam_eps=1e-08,
            adam_beta1=0.9,
            adam_beta2=0.95,
            torch_adam_is_fused=True,
        ),
    )

    tokenizer = TokenizerArgs(
        tokenizer_name_or_path="HuggingFaceTB/cosmo2-tokenizer",
    )

    # Uncomment if you want to upload the checkpoints to s3 or load a ckpt from s3
    # s3_upload = S3UploadArgs(
    #     upload_s3_path=f"S3_PATH",
    #     remove_after_upload=True,
    #     s5cmd_numworkers=16,
    #     s5cmd_concurrency=5,
    #     s5cmd_path="PATH_TO_S5CMD",
    # )

    data_stages = [
        DatasetStageArgs(
            data=DataArgs(
                # 1. Un-tokenized dataset from HuggingFace
                dataset=PretrainDatasetsArgs(
                    hf_dataset_or_datasets="HuggingFaceTB/smollm-corpus",  # feel free to replace it by a smaller one if you don't have enough memory
                    hf_dataset_splits="train",
                    hf_dataset_config_name="cosmopedia-v2",
                    text_column_name="text",
                ),
                # 2. Pre-tokenized local dataset with Nanoset
                # dataset=NanosetDatasetsArgs(
                #     dataset_folder="datasets/cosmopedia-v2",
                # ),
                # num_loading_workers=0,
                # seed=general.seed,
            ),
            name="training stage",
            start_training_step=1,
        ),
        # You can add a decay stage here if you want to change the data mixture
        # Example (weight are arbitrary here):
        # DatasetStageArgs(
        #     data=DataArgs(
        #         dataset=NanosetDatasetsArgs(
        #             dataset_folder={
        #                 "datasets/fineweb-edu-dedup": 50,
        #                 "datasets/cosmopedia-v2": 30,
        #                 "datasets/python-edu": 10,
        #                 "datasets/open-web-math": 10,
        #             }
        #         ),
        #         num_loading_workers=0,
        #         seed=general.seed,
        #     ),
        #     name="decay stage",
        #     start_training_step=optimizer.learning_rate_scheduler.lr_decay_starting_step,
        # ),
    ]

    config = Config(
        general=general,
        checkpoints=checkpoints,
        parallelism=parallelism,
        model=model,
        tokenizer=tokenizer,
        logging=logging,
        tokens=tokens,
        optimizer=optimizer,
        data_stages=data_stages,
        lighteval=lighteval,
    )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    config_path_yaml = save_path / f"{SAVE_NAME}.yaml"
    config.save_as_yaml(config_path_yaml)

    print(f"💾 Configuration saved in: {str(save_path)}")
    print("To launch this configuration, run:")
    print(f"python launcher.py --config-path configs/{str(config_path_yaml)}")
