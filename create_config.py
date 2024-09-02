import os
from pathlib import Path
import subprocess
from datetime import datetime
import math
import torch

import argparse

from nanotron.models.llama import LlamaConfig

from nanotron.config import (
    Config,
    DataArgs,
    NanosetDatasetsArgs,
    S3UploadArgs,
    CheckpointsArgs,
    GeneralArgs,
    LightEvalConfig,
    LightEvalLoggingArgs,
    LightEvalTasksArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    AdamWOptimizerArgs,
    ParallelismArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
    DatasetStageArgs,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", help="path to save the configuration file", type=str, required=True)
    parser.add_argument("--seed", help="seed", type=int, default=8)
    parser.add_argument("--priority", "--qos", "-p", help="qos to use", type=str, default="high")
    parser.add_argument("--override", nargs="+", metavar="KEY=VALUE",
                        help="Override config values. Use dot notation for nested keys.")
    parser.add_argument("--launch", action="store_true", help="Launch the configuration immediately")
    parser.add_argument("--run", help="name of the run", type=str)
    parser.add_argument("--logs-path", help="path to the logs folder", type=str)
    parser.add_argument("--slurm", help="use slurm", action="store_true")
    parser.add_argument("--nodes", help="specify the number of nodes", type=int)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    general = GeneralArgs(
        project="smollm",
        seed=args.seed,
        temp_dir="/scratch",
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

    lighteval = LightEvalConfig(
        tasks=LightEvalTasksArgs(
            tasks="early-signal",  # "generatives", "all"
            custom_tasks="nanotron.lighteval.evaluation_tasks",
            max_samples=1000,  # Cap very large evals or for debugging
            dataset_loading_processes=8,
        ),
        parallelism=ParallelismArgs(
            dp=8,
            pp=1,
            tp=1,
            pp_engine="1f1b",
            tp_mode="ALL_REDUCE",
            # recompute_granularity="selective",
            tp_linear_async_communication=False,
        ),
        batch_size=16,
        logging=LightEvalLoggingArgs(
            local_output_path=f"/fsx/elie_bakouch/refactor-lighteval-logs/{general.project}-{general.run}",
            #local_output_path=PATH_TO_LOCAL_LOG,
            private=True,
            push_details_to_hub=True,
            push_results_to_hub=True,
            push_results_to_tensorboard=True,
            hf_user_or_org="eliebak",
            #hf_user_or_org="USER_OR_ORG",
            hub_repo_results="lighteval-results",
            hub_repo_details="lighteval-details",
            hub_repo_tensorboard="smollm-evals-visualization",
            tensorboard_metric_prefix="eval",
        ),
        slurm_template="/fsx/elie_bakouch/nanotron/slurm/run_eval.slurm.jinja",
    )


    checkpoints = CheckpointsArgs(
        checkpoints_path=f"/fsx/elie_bakouch/refactor-checkpoints/{general.project}-{general.run}",
        #checkpoints_path="CHECKPOINTS_PATH",
        checkpoints_path_is_shared_file_system=False,
        resume_checkpoint_path=None,
        checkpoint_interval=20,
        save_initial_state=False,
    )

    parallelism = ParallelismArgs(
        dp=8,
        pp=1,
        tp=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )

    tokens = TokensArgs(
        batch_accumulation_per_replica=8,
        micro_batch_size=16,
        sequence_length=2048,
        train_steps=100,
        val_check_interval=-1,
    )

    model = ModelArgs(
        model_config=model_config,
        init_method=RandomInit(
            std=1/math.sqrt(model_config.hidden_size),
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
        learning_rate=1e-4,
        lr_warmup_steps=10,
        lr_warmup_style="linear",
        lr_decay_style="linear",            
        lr_decay_steps = 20,
        lr_decay_starting_step= 80,
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

    # s3_upload = S3UploadArgs(
    #     upload_s3_path=f"s3://elie-exp/debug_nanotron/{general.project}-{general.run}-{timestamp}",
    #     remove_after_upload=True,
    #     s5cmd_numworkers=16,
    #     s5cmd_concurrency=5,
    #     s5cmd_path="/fsx/elie_bakouch/miniconda3/envs/smollm/bin/s5cmd",
    # )
    data_stages=[
        DatasetStageArgs(
            data=DataArgs(
                dataset=NanosetDatasetsArgs(
                    dataset_folder="/fsx/elie_bakouch/nanotron/datasets/cosmopedia-v2",
                ),
                num_loading_workers=0,
                seed=general.seed,
            ),
            name="training stage",
            start_training_step=1,
        ),
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

    save_path= Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    config_path_yaml = save_path / f"{args.run}-{timestamp}.yaml"
    config.save_as_yaml(config_path_yaml)

    print(f"💾 Configuration saved in: {str(save_path)}")

    if args.launch:
        # Change the launcher_path
        # Sanity check for logs_path and run
        if not args.logs_path:
            raise ValueError("--logs_path must be defined. Please provide a path for the logs.")
        if not args.run:
            raise ValueError("--run must be defined. Please provide a name for the run.")

        launcher_path = Path("launcher.py")
        if not launcher_path.exists():
            raise FileNotFoundError(f"Launcher not found at {launcher_path}. Please ensure the file exists or change the launcher path in the create_config.py file.")
        launch_command = [
            "python", str(launcher_path),
            "--config-path", str(config_path_yaml),
        ]
        launch_command.extend([
            "--logs-path", args.logs_path,
            "--run", args.run
        ])
        if args.slurm:
            launch_command.append("--slurm")
    
        if args.nodes:
            launch_command.extend(["--nodes", str(args.nodes)])
        
        
        print(f"🧪 Launching configuration with command: {' '.join(launch_command)}")
        subprocess.run(launch_command, check=True)
    else:
        print("To launch this configuration, run:")
        print(f"python {os.path.join(dir, 'launcher.py')} {config_path_yaml}")