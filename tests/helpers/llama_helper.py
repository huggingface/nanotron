import torch
from nanotron.config import (
    AdamWOptimizerArgs,
    AllForwardAllBackwardPipelineEngine,
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
    TensorParallelLinearMode,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.config.config import PretrainDatasetsArgs
from nanotron.models import build_model
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel.context import ParallelContext
from nanotron.trainer import mark_tied_parameters

TINY_LLAMA_CONFIG = LlamaConfig(
    **{
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 128 * 4,
        "is_llama_config": True,
        "max_position_embeddings": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "pad_token_id": None,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 4096,
    }
)


def get_llama_training_config(model_config: ModelArgs):
    return Config(
        model=model_config,
        general=GeneralArgs(project="unittest", run="sanity_llama", seed=42),
        checkpoints=CheckpointsArgs(
            checkpoints_path="./checkpoints",
            checkpoint_interval=10,
        ),
        parallelism=ParallelismArgs(
            dp=1,
            pp=1,
            tp=2,
            expert_parallel_size=2,
            pp_engine="1f1b",
            tp_mode="ALL_REDUCE",
            tp_linear_async_communication=False,
        ),
        tokenizer=TokenizerArgs("gpt2"),
        optimizer=OptimizerArgs(
            optimizer_factory=AdamWOptimizerArgs(
                adam_eps=1e-08,
                adam_beta1=0.9,
                adam_beta2=0.95,
                torch_adam_is_fused=True,
            ),
            zero_stage=0,
            weight_decay=0.01,
            clip_grad=1.0,
            accumulate_grad_in_fp32=False,
            learning_rate_scheduler=LRSchedulerArgs(
                learning_rate=3e-4,
                lr_warmup_steps=100,
                lr_warmup_style="linear",
                lr_decay_style="cosine",
                min_decay_lr=1e-5,
            ),
        ),
        logging=LoggingArgs(),
        tokens=TokensArgs(sequence_length=16, train_steps=10, micro_batch_size=16, batch_accumulation_per_replica=1),
        data_stages=[
            DatasetStageArgs(
                name="train",
                start_training_step=1,
                data=DataArgs(
                    seed=42,
                    num_loading_workers=1,
                    dataset=PretrainDatasetsArgs(
                        hf_dataset_or_datasets="HuggingFaceH4/testing_alpaca_small",
                        hf_dataset_splits="train",
                        text_column_name="completion",
                        dataset_processing_num_proc_per_process=12,
                    ),
                ),
            )
        ],
    )


def create_llama_from_config(
    model_config: LlamaConfig, device: torch.device, parallel_context: ParallelContext
) -> LlamaForTraining:

    """
    Creates and returns a nanotron model.
    If `model_config` is None, then `checkpoint_path` must be set, in which case
    the configuration will be loaded from such path.
    If `checkpoint_path` is None, then `model_config` must be set, in which case
    the model created will have random weights.
    """

    parallel_config = ParallelismArgs(
        dp=parallel_context.data_parallel_size,
        pp=parallel_context.pipeline_parallel_size,
        tp=parallel_context.tensor_parallel_size,
        pp_engine=AllForwardAllBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    model = build_model(
        model_builder=lambda: LlamaForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=torch.bfloat16,
        device=device,
    )
    mark_tied_parameters(model=model, parallel_context=parallel_context)
    return model
