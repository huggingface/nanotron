""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os
from dataclasses import dataclass
from typing import Optional

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
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.config.config import PretrainDatasetsArgs
from nanotron.logging import human_format


@dataclass
class LlaMoEConfig:
    """Configuration for a LLAMA model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_llamoe_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000

    ## MoE specific
    # Number of experts per Sparse MLP layer.
    moe_num_experts: int = 1
    # the number of experts to root per-token, can be also interpreted as the `top-p` routing parameter
    num_experts_per_tok: int = 1
    moe_capacity_factor: int = 1

    def __post_init__(self):
        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        assert (
            self.num_experts_per_tok <= self.moe_num_experts
        ), f"num_experts_per_tok ({self.num_experts_per_tok}) must be <= moe_num_experts ({self.moe_num_experts})"


model_config = LlaMoEConfig(
    # Config for a 52M llama model
    num_hidden_layers=1,
    hidden_size=512,
    num_attention_heads=8,
    intermediate_size=512 * 4,
    max_position_embeddings=128,
    tie_word_embeddings=False,
    vocab_size=32000,
    moe_num_experts=4,
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

SEED = 42

learning_rate = LRSchedulerArgs(
    learning_rate=3e-4, lr_warmup_steps=100, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
)

optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=False,
    learning_rate_scheduler=learning_rate,
    optimizer_factory=AdamWOptimizerArgs(
        adam_eps=1e-08,
        adam_beta1=0.9,
        adam_beta2=0.95,
        torch_adam_is_fused=True,
    ),
)

parallelism = ParallelismArgs(
    dp=1,
    pp=1,
    tp=2,
    expert_parallel_size=2,
    pp_engine="1f1b",
    tp_mode="ALL_REDUCE",
    tp_linear_async_communication=False,
)

assert (
    model_config.moe_num_experts % parallelism.expert_parallel_size == 0
), "Number of experts must be divisible by expert_parallel_size"

tokens = TokensArgs(sequence_length=256, train_steps=1918, micro_batch_size=256, batch_accumulation_per_replica=2)

data = DataArgs(
    seed=SEED,
    num_loading_workers=1,
    # dataset=None
    dataset=PretrainDatasetsArgs(
        hf_dataset_or_datasets="roneneldan/TinyStories",
        hf_dataset_splits="train",
        text_column_name="text",
        dataset_processing_num_proc_per_process=12,
    ),
)


checkpoints_path = os.path.dirname(os.path.dirname(__file__)) + "/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = Config(
    general=GeneralArgs(project="moe", run="llamoe", seed=SEED),
    checkpoints=CheckpointsArgs(
        checkpoints_path=checkpoints_path,
        checkpoint_interval=100000,
        save_initial_state=True,
        resume_checkpoint_path=checkpoints_path,
    ),
    parallelism=parallelism,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
    tokenizer=TokenizerArgs("meta-llama/Llama-2-7b-hf"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data_stages=[
        DatasetStageArgs(name="Stable Training Stage", start_training_step=1, data=data),
        DatasetStageArgs(name="Annealing Phase", start_training_step=10, data=data),
    ],
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    filename = os.path.basename(__file__).replace(".py", ".yaml")
    config.save_as_yaml(f"{dir}/{filename}")
    print(f"Config saved as {dir}/{filename}")
    # You can now train a model with this config using `/run_train.py`
