""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os

import torch
from nanotron.config import (
    AdamWOptimizerArgs,
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
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.config.config import PretrainDatasetsArgs

SEED = 42

model_config = LlamaConfig(
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="gelu",
    hidden_size=512,
    initializer_range=0.02,
    intermediate_size=2048,
    is_llama_config=True,
    max_position_embeddings=128,
    num_attention_heads=16,
    num_hidden_layers=4,
    num_key_value_heads=16,
    pad_token_id=None,
    pretraining_tp=1,
    rms_norm_eps=1.0e-05,
    rope_scaling=None,
    tie_word_embeddings=False,
    use_cache=True,
    vocab_size=49152,
)

learning_rate = LRSchedulerArgs(
    learning_rate=0.0006,
    lr_decay_starting_step=None,
    lr_decay_steps=None,
    lr_decay_style="cosine",
    lr_warmup_steps=5000,
    lr_warmup_style="linear",
    min_decay_lr=0.00006,
)

optimizer = OptimizerArgs(
    accumulate_grad_in_fp32=False,
    learning_rate_scheduler=learning_rate,
    optimizer_factory=AdamWOptimizerArgs(
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-08,
        torch_adam_is_fused=True,
    ),
    weight_decay=0.1,
    zero_stage=0,
    # update_clipping=True,
    clip_grad=None,
)

parallelism = ParallelismArgs(
    dp=1,
    expert_parallel_size=1,
    pp=1,
    tp=1,
    pp_engine="1f1b",
    tp_mode="ALL_REDUCE",
    tp_linear_async_communication=False,
)

tokens = TokensArgs(
    batch_accumulation_per_replica=1,
    limit_test_batches=0,
    limit_val_batches=0,
    micro_batch_size=256,
    sequence_length=128,
    train_steps=24376,
    val_check_interval=-1,
)

data = DataArgs(
    seed=SEED,
    num_loading_workers=1,
    dataset=PretrainDatasetsArgs(
        dataset_overwrite_cache=False,
        dataset_processing_num_proc_per_process=1,
        hf_dataset_config_name=None,
        hf_dataset_or_datasets="roneneldan/TinyStories",
        hf_dataset_splits="train",
        text_column_name="text",
    ),
)

config = Config(
    checkpoints=CheckpointsArgs(
        checkpoint_interval=50000,
        checkpoints_path="checkpoints",
        checkpoints_path_is_shared_file_system=False,
        save_initial_state=False,
    ),
    data_stages=[
        DatasetStageArgs(name="Stable Training Stage", start_training_step=1, data=data),
    ],
    general=GeneralArgs(
        benchmark_csv_path=None,
        consumed_train_samples=None,
        ignore_sanity_checks=True,
        project="fp8_for_nanotron",
        run="exp51_4_layers_with_layerscale_and_actsmooth_and_clipped_softmax_and_update_clipping_and_only_quant_1mlp_down_proj",
        seed=SEED,
        step=None,
    ),
    lighteval=None,
    logging=LoggingArgs(
        iteration_step_info_interval=1,
        log_level="info",
        log_level_replica="info",
        monitor_model_states=True,
        monitor_model_states_using_hooks=True,
    ),
    model=ModelArgs(
        ddp_bucket_cap_mb=25,
        dtype="float8",
        init_method=RandomInit(std=0.04419417382415922),
        make_vocab_size_divisible_by=1,
        model_config=model_config,
    ),
    optimizer=optimizer,
    parallelism=parallelism,
    profiler=None,
    tokenizer=TokenizerArgs(
        tokenizer_max_length=None,
        tokenizer_name_or_path="lvwerra/the-tokenizer-v1",
        tokenizer_revision=None,
    ),
    tokens=tokens,
)

# NOTE:
# 1. combination of quantize layers
#  1.0 quantize which layers
#  1.1 quantize linear layer
#  1.2 quantize attention layer
# 2. smooth_quant = [True, False]
# 3. fp8_layer
#  3.0 accum_dtype
#  3.1 input
#  3.2 weight
#  3.3 input_grad
#  3.4 weight_grad
#  3.5 output_grad
#  3.5 smooth_quant = [True, False]

# 2. update_clipping = [True, False]
# 3. clipped_softmax = [True, False]
# 4. layer_scale = [True, False]
# 5. qk_norm = [True, False]
# 6. qk_norm_before_pos = [True, False]

from nanotron.config.fp8_config import FP8LayerArgs
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.recipe import FP8Accumulate, FP8SplitAccumulator, FP8TensorRecipe


def gen_layer_idxs_combo():
    from itertools import chain, combinations

    def all_combinations(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    layer_idxs = [0, 1, 2, 3]
    combinations = list(all_combinations(layer_idxs))
    return combinations


arg_layer_idx_combos = gen_layer_idxs_combo()
arg_quantize_which_parts = [
    [
        "model.decoder.{}.attn.qkv_proj",
        "model.decoder.{}.attn.o_proj",
    ],
    [
        "model.decoder.{}.mlp.gate_up_proj",
        "model.decoder.{}.mlp.down_proj",
    ],
    [
        "model.decoder.{}.attn.qkv_proj",
        "model.decoder.{}.attn.o_proj",
        "model.decoder.{}.mlp.gate_up_proj",
        "model.decoder.{}.mlp.down_proj",
    ],
]
arg_update_clippings = [True, False]
arg_clipped_softmax = [True, False]
arg_layer_scale = [True, False]
arg_qk_norm = [True, False]
arg_qk_norm_before_pos = [True, False]
# NOTE: input_grad_recipe, weight_grad_recipe, output_grad_recipe
# args_fp8_tensor_types = [
#     [
#         FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#         FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#         FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#     ],
#     [
#         FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
#         FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
#         FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
#     ],
#     [
#         FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
#         FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#         FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
#     ],
#     [
#         FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=16),
#         FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
#         FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=16),
#     ],
# ]
args_fp8_tensor_interval = [1, 16]
args_fp8_linear_accum_dtypes = [torch.bfloat16]
args_fp8_linear_smooth_quant = [True, False]

# args_backward_tensors = []
# # NOTE: input_grad, weight_grad, output_grad
# for backward_dtype in [DTypes.FP8E5M2, DTypes.FP8E4M3]:
#     xs = []
#     for interval in args_fp8_tensor_interval:
#         for _ in range(3):
#             xs.append(FP8TensorRecipe(dtype=backward_dtype, margin=0, interval=interval))
#     args_backward_tensors.append(xs)

# # NOTE: input, weight
# forward_tensors = []
# for _ in range(2):
#     for interval in args_fp8_tensor_interval:
#         forward_tensors.append(FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=interval))


# def gen_fp8_tensor_recipe():
#     from dataclasses import dataclass
#     from itertools import product
#     dtypes = [DTypes.FP8E5M2, DTypes.FP8E4M3]
#     intervals = [1, 16]

#     # Create all possible combinations for the last 3 tensors
#     variable_tensor_configs = list(product(dtypes, intervals))

#     # Generate all possible combinations
#     all_configs = list(product(intervals, intervals, variable_tensor_configs, variable_tensor_configs, variable_tensor_configs))

#     # Create the final set of configs
#     config_set = []
#     for config in all_configs:
#         config_tuple = (
#             FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=config[0]),  # input
#             FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=config[1]),  # weight
#             FP8TensorRecipe(dtype=config[2][0], margin=0, interval=config[2][1]),  # input gradient
#             FP8TensorRecipe(dtype=config[3][0], margin=0, interval=config[3][1]),  # weight gradient
#             FP8TensorRecipe(dtype=config[4][0], margin=0, interval=config[4][1])   # output gradient
#         )
#         config_set.append(config_tuple)

#     return config_set

# args_fp8_tensors = gen_fp8_tensor_recipe()
args_fp8_tensors = [
    [
        FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
        FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    ]
]

i = 0
for accum_dtype in args_fp8_linear_accum_dtypes:
    for smooth_quant in args_fp8_linear_smooth_quant:
        for input, weight, input_grad, weight_grad, output_grad in args_fp8_tensors:
            for layer_idxs in arg_layer_idx_combos:
                for parts in arg_quantize_which_parts:
                    fp8_linears = []
                    for layer_idx in layer_idxs:
                        for part in parts:

                            config_fp8_layer = FP8LayerArgs(
                                accum_dtype=accum_dtype,
                                input=input,
                                weight=weight,
                                bias=accum_dtype,
                                input_grad=input_grad,
                                weight_grad=weight_grad,
                                output_grad=output_grad,
                                split_accumulator=FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
                                accumulate=FP8Accumulate(output=True, input_grad=True, weight_grad=True),
                                smooth_quant=smooth_quant,
                                module_name=part.format(layer_idx),
                            )
                            fp8_linears.append(config_fp8_layer)
                    i += 1

assert 1 == 1


if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    filename = os.path.basename(__file__).replace(".py", ".yaml")
    assert 1 == 1

    # config.save_as_yaml(f"{dir}/{filename}")
    # print(f"Config saved as {dir}/{filename}")
    # You can now train a model with this config using `/run_train.py`
