from typing import Optional

import nanotron
import torch
from nanotron.config import AdamOptimizerArgs, LRSchedulerArgs, OptimizerArgs, ParallelismArgs
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel import ParallelContext
from nanotron.random import RandomState
from nanotron.trainer import mark_tied_parameters

DEFAULT_LLAMA_CONFIG = NanotronLlamaConfig(
    **{
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        # NOTE: make sure sharded hidden size is divisible by 16
        # for FP8 gemm to run
        "hidden_size": 32,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "is_llama_config": True,
        "max_position_embeddings": 128,
        "num_attention_heads": 8,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "pad_token_id": None,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 4096,
    }
)


DEFAULT_OPTIMIZER_CONFIG = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.1,
    clip_grad=1.0,
    accumulate_grad_in_fp32=False,
    learning_rate_scheduler=LRSchedulerArgs(
        # NOTE(xrsrke): use a high learning rate to make changes in the weights more visible
        learning_rate=0.001,
        lr_warmup_steps=100,
        lr_warmup_style="linear",
        lr_decay_style="cosine",
        min_decay_lr=1e-5,
    ),
    optimizer_factory=AdamOptimizerArgs(
        name="custom_adam",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-08,
        torch_adam_is_fused=False,
    ),
)


def make_parallel_config(
    dp: int = 1,
    pp: int = 1,
    tp: int = 1,
) -> ParallelismArgs:
    # TODO(xrsrke): allow test other tp_mode and tp_linear_async_communication
    parallel_config = ParallelismArgs(
        dp=dp,
        pp=pp,
        tp=tp,
        pp_engine=nanotron.config.AllForwardAllBackwardPipelineEngine(),
        tp_mode=nanotron.config.TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    return parallel_config


def create_nanotron_model(
    parallel_context: ParallelContext, dtype: torch.dtype = torch.bfloat16, random_states: Optional[RandomState] = None
) -> LlamaForTraining:
    parallel_config = make_parallel_config(
        tp=parallel_context.tensor_parallel_size,
        dp=parallel_context.data_parallel_size,
        pp=parallel_context.pipeline_parallel_size,
    )
    nanotron_model = nanotron.models.build_model(
        model_builder=lambda: LlamaForTraining(
            config=DEFAULT_LLAMA_CONFIG,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    return nanotron_model


def available_gpus():
    if not torch.cuda.is_available():
        return 0

    device_properties = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]

    # We filter out
    blacklisted_gpu_names = {"NVIDIA DGX Display"}
    device_properties = [property_ for property_ in device_properties if property_.name not in blacklisted_gpu_names]

    # TODO @thomasw21: Can we do this cross node
    return len(device_properties)
