import nanotron
import torch
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.config import ParallelismArgs
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel import ParallelContext
from nanotron.trainer import mark_tied_parameters

CONFIG = NanotronLlamaConfig(
    **{
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 64,
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


def create_nanotron_model(parallel_context: ParallelContext, dtype: torch.dtype = torch.bfloat16) -> LlamaForTraining:
    parallel_config = make_parallel_config(
        tp=parallel_context.tensor_parallel_size,
        dp=parallel_context.data_parallel_size,
        pp=parallel_context.pipeline_parallel_size,
    )
    nanotron_model = nanotron.models.build_model(
        model_builder=lambda: LlamaForTraining(
            config=CONFIG,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    return nanotron_model
