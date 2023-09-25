""" Example of generation with a pretrained Llama model.

USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun  --nproc_per_node=4 scripts/generate.py --pp 2 --tp 2 --model_name huggyllama/llama-7b --ckpt-path /fsx/nouamane/projects/brrr/pretrained/llama-2-7b
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from nanotron.config import ParallelismArgs
from nanotron.core import distributed as dist
from nanotron.core import logging
from nanotron.core.logging import log_rank
from nanotron.core.parallelism.parameters import sanity_check
from nanotron.core.parallelism.pipeline_parallelism.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.core.parallelism.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.core.parallelism.tensor_parallelism.enum import TensorParallelLinearMode
from nanotron.core.process_groups_initializer import get_process_groups
from nanotron.core.random import (
    set_random_seed,
)
from nanotron.core.serialize import (
    load_weights,
)
from nanotron.generation import GenerationConfig, GenerationInput, TokenizerConfig, greedy_search
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, DistributedTrainer, mark_tied_parameters

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()
    checkpoint_path = args.ckpt_path
    parallel_config = ParallelismArgs(
        dp=args.dp,
        pp=args.pp,
        tp=args.tp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        recompute_granularity=None,
        tp_linear_async_communication=False,
    )
    dtype = torch.bfloat16

    # Set random states
    set_random_seed(42)

    # Initialise all process groups
    dpg = get_process_groups(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    model_name = args.model_name
    model_config: AutoConfig = AutoConfig.from_pretrained(model_name)
    # model_config.num_hidden_layers = 1

    model_config_cls = model_config.__class__.__name__
    if model_config_cls not in CONFIG_TO_MODEL_CLASS:
        raise ValueError(
            f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"
        )

    model = DistributedTrainer.build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            dpg=dpg,
            parallel_config=parallel_config,
            random_states=None,
        ),
        model_config=model_config,
        dtype=dtype,
        dpg=dpg,
    )

    # Mark some parameters as tied
    # TODO @nouamane: this is only needed for training, can we just mark params as NanotronParameter instead?
    mark_tied_parameters(model=model, dpg=dpg, parallel_config=parallel_config)

    # Sanity check model
    sanity_check(root_module=model)

    # Load checkpoint
    log_rank(
        f"Loading checkpoint from {checkpoint_path}:",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    load_weights(model=model, dpg=dpg, root_folder=checkpoint_path)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dummy_inputs = [
        "This film was probably inspired by Godzilla",
        "If the crew behind 'Zombieland' had a",
    ]

    lm_without_head = model.transformer if model_config_cls == "FalconConfig" else model.model
    outputs = greedy_search(
        input_iter=(GenerationInput(text=text) for text in dummy_inputs),
        tokenizer=tokenizer,
        # TODO @thomasw21: From ModelWithLoss extract the model.
        model=lm_without_head,
        # TODO @thomasw21: Figure out how to pass p2p.
        p2p=lm_without_head.p2p,
        dpg=dpg,
        generation_config=GenerationConfig(max_new_tokens=40, max_micro_batch_size=8),
        tokenizer_config=TokenizerConfig(max_input_length=8),
    )
    dist.barrier()
    for output in outputs:
        input_ids = output.input_ids
        generated_ids = output.generation_ids
        if isinstance(input_ids, TensorPointer):
            assert isinstance(generated_ids, TensorPointer)
            continue
        assert isinstance(generated_ids, torch.Tensor)
        print(
            {
                "input": tokenizer.decode(input_ids, clean_up_tokenization_spaces=False),
                "generation": tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False),
            }
        )
    dist.barrier()


if __name__ == "__main__":
    main()
