"""
Nanotron Inference Script

Usage:
```
export USE_FAST=1 # optional, for faster inference. Requires flash-attn
torchrun --nproc_per_node=8 run_generate.py --pp 2 --tp 4 --ckpt-path nanotron/checkpoints/10
```
"""

import argparse
import os
from pathlib import Path

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import GenerationArgs, LoggingArgs, ParallelismArgs, get_config_from_file
from nanotron.generation.decode import (
    GenerationInput,
    TokenizerConfig,
    decode_text,
)
from nanotron.logging import log_rank, set_logger_verbosity_format
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
    set_random_seed,
)
from nanotron.serialize import (
    load_weights,
)
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, DistributedTrainer, mark_tied_parameters
from transformers import AutoConfig, AutoTokenizer

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Used to get model config in case there is no `config.yaml` or `model_config.json` in checkpoint path",
    )
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    return parser.parse_args()


def main():
    args = get_args()
    parallel_config = ParallelismArgs(
        dp=args.dp,
        pp=args.pp,
        tp=args.tp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        recompute_granularity=None,
        tp_linear_async_communication=True,
    )

    logging_config = LoggingArgs(
        log_level="info",
        log_level_replica="info",
    )

    dtype = torch.bfloat16

    # Set random states
    set_random_seed(42)

    # Initialise all process groups
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    # Set log levels
    if dist.get_rank(parallel_context.world_pg) == 0:
        if logging_config.log_level is not None:
            set_logger_verbosity_format(logging_config.log_level, parallel_context=parallel_context)
    else:
        if logging_config.log_level_replica is not None:
            set_logger_verbosity_format(logging_config.log_level_replica, parallel_context=parallel_context)

    tokenizer_path = args.model_name
    # if config.yaml in checkpoint path we use it
    if (args.ckpt_path / "config.yaml").exists():
        config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
        model_config = config.model.model_config
        tokenizer_path = config.tokenizer.tokenizer_name_or_path
    elif (args.ckpt_path / "model_config.json").exists():
        model_config = AutoConfig.from_pretrained(args.ckpt_path / "model_config.json")
        if args.model_name is None:
            tokenizer_path = model_config._name_or_path
    else:
        assert args.model_name is not None, "model_name must be provided or config.yaml must be in checkpoint path"
        model_name = args.model_name
        model_config: AutoConfig = AutoConfig.from_pretrained(model_name)

    log_rank(f"model_config: {model_config}", logger=logger, level=logging.INFO, rank=0)
    log_rank(f"tokenizer_path: {tokenizer_path}", logger=logger, level=logging.INFO, rank=0)

    model_config_cls = model_config.__class__.__name__
    if model_config_cls not in CONFIG_TO_MODEL_CLASS:
        raise ValueError(
            f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"
        )

    # Get synchronized random states
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)}
        )
    else:
        # We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
        random_states = RandomStates({})

    model = DistributedTrainer.build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=dtype,
        parallel_context=parallel_context,
    )

    # Mark some parameters as tied
    # TODO @nouamane: this is only needed for training, can we just mark params as NanotronParameter instead?
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)

    # Sanity check model
    sanity_check(root_module=model)

    # Load checkpoint
    checkpoint_path = args.ckpt_path
    log_rank(
        f"Loading checkpoint from {checkpoint_path}:",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif getattr(model.config, "pad_token_id", None) is not None:
            tokenizer.pad_token_id = int(model.config.pad_token_id)
        elif getattr(model.config, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = int(model.config.eos_token_id)
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"  # TODO @nouamane: do we want this?
    dummy_inputs = [
        # "Passage: Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:",
        "def fib(n)",
        # "This film was probably inspired by Godzilla",
    ]

    outputs = decode_text(
        input_iter=(GenerationInput(text=text) for text in dummy_inputs),
        tokenizer=tokenizer,
        # TODO @thomasw21: From ModelWithLoss extract the model.
        model=model.model,
        # TODO @thomasw21: Figure out how to pass p2p.
        p2p=model.model.p2p,
        parallel_context=parallel_context,
        max_new_tokens=args.max_new_tokens,
        max_micro_batch_size=2,
        generation_config=GenerationArgs(sampler="greedy", use_cache=False),
        tokenizer_config=TokenizerConfig(max_input_length=None),
        is_bench=os.environ.get("USE_BENCH", "0") == "1",
    )

    dist.barrier()

    for output in outputs:
        input_ids = output.input_ids
        generated_ids = output.generation_ids
        if isinstance(input_ids, TensorPointer):
            assert isinstance(generated_ids, TensorPointer)
            continue
        assert isinstance(generated_ids, torch.Tensor)

        log_rank(
            f"input: {tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)[:1000]}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        log_rank(
            f"generation: {tokenizer.decode(generated_ids[len(input_ids) :], clean_up_tokenization_spaces=False)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        log_rank(
            "--------------------------------------------------",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )


if __name__ == "__main__":
    main()
