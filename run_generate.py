""" Example of generation with a pretrained model.
- llama:
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 examples/generate.py --pp 2 --tp 1 --model_name huggyllama/llama-7b --ckpt-path pretrained/llama-7b
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 examples/generate.py --pp 2 --tp 1 --model_name codellama/CodeLlama-7b-hf --ckpt-path pretrained/CodeLlama-7b-hf
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 examples/generate.py --pp 2 --tp 1 --ckpt-path pretrained/Llama-2-7b-hf
- falcon:
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 examples/generate.py --pp 4 --tp 2 --model_name /fsx/shared-falcon-180B/falcon-180B/ --ckpt-path /fsx/shared-falcon-180B/nanotron-falcon-180B
- santacoder:
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/generate.py --pp 2 --tp 2 --model_name bigcode/gpt_bigcode-santacoder --ckpt-path pretrained/gpt_bigcode-santacoder
- starcoder:
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/generate.py --pp 2 --tp 2 --model_name bigcode/starcoder --ckpt-path pretrained/starcoder
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/generate.py --pp 2 --tp 2 --model_name bigcode/starcoder --ckpt-path /fsx/nouamane/checkpoints/nanotron/starcoder_s64k_sw4k_dp16_gbs1M/10000
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 examples/generate.py --pp 1 --tp 1 --ckpt-path /fsx/nouamane/checkpoints/nanotron/test/12
- Benchmark:
USE_BENCH=1 USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 run_benchmark2.py --pp 2 --tp 1 --dp 1 --model_name huggyllama/llama-7b --ckpt-path /admin/home/ferdinand_mom/.cache/huggingface/hub/models--HuggingFaceBR4--llama-7b-orig/snapshots/2160b3d0134a99d365851a7e95864b21e873e1c3
"""
import os
import argparse
from pathlib import Path

import torch
from nanotron import logging
from nanotron.config import GenerationArgs, ParallelismArgs, LoggingArgs, get_config_from_file
from nanotron.core import distributed as dist
from nanotron.core.parallel.parameters import sanity_check
from nanotron.core.parallel.pipeline_parallelism.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.core.parallel.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.core.parallel.tensor_parallelism.enum import TensorParallelLinearMode
from nanotron.core.process_groups import get_process_groups
from nanotron.core.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
    set_random_seed,
)
from nanotron.generate.generation import (
    GenerationInput,
    TokenizerConfig,
    greedy_search_text,
)

from nanotron.helpers import set_logger_verbosity_format
from nanotron.logging import log_rank
from nanotron.serialize import (
    load_weights,
)
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, DistributedTrainer, mark_tied_parameters
from transformers import AutoConfig, AutoTokenizer

logger = logging.get_logger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--compare-with-no-cache", action="store_true")
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
    dpg = get_process_groups(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    # Set log levels
    if dist.get_rank(dpg.world_pg) == 0:
        if logging_config.log_level is not None:
            set_logger_verbosity_format(logging_config.log_level, dpg=dpg)
    else:
        if logging_config.log_level_replica is not None:
            set_logger_verbosity_format(logging_config.log_level_replica, dpg=dpg)


    tokenizer_path = args.model_name
    # if config.yaml in checkpoint path we use it
    if (args.ckpt_path / "config.yaml").exists():
        config_path = args.ckpt_path / "config.yaml"
        # parse config
        config = get_config_from_file(config_path.as_posix())
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
  
    # model_config.num_hidden_layers = 1
    log_rank(f"model_config: {model_config}", logger=logger, level=logging.INFO, rank=0)

    model_config_cls = model_config.__class__.__name__
    if model_config_cls not in CONFIG_TO_MODEL_CLASS:
        raise ValueError(
            f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"
        )

    # Get synchronized random states
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=dpg.tp_pg)}
        )
    else:
        # We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
        random_states = RandomStates({})

    model = DistributedTrainer.build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            dpg=dpg,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=dtype,
        dpg=dpg,
    )

    # Mark some parameters as tied
    # TODO @nouamane: this is only needed for training, can we just mark params as NanotronParameter instead?
    mark_tied_parameters(model=model, dpg=dpg, parallel_config=parallel_config)

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
    load_weights(model=model, dpg=dpg, root_folder=checkpoint_path)

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
    
    outputs = greedy_search_text(
        input_iter=(GenerationInput(text=text) for text in dummy_inputs),
        tokenizer=tokenizer,
        # TODO @thomasw21: From ModelWithLoss extract the model.
        model=model.model,
        # TODO @thomasw21: Figure out how to pass p2p.
        p2p=model.model.p2p,
        dpg=dpg,
        max_new_tokens=args.max_new_tokens,
        max_micro_batch_size=2,
        generation_config=GenerationArgs(sampler="greedy", use_cache=False),
        # tokenizer_config=TokenizerConfig(max_input_length=8),
        # tokenizer_config=TokenizerConfig(max_input_length=1024), #TODO @nouamane: fix padding for starcoder
        tokenizer_config=TokenizerConfig(max_input_length=None),
        # tokenizer_config=TokenizerConfig(max_input_length=model.config.max_position_embeddings - args.max_new_tokens),
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
        
    if args.compare_with_no_cache:

        outputs = greedy_search_text(
            input_iter=(GenerationInput(text=text) for text in dummy_inputs),
            tokenizer=tokenizer,
            # TODO @thomasw21: From ModelWithLoss extract the model.
            model=model.model,
            # TODO @thomasw21: Figure out how to pass p2p.
            p2p=model.model.p2p,
            dpg=dpg,
            max_new_tokens=args.max_new_tokens,
            max_micro_batch_size=2,
            generation_config=GenerationArgs(sampler="greedy", use_cache=True),
            # tokenizer_config=TokenizerConfig(max_input_length=8),
            # tokenizer_config=TokenizerConfig(max_input_length=1024), #TODO @nouamane: fix padding for starcoder
            tokenizer_config=TokenizerConfig(max_input_length=None),
            # tokenizer_config=TokenizerConfig(max_input_length=model.config.max_position_embeddings - args.max_new_tokens),
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
