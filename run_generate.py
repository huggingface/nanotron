"""
Nanotron Inference Script

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 run_generate.py ---ckpt-path checkpoints/test/4
```
"""
import argparse
from pathlib import Path

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    GenerationArgs,
    LoggingArgs,
    ParallelismArgs,
    get_config_from_file,
)
from nanotron.distributed import get_global_rank
from nanotron.generation.decode import (
    GenerationInputs,
    GenerationStates,
    run_one_inference_step,
)
from nanotron.generation.generate_store import Store
from nanotron.generation.sampler import BasicSampler, GreedySampler, SamplerType, TopKSampler, TopPSampler
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
    set_random_seed,
)
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--use-cache", action="store_true", help="Use cache for generation")
    return parser.parse_args()


def main():
    args = get_args()

    assert args.ckpt_path.exists(), f"Checkpoint path {args.ckpt_path} does not exist"

    config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    parallel_config = ParallelismArgs(
        dp=args.dp,
        pp=args.pp,
        tp=args.tp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )

    # Initialise all process groups
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    # Set log levels
    logging_config = LoggingArgs(
        log_level="info",
        log_level_replica="info",
    )

    # Set log levels
    set_ranks_logging_level(parallel_context=parallel_context, logging_config=logging_config)

    log_rank(f"model_config: {model_config}", logger=logger, level=logging.INFO, rank=0)
    log_rank(f"tokenizer_path: {tokenizer_path}", logger=logger, level=logging.INFO, rank=0)

    dtype = torch.bfloat16

    # Set random states
    set_random_seed(42)

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

    model = build_model(
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
    if AutoTokenizer is not None:
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
            "The future of AI is",
            # "Passage: Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:",
            # "def fib(n)",
            # 'Here is an extract from a webpage: "Have you ever experienced heel pain after a heavy physical activity, or even right after a long period of standing? If you regard this as something usual and normal, then think again. Miscalled as heel pain, plantar fasciitis causes these frequent mild pains experienced in the soles of the feet. It is the inflammation and enlargement the plantar fascia tissue that is located in the heels of the feet, stretching to the base of the toes. This tissue is responsible for absorbing shock in the feet and for supporting the arches. It also plays a vital role in foot movements during walking and standing. Many factors such as excessive walking, standing, and running trigger heel pain and plantar fasciitis. A sudden increase in intensity of activities, increase in weight, and abrupt change of footwear also cause the swelling of the ligament. Non-supportive footwear lacking arch cushions and improper and worn out running or training can also lead to the problem. It is also most evident among those". Write an extensive and detailed course unit suitable for a textbook targeted at college students, related to the given extract, within the context of "Medicine". Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on: - Rigor: Ensure in-depth coverage of the concepts/sections. - Engagement: Write with an academic, professional and engaging tone that captivates interest. - Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.',
            # "Advancements in technology will lead to",
            # "Tomorrow's world is shaped by",
        ]

        log_rank(f"Using cache for generation: {args.use_cache}", logger=logger, level=logging.INFO, rank=0)

        # NOTE: This doesn't support micro-batches and batch inference
        device = torch.cuda.current_device()
        generation_config = GenerationArgs(sampler="greedy", use_cache=args.use_cache)
        logits_are_batch_first = True

        if generation_config:
            if isinstance(generation_config.sampler, str):
                sampler_type = SamplerType(generation_config.sampler.upper())
            else:
                sampler_type = generation_config.sampler
        else:
            sampler_type = SamplerType.GREEDY

        tokenized_prompts = tokenizer(
            dummy_inputs,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        tokenized_prompts["input_ids"] = tokenized_prompts["input_ids"].to(device)
        tokenized_prompts["attention_mask"] = tokenized_prompts["attention_mask"].to(dtype=torch.bool, device=device)

        store = Store()
        batch_prompts = None

        for i in range(args.max_new_tokens):

            if generation_config.use_cache:
                # Prepare the batch prompts
                batch_prompts = GenerationStates(
                    new_input_ids=tokenized_prompts["input_ids"]
                    if i == 0
                    else tokenized_prompts["input_ids"][:, -1].unsqueeze(0),
                    new_input_mask=tokenized_prompts["attention_mask"]
                    if i == 0
                    else tokenized_prompts["attention_mask"][:, -1].unsqueeze(0),
                    store=store,
                    generation_ids=tokenized_prompts["input_ids"],
                    generation_mask=tokenized_prompts["attention_mask"],
                )
            else:
                batch_prompts = GenerationInputs(
                    input_ids=tokenized_prompts["input_ids"],
                    input_masks=tokenized_prompts["attention_mask"],
                )

            logits = run_one_inference_step(
                model, batch_prompts, parallel_context, device, use_cache=generation_config.use_cache, store=store
            )

            # Sample new token
            if parallel_context.is_pipeline_last_stage:
                assert logits is not None and isinstance(logits, torch.Tensor)

                # Get sampler
                if sampler_type == SamplerType.GREEDY:
                    sampler = GreedySampler(pg=parallel_context.tp_pg)
                elif sampler_type == SamplerType.TOP_K:
                    sampler = TopKSampler(pg=parallel_context.tp_pg)
                elif sampler_type == SamplerType.TOP_P:
                    sampler = TopPSampler(pg=parallel_context.tp_pg)
                elif sampler_type == SamplerType.BASIC:
                    sampler = BasicSampler(pg=parallel_context.tp_pg)
                else:
                    raise NotImplementedError(f"Sampler type {sampler_type} is not implemented")

                if logits_are_batch_first:
                    logits = logits.transpose(0, 1)

                # Predict next token
                next_token = sampler(sharded_logits=logits[:, -1])

                # Extend the tokenized prompts to insert the new token
                tokenized_prompts["input_ids"] = torch.cat([tokenized_prompts["input_ids"], next_token], dim=-1)
                tokenized_prompts["attention_mask"] = torch.cat(
                    [
                        tokenized_prompts["attention_mask"],
                        torch.ones((tokenized_prompts["attention_mask"].shape[0], 1), dtype=torch.bool, device=device),
                    ],
                    dim=-1,
                )
            else:
                # Extend the tokenized prompts to receive the new token
                tokenized_prompts["input_ids"] = torch.zeros(
                    (tokenized_prompts["input_ids"].shape[0], tokenized_prompts["input_ids"].shape[1] + 1),
                    dtype=torch.int64,
                    device=device,
                )
                tokenized_prompts["attention_mask"] = torch.zeros(
                    (
                        tokenized_prompts["attention_mask"].shape[0],
                        tokenized_prompts["attention_mask"].shape[1] + 1,
                    ),
                    dtype=torch.bool,
                    device=device,
                )

            # Broadcast the new token to all the pipeline stages
            dist.broadcast(
                tokenized_prompts["input_ids"],
                src=get_global_rank(
                    group=parallel_context.pp_pg, group_rank=parallel_context.pipeline_parallel_last_rank
                ),
                group=parallel_context.pp_pg,
            )
            dist.broadcast(
                tokenized_prompts["attention_mask"],
                src=get_global_rank(
                    group=parallel_context.pp_pg, group_rank=parallel_context.pipeline_parallel_last_rank
                ),
                group=parallel_context.pp_pg,
            )

        # Decode the generated text
        if dist.get_rank() == 0:
            for i, prompt in enumerate(dummy_inputs):
                if generation_config.use_cache:
                    tokenized_outputs = torch.cat(
                        [tokens.view(1, -1) for tokens in batch_prompts.generation_ids], dim=1
                    )
                    outputs = tokenizer.decode(tokenized_outputs[0], clean_up_tokenization_spaces=False)
                else:
                    tokenized_outputs = tokenized_prompts["input_ids"][
                        i, tokenized_prompts["input_ids"].shape[1] - args.max_new_tokens :
                    ]
                    outputs = tokenizer.decode(tokenized_outputs, clean_up_tokenization_spaces=False)

                log_rank(f"Input: {prompt}", logger=logger, level=logging.INFO, rank=0)
                log_rank(f"Output: {outputs}", logger=logger, level=logging.INFO, rank=0)

    dist.barrier()


if __name__ == "__main__":
    main()
