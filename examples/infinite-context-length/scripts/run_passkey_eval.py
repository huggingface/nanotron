"""
Nanotron Inference Script

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 run_generate.py ---ckpt-path checkpoints/test/4
```
"""

import argparse
import os
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
from nanotron.generation.decode import (
    GenerationInput,
    TokenizerConfig,
    decode_text,
)
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
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
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.get_logger(__name__)

USE_CACHE = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=0)
    parser.add_argument("--pp", type=int, default=0)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=15, help="Maximum number of new tokens to generate")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, required=True)
    parser.add_argument("--num_shots", type=int, required=True)
    parser.add_argument("--num_digits", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    return parser.parse_args()


def generate(args, model, tokenizer, inputs, parallel_context):
    outputs = decode_text(
        input_iter=(GenerationInput(text=text) for text in inputs),
        tokenizer=tokenizer,
        # TODO @thomasw21: From ModelWithLoss extract the model.
        model=model.model,
        parallel_context=parallel_context,
        max_new_tokens=args.max_new_tokens,
        max_micro_batch_size=1,
        generation_config=GenerationArgs(sampler="greedy", use_cache=USE_CACHE),
        # generation_config=GenerationArgs(sampler="top_p", use_cache=False),
        tokenizer_config=TokenizerConfig(max_input_length=None),
        is_bench=os.environ.get("USE_BENCH", "0") == "1",
    )

    responses = []
    answer_idxs = []
    for output in outputs:
        input_ids = output.input_ids
        generated_ids = output.generation_ids

        answer_ids = generated_ids[len(input_ids) :]
        decoded_answer = tokenizer.decode(answer_ids, clean_up_tokenization_spaces=False)

        if isinstance(input_ids, TensorPointer):
            assert isinstance(generated_ids, TensorPointer)
            continue
        assert isinstance(generated_ids, torch.Tensor)

        log_rank(
            f"""
            decoded_generation: {decoded_answer}""",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        responses.append(decoded_answer)
        answer_idxs.append(answer_ids.tolist())

    dist.barrier()

    return responses, answer_idxs


def load_and_filter_dataset(eval_dataset_path, depth_percent, num_shots, num_digits, seed, num_samples):
    import random

    from datasets import load_dataset

    # Set seeds for reproducibility
    random.seed(seed)

    # Load the dataset
    dataset = load_dataset(eval_dataset_path, split="train")

    # Filter the dataset
    filtered_dataset = dataset.filter(lambda x: x["depth_percent"] == depth_percent and x["num_shots"] == num_shots)
    if num_digits > 0:
        filtered_dataset = filtered_dataset.filter(lambda x: x["num_digits"] == num_digits)

    # filtered_dataset = dataset.filter(
    #     lambda x: x["depth_percent"] == depth_percent and
    #             x["num_shots"] == num_shots and
    #             x["num_digits"] == num_digits
    # )

    # Shuffle the dataset deterministically
    shuffled_dataset = filtered_dataset.shuffle(seed=seed)

    # Select only the first 10 samples
    final_dataset = shuffled_dataset.select(range(min(num_samples, len(shuffled_dataset))))

    return final_dataset


def main():
    args = get_args()
    # depth_percent = args.depth_percent
    save_path = args.save_path
    eval_dataset_path = args.eval_dataset_path
    num_shots = args.num_shots
    num_digits = args.num_digits
    seed = args.seed
    num_samples = args.num_samples

    assert args.ckpt_path.exists(), f"Checkpoint path {args.ckpt_path} does not exist"

    config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
    from nanotron import constants

    constants.CONFIG = config
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    parallel_config = ParallelismArgs(
        dp=args.dp or config.parallelism.dp,
        pp=args.pp or config.parallelism.pp,
        tp=args.tp or config.parallelism.tp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        # tp_mode=TensorParallelLinearMode.REDUCE_SCATTER,
        # tp_linear_async_communication=True,
        # # NOTE: the one from main branch
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

    # from nanotron.debug.monitor import monitor_nanotron_model
    # monitor_nanotron_model(model=model, parallel_context=parallel_context)

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

        for depth_percent in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
            log_rank(
                f"depth_percent: {depth_percent}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # from datasets import load_dataset

            # # dataset = load_dataset("nanotron/llama3-16k-passkey-retrieval-eval", split="train")
            # # df = load_dataset("nanotron/llama3-16k-passkey-retrieval-eval", split="train")

            # dataset = load_dataset(eval_dataset_path, split="train")
            # df = load_dataset(eval_dataset_path, split="train")

            # dataset = dataset.filter(lambda x: x["depth_percent"] == depth_percent and x["num_shots"] == num_shots and x["num_digits"] == num_digits)
            # df = df.filter(lambda x: x["depth_percent"] == depth_percent and x["num_shots"] == num_shots and x["num_digits"] == num_digits)

            dataset = load_and_filter_dataset(
                eval_dataset_path, depth_percent, num_shots, num_digits, seed=seed, num_samples=num_samples
            )
            df = load_and_filter_dataset(
                eval_dataset_path, depth_percent, num_shots, num_digits, seed=seed, num_samples=num_samples
            )

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            df.set_format("pandas")
            df = df[:]

            responses = []
            answer_idxs = []
            from tqdm import tqdm

            for batch in tqdm(dataloader):
                log_rank(
                    """--------------------------------------------------""",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
                log_rank(
                    f"target answer: {batch['answer']}",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )

                texts = batch["prompt"]
                response, answer_ids = generate(args, model, tokenizer, texts, parallel_context)

                log_rank(
                    f"response: {response}",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )

                responses.append(response)
                answer_idxs.append(answer_ids)

            # NOTE: now flatten the responses
            responses = [x for sublist in responses for x in sublist]
            answer_idxs = [x for sublist in answer_idxs for x in sublist]

            df["generation_text"] = responses
            df["generation_ids"] = answer_idxs

            df.to_pickle(
                f"{save_path}/passkey_eval_results_for_{depth_percent}_depth_and_num_shots_{num_shots}_and_num_samples_{num_samples}_and_num_digits_{num_digits}_and_seed_{seed}.pkl"
            )


if __name__ == "__main__":
    main()
