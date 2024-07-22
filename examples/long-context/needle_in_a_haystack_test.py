"""
CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=4    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29600    --max_restarts=0    --tee=3    needle_in_a_haystack_test.py --ckpt-path /fsx/haojun/long_context_weights/experiment_gradient_reproduce_LR=2e-5/120
"""

import argparse
import glob
import os
from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Optional, Type

import dacite
import numpy as np
import torch
import yaml
from dacite import from_dict
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    LoggingArgs,
    ParallelismArgs,
)
from nanotron.config.config import Config
from nanotron.config.utils_config import RecomputeGranularity, cast_str_to_pipeline_engine, cast_str_to_torch_dtype
from nanotron.generation.decode import (
    decode_text_simple,
)
from nanotron.generation.sampler import SamplerType
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
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
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
import pickle

logger = logging.get_logger(__name__)
DEVICE = torch.device("cuda")

# copied this function to avoid strict match
def get_config_from_dict(
    config_dict: dict, config_class: Type = Config, skip_unused_config_keys: bool = False, skip_null_keys: bool = False
):
    if skip_unused_config_keys:
        config_dict = {
            field.name: config_dict[field.name] for field in fields(config_class) if field.name in config_dict
        }
    if skip_null_keys:
        config_dict = {
            k: {kk: vv for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else v
            for k, v in config_dict.items()
            if v is not None
        }
    config_dict["data_stages"] = None
    return from_dict(
        data_class=config_class,
        data=config_dict,
        config=dacite.Config(
            cast=[Path],
            type_hooks={
                torch.dtype: cast_str_to_torch_dtype,
                PipelineEngine: cast_str_to_pipeline_engine,
                TensorParallelLinearMode: lambda x: TensorParallelLinearMode[x.upper()],
                RecomputeGranularity: lambda x: RecomputeGranularity[x.upper()],
                SamplerType: lambda x: SamplerType[x.upper()],
            },
            strict=False,
        ),
    )


def get_config_from_file(
    config_path: str,
    config_class: Type = Config,
    model_config_class: Optional[Type] = None,
    skip_unused_config_keys: bool = True,
    skip_null_keys: bool = True,
) -> Config:
    """Get a config object from a file (python or YAML)

    Args:
        config_path: path to the config file
        config_type: if the file is a python file, type of the config object to get as a
            ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
            if None, will default to Config
        skip_unused_config_keys: whether to skip unused first-nesting-level keys in the config file (for config with additional sections)
        skip_null_keys: whether to skip keys with value None at first and second nesting level
    """
    # Open the file and load the file
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    config = get_config_from_dict(
        config_dict,
        config_class=config_class,
        skip_unused_config_keys=skip_unused_config_keys,
        skip_null_keys=skip_null_keys,
    )
    if model_config_class is not None:
        if not isinstance(config.model.model_config, (dict, model_config_class)):
            raise ValueError(
                f"model_config should be a dictionary or a {model_config_class} and not {config.model.model_config}"
            )
        config.model.model_config = model_config_class(**config.model.model_config)
    return config


## Ref: https://github.com/jzhang38/EasyContext

## PREFIX + Haystack + NEEDLE_FORMAT + Haystack + QUESTION_STR
PREFIX = "This is a very long story book: <book>"
NEEDLE_FORMAT = "\nThe special magic Singapore number is: {}.\n"
CITIES = ["Singapore", "Chicago", "Sydney", "Amsterdam"]
QUESTION_STR = "</book>.\n Based on the content of the book, Question: What is the special magic Singapore number? Answer: The special magic {}"
PROMPT_TOKENS = 100  # tokens for prefix, needle, question, generation


def generate_question_str(num_needles, cities):
    assert num_needles <= len(cities)
    assert num_needles == 1, "Only support retrieve 1 city for now."
    if num_needles == 1:
        return QUESTION_STR.format(cities[0]) + " number is: "
    else:
        return QUESTION_STR.format(", ".join(cities[:num_needles])) + " numbers are: "


def generate_random_number(num_digits):
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return np.random.randint(lower_bound, upper_bound)


def load_haystack(tokenizer, max_context_length, haystack_dir):
    context = ""
    # do not count <s>
    while len(tokenizer.encode(context)) - 1 < max_context_length:
        if dist.get_rank() == 0:
            print(f"Current Context Length: {len(tokenizer.encode(context))-1}")
            print(f'Loading files {glob.glob(f"{haystack_dir}/*.txt")[:5]}')
        for file in glob.glob(f"{haystack_dir}/*.txt"):
            with open(file, "r") as f:
                context += f.read()
            if len(tokenizer.encode(context)) - 1 > max_context_length:
                break
    tokenized_haystack = tokenizer.encode(context)
    return tokenized_haystack


## replace the above function by loading from a file to save tokenization time
def load_tokenized_haystack(context_directory, max_context_length):
    load_path = f"{context_directory}/tokenized_haystack_{max_context_length}.pt"
    if os.path.exists(load_path):
        tokenized_haystack = torch.load(load_path)
        return tokenized_haystack
    return None


def construct_prompt(
    tokenized_haystack,
    tokenized_prefix,
    tokenized_postfix,
    tokenized_needle,
    context_length,
    tokenized_distractor_list,
    depth,
):
    # insert the needle into depth of the haystack
    period_tokens = [
        13,
        3343,
    ]  # this is the period token for llama3 tokenizer. https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/tokenizer.json
    prompt = tokenized_haystack[: context_length - PROMPT_TOKENS]  # test it.
    if depth == 0:
        start_index = 0
    else:
        start_index = int(len(prompt) * depth)
        # find the closest period token
        for i in range(start_index, len(prompt)):
            if prompt[i] in period_tokens:
                start_index = i + 1
                break
    prompt = prompt[:start_index] + tokenized_needle + prompt[start_index:]
    # insert distractors
    for distractor in tokenized_distractor_list:
        start_index = np.random.randint(0, len(prompt))
        for i in range(start_index, len(prompt)):
            if prompt[i] in period_tokens:
                start_index = i + 1
                break
        prompt = prompt[:start_index] + distractor + prompt[start_index:]
    prompt = tokenized_prefix + prompt + tokenized_postfix
    return prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--max-new-tokens", type=int, default=7, help="Maximum number of new tokens to generate")
    parser.add_argument("--min-context-length", type=int, default=4000)
    parser.add_argument("--max-context-length", type=int, default=64000)
    parser.add_argument("--context-interval", type=int, default=4000)
    parser.add_argument("--depth-interval", type=str, default="0.2")
    parser.add_argument("--rnd-number-digits", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--num-distractors", type=int, default=0)
    parser.add_argument("--num-cities-retrieval", type=int, default=1)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--haystack-dir", type=str, required=True, help="Haystack directory")
    return parser.parse_args()


depth_interval = Decimal("0.2")


def main():
    args = get_args()

    min_context_length = args.min_context_length
    max_context_length = args.max_context_length
    context_interval = args.context_interval
    num_cities_retrieval = args.num_cities_retrieval
    depth_interval = Decimal(args.depth_interval)
    haystack_dir = args.haystack_dir

    rnd_number_digits = args.rnd_number_digits
    num_samples = args.num_samples
    num_distractor = args.num_distractors

    config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    parallel_config = ParallelismArgs(
        dp=1,
        pp=args.pp,
        tp=args.tp,
        sp=args.sp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.REDUCE_SCATTER,
        # tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )

    # Initialise all process groups
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
        sequence_parallel_size=parallel_config.sp,
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

    # Reset the model max context length
    if model_config.max_position_embeddings <= max_context_length:
        model_config.max_position_embeddings = max_context_length

    # copy from _init_model
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

    # code for evaluation
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        question_str = generate_question_str(num_cities_retrieval, CITIES)

        tokenized_prefix = tokenizer.encode(PREFIX)[1:]
        tokenized_postfix = tokenizer.encode(question_str)[1:]
        tokenized_haystack = load_haystack(tokenizer, max_context_length - PROMPT_TOKENS, haystack_dir)[1:]

        DISTRACTOR_LIST = [
            "\nThe special magic New York number is: {}.\n",
            "\nThe special magic London number is: {}.\n",
            "\nThe special magic Paris number is: {}.\n",
            "\nThe special magic Tokyo number is: {}.\n",
            "\nThe special magic Beijing number is: {}.\n",
            "\nThe special magic Berlin number is: {}.\n",
        ]

        random_number_list = [generate_random_number(rnd_number_digits) for i in range(num_samples)]

        depth_values = [float(i) for i in np.arange(Decimal("0.0"), Decimal("1.0") + depth_interval, depth_interval)]
        distractor_number_list = [int(np.random.randint(10**rnd_number_digits)) for i in range(num_distractor)]
        distractor_str_list = [
            DISTRACTOR_LIST[i % len(DISTRACTOR_LIST)].format(distractor_number_list[i]) for i in range(num_distractor)
        ]
        tokenized_distractor_list = [tokenizer.encode(distractor_str)[1:] for distractor_str in distractor_str_list]

        results_dict = {}

        if dist.get_rank() == 0:
            context_lengths = tqdm(range(min_context_length, max_context_length + 1, context_interval))
        else:
            context_lengths = range(min_context_length, max_context_length + 1, context_interval)

        for context_length in context_lengths:
            correct = 0
            total = 0
            # Store the result in the dictionary
            if context_length not in results_dict:
                results_dict[context_length] = {}
            for depth in depth_values:
                depth_correct = 0.0
                depth_total = 0.0
                for random_number in random_number_list:
                    needle_str = NEEDLE_FORMAT.format(random_number)
                    ground_truth_str = str(random_number)
                    tokenized_needle = tokenizer.encode(needle_str)[1:]
                    tokenizer_answer = tokenizer.encode(ground_truth_str)[1:]

                    prompt = construct_prompt(
                        tokenized_haystack,
                        tokenized_prefix,
                        tokenized_postfix,
                        tokenized_needle,
                        context_length,
                        tokenized_distractor_list,
                        depth,
                    )
                    input_ids = torch.tensor(prompt).to(DEVICE)
                    prompt_text = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
                    ground_truth_str = tokenizer.decode(
                        torch.tensor(tokenizer_answer).to(DEVICE), clean_up_tokenization_spaces=False
                    )

                    dummy_inputs = [
                        prompt_text,
                    ]
                    pad_to_multiple_of = (
                        parallel_context.tensor_parallel_size * parallel_context.sequence_parallel_size * 2
                    )
                    outputs = decode_text_simple(
                        input_texts=dummy_inputs,
                        model=model.model,
                        tokenizer=tokenizer,
                        parallel_context=parallel_context,
                        max_new_tokens=args.max_new_tokens,
                        padding_left=False,  # right padding.
                        pad_to_multiple_of=pad_to_multiple_of,
                    )

                    for output in outputs:
                        genenration = output["generation"]
                        if ground_truth_str in genenration:
                            correct += 1
                        total += 1
                        depth_correct += ground_truth_str in genenration
                        depth_total += 1
                        log_rank(
                            f"Context length: {context_length}, Depth: {depth}, Genenration: {genenration}, Ground Truth: {ground_truth_str}, Passed test: {ground_truth_str in genenration}",
                            logger=logger,
                            level=logging.INFO,
                            rank=0,
                        )

                # register the result for this depth
                results_dict[context_length][depth] = depth_correct / depth_total
                log_rank(
                    f"Context length: {context_length}, Depth: {depth}, Passed test: {results_dict[context_length][depth]}",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
            # if output_rank:
            log_rank(
                f"Context length: {context_length}, Accuracy: {correct/total}",
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

        results_dict_path = "results.pkl"
        with open(results_dict_path, "wb") as f:
            pickle.dump(results_dict, f)
        print(f"Evaluation result saved to {results_dict_path}")

    dist.barrier()

    # Get the maximum memory allocated
    max_memory_allocated = torch.cuda.max_memory_allocated()
    log_rank(
        f"Maximum CUDA memory allocated: {max_memory_allocated / 1024 ** 3} GB",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )


if __name__ == "__main__":
    main()
