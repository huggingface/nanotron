"""
Nanotron Inference Script

Usage:
CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=1    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29600    --max_restarts=0    --tee=3 tests/test_llama_generation.py --ckpt-path /fsx/haojun/lighteval_evaluation_model/Llama-3-8B-split

"""

import argparse
from pathlib import Path

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    LoggingArgs,
    ParallelismArgs,
    get_config_from_file,
)
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.models.llama3_1 import LlamaForTraining as LlamaForTraining_test
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
from nanotron.trainer import mark_tied_parameters
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    return parser.parse_args()


def main():
    args = get_args()

    assert args.ckpt_path.exists(), f"Checkpoint path {args.ckpt_path} does not exist"

    config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    # as tp/pp/sp will introduce small differences in the output, we need to set them to 1
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        sp=1,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
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

    # Get synchronized random states
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)}
        )
    else:
        # We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
        random_states = RandomStates({})

    model = build_model(
        model_builder=lambda: LlamaForTraining_test(
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

    ## Tokenizer
    pretrained_model_name_or_path = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    ## Transformers model
    attn_implementation = "flash_attention_2"  # 'sdpa' / 'flash_attention_2'
    transformer_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation
    ).to("cuda")

    model.eval()
    transformer_model.eval()

    # Input prompt
    input_text = "The future of AI is"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # TODO: also compare the generation with cache.
    for i in range(args.max_new_tokens):
        output_logits = transformer_model(**inputs)[
            "logits"
        ]  # output logits of transformer model is still in float32 even dtype is bfloat16
        my_output_logits = model.model(inputs["input_ids"], inputs["attention_mask"])
        next_token_id = torch.argmax(output_logits[:, -1, :], dim=-1)
        my_next_token_id = torch.argmax(my_output_logits[-1, :, :], dim=-1)
        try:
            # test logits and generation on the same time.
            torch.testing.assert_close(
                my_output_logits[:, 0, :], output_logits[0, :, :], rtol=1e-5, atol=1e-5
            )  # check if the output logits are close
            assert torch.equal(
                my_output_logits[:, 0, :], output_logits[0, :, :]
            ), "Output logits are not the same"  # check if the output logits are the same
        except AssertionError as e:
            print(f"Token {i+1} failed: {e}")
            print("Reference: ", output_logits)
            print("My output: ", my_output_logits)

        assert (
            next_token_id == my_next_token_id
        ), f"Predictions are not the same: {next_token_id} != {my_next_token_id}"
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_id.unsqueeze(-1)], dim=-1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones(1, 1).to(dtype=torch.bool, device="cuda")], dim=-1
        )
        if next_token_id == tokenizer.eos_token_id:
            break
    print("Input prompt:", input_text)
    print(
        "Generated text:", tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)[len(input_text) :]
    )  # remove the input text from the generated text
    print("Test passed!")

    dist.barrier()


if __name__ == "__main__":
    main()
