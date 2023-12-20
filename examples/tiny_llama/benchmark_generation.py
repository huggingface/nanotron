from nanotron.models.llama import LlamaForTraining
from nanotron.core.process_groups_initializer import get_process_groups
from nanotron.core.random import set_random_seed
from nanotron.core.parallelism.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.core import logging
from nanotron.core.parallelism.pipeline_parallelism.engine import OneForwardOneBackwardPipelineEngine
from nanotron.core.parallelism.tensor_parallelism.enum import TensorParallelLinearMode
from nanotron.config import ParallelismArgs, LoggingArgs
from nanotron.core.serialize.weights import load_weights
from nanotron.core.logging import log_rank
from nanotron.core.process_groups_initializer import DistributedProcessGroups
from nanotron.core.utils import init_on_device_and_dtype
from nanotron.models.llama import LlamaDecoderLayer, RotaryEmbedding
from nanotron.core.parallelism.tensor_parallelism.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
)
from nanotron.core.parallelism.parameters import NanotronParameter, sanity_check
from nanotron.core.parallelism.pipeline_parallelism.block import PipelineBlock
from nanotron.core.parallelism.tied_parameters import (
    create_pg_for_tied_weights,
    tie_parameters,
)
from nanotron.logger import LogItem

import os
import math

import numpy as np
import torch.nn as nn
from typing import Callable, List, Optional
from pathlib import Path
import torch.distributed as dist
from transformers import LlamaConfig, LlamaTokenizerFast
import torch
import itertools
import time
import sys
import logging as lg
from dataclasses import dataclass

from nanotron.core.serialize.serialize import fs_open
from nanotron.generation import GenerationConfig, GenerationInput, TokenizerConfig, greedy_search
from nanotron.helpers import set_logger_verbosity_format
from utils import init_model
from nanotron.config import Config

logger = logging.get_logger(__name__)

def log_throughput(
    config: Config,
    dpg: DistributedProcessGroups,
    model_tflops=0,
    hardware_tflops=0,
    tokens_per_sec=0,
    bandwidth=0,
):
    micro_batch_size = config.micro_batch_size
    n_micro_batches_per_batch = config.batch_accumulation_per_replica
    global_batch_size = micro_batch_size * n_micro_batches_per_batch * dpg.dp_pg.size()
    sequence_length = config.sequence_length
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    csv_filename = config.benchmark_csv_path
    table_log = [
        LogItem("model_name", config.model_name, "s"),
        LogItem("nodes", math.ceil(dpg.world_pg.size() / 8), "d"),
        LogItem("seq_len", (sequence_length), "d"),
        LogItem("mbs", micro_batch_size, "d"),
        LogItem("batch_accum", n_micro_batches_per_batch, "d"),
        LogItem("gbs", global_batch_size, "d"),
        LogItem("mTFLOPs", model_tflops, ".2f"),
        LogItem("hTFLOPs", hardware_tflops, ".2f"),
        LogItem("tok/s/gpu", tokens_per_sec / dpg.world_pg.size(), ".2f"),
        LogItem("Bandwidth (GB/s)", bandwidth, ".2f"),
        LogItem("Mem Alloc (GB)", torch.cuda.max_memory_allocated() / 1024**3, ".2f"),
        LogItem("Mem Res (GB)", torch.cuda.max_memory_reserved() / 1024**3, ".2f"),
    ]
    
    column_widths = [max(len(item.tag), len(f"{item.scalar_value:{item.log_format}}")) for item in table_log]
    header_row = "| " + " | ".join([item.tag.ljust(width) for item, width in zip(table_log, column_widths)]) + " |"
    separator_row = "| " + " | ".join(['-' * width for width in column_widths]) + " |"
    data_row = "| " + " | ".join([f"{item.scalar_value:{item.log_format}}".ljust(width) for item, width in zip(table_log, column_widths)]) + " |"
    table_output = f"{header_row}\n{separator_row}\n{data_row}"

    log_rank(
        table_output,
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    
    import csv

    if dist.get_rank(dpg.world_pg) == 0:
        if not os.path.exists(csv_filename):
            with fs_open(csv_filename, mode="w") as fo:
                writer = csv.writer(fo)
                writer.writerow([item.tag for item in table_log])
                writer.writerow([f"{item.scalar_value:{item.log_format}}" for item in table_log])
        elif model_tflops > 0:
            # replace line with same job_id
            with fs_open(csv_filename, mode="r") as fi:
                lines = fi.readlines()
            with fs_open(csv_filename, mode="w") as fo:
                writer = csv.writer(fo)
                for line in lines:
                    if line.startswith(slurm_job_id):
                        writer.writerow([f"{item.scalar_value:{item.log_format}}" for item in table_log])
                    else:
                        fo.write(line)
        else:
            with fs_open(csv_filename, mode="a") as fo:
                writer = csv.writer(fo)
                writer.writerow([f"{item.scalar_value:{item.log_format}}" for item in table_log])

@dataclass
class BenchArgs:
    model_name: str
    sequence_length: int
    micro_batch_size: int
    batch_accumulation_per_replica: int
    benchmark_csv_path: str    

def main():
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        recompute_granularity=None,
        tp_linear_async_communication=False,
    )

    logging_config = LoggingArgs(
        log_level="info",
        log_level_replica="info",
        iteration_step_info_interval=9999999999999,
        tensorboard_logger=None
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


    model_name = "huggyllama/llama-7b"
    model_config = LlamaConfig.from_pretrained(model_name)
    model = init_model(
        model_builder=lambda: LlamaForTraining(config=model_config, dpg=dpg, parallel_config=parallel_config),
        model_config=model_config,
        parallel_config=parallel_config,
        dtype=dtype,
        dpg=dpg,
        make_ddp=False,
        logger=logger,
    )

    # Load checkpoint
    # checkpoint_path = Path("/home/ubuntu/.cache/huggingface/hub/models--HuggingFaceBR4--llama-7b-orig/snapshots/2160b3d0134a99d365851a7e95864b21e873e1c3/model")
    # log_rank(
    #     f"Loading checkpoint from {checkpoint_path}:",
    #     logger=logger,
    #     level=logging.INFO,
    #     rank=0,
    # )
    # load_weights(model=model, dpg=dpg, root_folder=checkpoint_path)

    model.eval()

    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    set_random_seed(1234)
    model_size = sum([p.numel() * p.data.element_size() for p in itertools.chain(model.parameters(), model.buffers())])

    log_rank(
        f"Model size: {model_size / 1024 / 1024:.02f} MB",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    dummy_inputs = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
    ]

    MAX_NEW_TOKENS = 10
    MAX_INPUT_LENGTH = 1500 #TODO: set MAX_INPUT_LENGTH to the max length of the tokenized inputs
    #TODO: Investigate why increasing MAX_MICRO_BATCH_SIZE gives non-sense second and tokens/sec
    MAX_MICRO_BATCH_SIZE = 2

    bench_config = BenchArgs(
        model_name=model_name,
        sequence_length=MAX_NEW_TOKENS,
        micro_batch_size=MAX_MICRO_BATCH_SIZE,
        batch_accumulation_per_replica=1,
        benchmark_csv_path="benchmark.csv"
    )

    outputs_generator = greedy_search(
            input_iter=(GenerationInput(text=text) for text in dummy_inputs),
            tokenizer=tokenizer,
            # TODO @thomasw21: From ModelWithLoss extract the model.
            model=model.model,
            # TODO @thomasw21: Figure out how to pass p2p.
            p2p=model.model.p2p,
            dpg=dpg,
            generation_config=GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, max_micro_batch_size=MAX_MICRO_BATCH_SIZE),
            tokenizer_config=TokenizerConfig(max_input_length=MAX_INPUT_LENGTH),
        )

    dist.barrier()

    try:
        i = 0

        consumed_train_samples = 1 #TODO: Should depends on input
        sequence_length = MAX_NEW_TOKENS
        
                
        while True:
            
            torch.cuda.synchronize()
            dist.barrier()

            iteration_start_time = time.perf_counter()

            # return StopIteration if outputs_generator is empty
            output = next(outputs_generator)
            
            dist.barrier()
            torch.cuda.synchronize()

            elapsed_time_per_iteration = time.perf_counter() - iteration_start_time

            dist.barrier()

            input_ids = output.input_ids
            generated_ids = output.generation_ids
            
            if isinstance(input_ids, TensorPointer):
                assert isinstance(generated_ids, TensorPointer)
                continue
            assert isinstance(generated_ids, torch.Tensor)

            input_generated = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
            output_generated = tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False)
            
            tokens_per_sec = (
                consumed_train_samples * sequence_length / (elapsed_time_per_iteration)
            )  # tokens_per_sec is calculated using sequence_length

            model_tflops, hardware_tflops = model.model.get_flops_per_sec(
                iteration_time_in_sec=elapsed_time_per_iteration,
                sequence_length=sequence_length,
                global_batch_size=consumed_train_samples,
            )

            log_rank(f"Iteration {i}", logger=logger, level=logging.INFO, rank=0)
            
            log_rank(
                f"\n\t >>> Input: {input_generated}",
                logger=logger,
                level=logging.INFO,
                rank=0
            )
            log_rank(
                f"\t >>> Output: {output_generated}\n",
                logger=logger,
                level=logging.INFO,
                rank=0
            )            
                        
            log_throughput(
                bench_config,
                dpg,
                model_tflops,
                hardware_tflops,
                tokens_per_sec,
                bandwidth = model_size * tokens_per_sec / 1e9
            )

            
            i += 1

            dist.barrier()

    except StopIteration:
        pass

    dist.barrier()

if "__main__" == __name__:
    main()
