import dataclasses
import time
from itertools import chain, islice
from typing import TYPE_CHECKING, Generator, Iterable, List, Optional, Tuple, Union

import torch

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import BenchArgs, GenerationArgs
from nanotron.distributed import ProcessGroup, get_global_rank
from nanotron.generation.generate_store import Store, attach_store
from nanotron.generation.sampler import BasicSampler, GreedySampler, SamplerType, TopKSampler, TopPSampler
from nanotron.helpers import log_throughput
from nanotron.models.llama import LlamaModel
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import get_min_max_rank
from nanotron.parallel.pipeline_parallel.context_manager import attach_pipeline_state_to_model
from nanotron.parallel.pipeline_parallel.p2p import P2PTensorMetaData, view_as_contiguous
from nanotron.parallel.pipeline_parallel.state import PipelineEvalBatchState
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.utils import get_untyped_storage

if TYPE_CHECKING:
    try:
        from transformers import PreTrainedTokenizer
    except ImportError:
        PreTrainedTokenizer = None


logger = logging.get_logger(__name__)


@dataclasses.dataclass
class GenerationInput:
    text: str


@dataclasses.dataclass
class GenerationInputs:
    input_ids: Union[torch.Tensor, TensorPointer]  # [B, S]
    input_masks: Union[torch.Tensor, TensorPointer]


@dataclasses.dataclass
class GenerationOutput:
    input_ids: Union[torch.Tensor, TensorPointer]
    generation_ids: Union[torch.Tensor, TensorPointer]
    return_logits: Optional[Union[torch.Tensor, TensorPointer]] = None


@dataclasses.dataclass
class GenerationStates:
    new_input_ids: Union[torch.Tensor, TensorPointer]
    new_input_mask: Union[torch.Tensor, TensorPointer]
    store: Store

    # The rest of the state I need to reconstruct the generated output
    generation_ids: List[Union[torch.Tensor, TensorPointer]]
    generation_mask: List[Union[torch.Tensor, TensorPointer]]


@dataclasses.dataclass
class TokenizerConfig:
    max_input_length: Optional[int]
    truncation: Optional[Union[str, bool]] = None
    padding: Optional[Union[str, bool]] = None


def chunks(iterable, chunk_size: int) -> Generator[List, None, None]:
    """Yield successive n-sized chunks from `iterable`"""
    assert chunk_size >= 1
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, chunk_size - 1)))


def micro_batcher(
    input_iter: Iterable[GenerationInput],
    tokenizer: "PreTrainedTokenizer",
    max_micro_batch_size: int,
    tokenizer_config: TokenizerConfig,
    parallel_context: ParallelContext,
    input_rank: int,
) -> Generator[GenerationInputs, None, None]:
    """
    Returns:
        input_ids: [max_micro_batch_size, max_input_length]
        input_masks: [max_micro_batch_size, max_input_length]
    """
    if tokenizer_config.padding is None:
        tokenizer_config.padding = "max_length" if tokenizer_config.max_input_length is not None else True
    if tokenizer_config.truncation is None:
        tokenizer_config.truncation = True if tokenizer_config.max_input_length is not None else None

    for micro_batch_id, micro_batch in enumerate(chunks(input_iter, chunk_size=max_micro_batch_size)):
        if len(micro_batch) == 0:
            # Empty micro batches don't matter
            return

        if micro_batch_id % parallel_context.dp_pg.size() != dist.get_rank(parallel_context.dp_pg):
            # Each dp is responsible for its own micro batches
            continue

        if dist.get_rank(parallel_context.pp_pg) == input_rank:
            encodings = tokenizer(
                [elt.text for elt in micro_batch],
                return_tensors="pt",
                return_attention_mask=True,
                padding=tokenizer_config.padding,
                max_length=tokenizer_config.max_input_length,
                truncation=tokenizer_config.truncation,
            )

            encodings["attention_mask"] = encodings.attention_mask.to(dtype=torch.bool, device="cuda")
            encodings.to("cuda")
            yield GenerationInputs(input_ids=encodings.input_ids, input_masks=encodings.attention_mask)
        else:
            yield GenerationInputs(
                input_ids=TensorPointer(group_rank=input_rank), input_masks=TensorPointer(group_rank=input_rank)
            )


def micro_splitter(
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    max_micro_batch_size: int,
    parallel_context: ParallelContext,
    input_rank: int,
) -> Generator[GenerationInputs, None, None]:
    """
    Returns:
        input_ids: [max_micro_batch_size, max_input_length]
        input_masks: [max_micro_batch_size, max_input_length]
    """
    for micro_batch_id, (micro_batch_ids, micro_batch_mask) in enumerate(
        zip(torch.split(input_ids, max_micro_batch_size), torch.split(input_mask, max_micro_batch_size))
    ):
        if len(micro_batch_ids) == 0:
            # Empty micro batches don't matter
            return

        # if micro_batch_id % parallel_context.dp_pg.size() != dist.get_rank(parallel_context.dp_pg):
        #     # Each dp is responsible for its own micro batches
        #     continue

        if dist.get_rank(parallel_context.pp_pg) == input_rank:
            micro_batch_mask = micro_batch_mask.to(dtype=torch.bool, device="cuda")
            micro_batch_mask.to("cuda")
            yield GenerationInputs(input_ids=micro_batch_ids.clone(), input_masks=micro_batch_mask.clone())
        else:
            yield GenerationInputs(
                input_ids=TensorPointer(group_rank=input_rank), input_masks=TensorPointer(group_rank=input_rank)
            )


@torch.inference_mode()
def decode_text(
    input_iter: Iterable[GenerationInput],
    tokenizer: "PreTrainedTokenizer",
    model: LlamaModel,
    parallel_context: ParallelContext,
    generation_config: GenerationArgs,
    tokenizer_config: Optional[TokenizerConfig],
    max_micro_batch_size: int,
    max_new_tokens: int,
    is_bench: bool = False,
    logits_are_batch_first: bool = True,
) -> Generator[GenerationOutput, None, None]:
    """We assume the following:
    - Everyone receives ALL the input text. # TODO @thomasw21: technically only specific ranks need to receive input.
    - Only a specific rank will output the generated text_ids as `torch.Tensor`, the others return a `TensorPointer`. # TODO @thomasw21: Maybe all ranks should return the text.
    - We assume that within a model replica, the inputs are already synchronized.
    """
    decoder_input_rank, decoder_logit_rank = get_min_max_rank(module=model)

    if generation_config:
        if isinstance(generation_config.sampler, str):
            sampler_type = SamplerType(generation_config.sampler.upper())
        else:
            sampler_type = generation_config.sampler
    else:
        sampler_type = SamplerType.GREEDY

    # Compute flag
    is_decoder_input_rank = dist.get_rank(parallel_context.pp_pg) == decoder_input_rank
    is_decoder_logit_rank = dist.get_rank(parallel_context.pp_pg) == decoder_logit_rank
    max_nb_microbatches = decoder_logit_rank - decoder_input_rank + 1

    p2p = model.p2p

    # replicate input for n_samples times when using TOP_P or TOP_K samplers, in order to get diverse results
    if generation_config and generation_config.n_samples:
        if sampler_type != SamplerType.TOP_P and sampler_type != SamplerType.TOP_K:
            raise ValueError("Only support n_samples for TOP_P and TOP_K sampler")
        input_iter = [
            GenerationInput(text=input.text) for input in input_iter for _ in range(generation_config.n_samples)
        ]

    # That's annoying but I need this as soon as there's a change communication "cross"
    pipeline_state = PipelineEvalBatchState()
    with attach_pipeline_state_to_model(model=model, pipeline_state=pipeline_state):
        # We query the first `pipeline_size` batches
        for batches in chunks(
            iterable=micro_batcher(
                input_iter=input_iter,
                tokenizer=tokenizer,
                max_micro_batch_size=max_micro_batch_size,
                tokenizer_config=tokenizer_config,
                input_rank=decoder_input_rank,
                parallel_context=parallel_context,
            ),
            chunk_size=max_nb_microbatches,
        ):
            if len(batches) == 0:
                # It means we're out of element
                return

            # Number of micro batches
            number_states_in_buffer = len(batches)
            # Otherwise the pipelining doesn't work
            assert number_states_in_buffer <= max_nb_microbatches
            is_max_nb_microbatches = number_states_in_buffer == max_nb_microbatches

            # Initialize decoder states
            decoder_states: Iterable[GenerationStates] = (
                GenerationStates(
                    new_input_ids=batch.input_ids,
                    new_input_mask=batch.input_masks,
                    store=Store(),
                    generation_ids=[batch.input_ids],
                    generation_mask=[batch.input_masks],
                )
                for batch in batches
            )

            if is_bench:
                start_time, elapsed_time_first_iteration = time.perf_counter(), 0

            for generation_iter in range(max_new_tokens):
                if is_bench and generation_iter == 0:
                    torch.cuda.synchronize()
                    elapsed_time_first_iteration = start_time - time.perf_counter()

                all_new_decoder_input_ids_and_mask_same_rank: List[
                    Tuple[Union[torch.LongTensor, TensorPointer], Union[torch.BoolTensor, TensorPointer]]
                ] = []
                new_decoder_states: List[GenerationStates] = []
                for state_id, state in enumerate(decoder_states):
                    new_decoder_states.append(state)
                    # Get the new logits
                    if generation_config.use_cache:
                        with attach_store(model=model, store=state.store):
                            # transpose: [sequence_length, batch_size, vocab_size] -> [batch_size, sequence_length, vocab_size]
                            sharded_logits = model(
                                input_ids=state.new_input_ids,
                                input_mask=state.new_input_mask,
                            )
                    else:
                        if isinstance(state.new_input_ids, torch.Tensor):
                            batch_generated_ids = torch.cat(state.generation_ids, dim=-1)
                            batch_generated_mask = torch.cat(state.generation_mask, dim=-1)
                        else:
                            batch_generated_ids = state.new_input_ids
                            batch_generated_mask = state.new_input_mask
                        sharded_logits = model(
                            input_ids=batch_generated_ids,
                            input_mask=batch_generated_mask,
                        )

                    if isinstance(sharded_logits, torch.Tensor) and logits_are_batch_first:
                        sharded_logits = sharded_logits.transpose(0, 1)
                    # Communicate
                    # TODO @thomasw21: Make a diagram to show how this works
                    nb_send: int = 0
                    if is_decoder_input_rank:
                        if is_max_nb_microbatches:
                            if generation_iter == 0:
                                if state_id == number_states_in_buffer - 1:
                                    # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                    nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                                else:
                                    # Send everything
                                    nb_send = len(pipeline_state.microbatches_activations_to_send)
                            else:
                                # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                        else:
                            if number_states_in_buffer - 1 == state_id or generation_iter == 0:
                                # Send everything
                                nb_send = len(pipeline_state.microbatches_activations_to_send)
                            else:
                                # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                    else:
                        if state_id == number_states_in_buffer - 1:
                            if not is_max_nb_microbatches:
                                nb_send = len(pipeline_state.microbatches_activations_to_send)
                    for _ in range(nb_send):
                        pipeline_state.run_communication()

                    if is_decoder_logit_rank:
                        assert isinstance(sharded_logits, torch.Tensor)

                        # run a logit chooser.
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

                        new_decoder_input_ids = sampler(sharded_logits=sharded_logits[:, -1, :])

                        # TODO @thomasw21: Handle this correctly, ie from some point after <eos> this should only generate masked tokens
                        # TODO @thomasw21: Actually I can probably build this thing on the next device directly. Will save some communication
                        new_decoder_input_mask = torch.ones(
                            size=(new_decoder_input_ids.shape[0], 1),
                            dtype=torch.bool,
                            device=new_decoder_input_ids.device,
                        )

                        # TODO @thomasw21: We need to have stop condition.

                        # broadcast new_tokens to everyone
                        if decoder_input_rank == decoder_logit_rank:
                            # It's the same rank so no need to do anything too fancy
                            all_new_decoder_input_ids_and_mask_same_rank.append(
                                (new_decoder_input_ids, new_decoder_input_mask)
                            )
                        else:
                            pipeline_state.register_send_activation(
                                new_decoder_input_ids, to_rank=decoder_input_rank, p2p=p2p
                            )
                            pipeline_state.register_send_activation(
                                new_decoder_input_mask, to_rank=decoder_input_rank, p2p=p2p
                            )
                            if not is_max_nb_microbatches and state_id == number_states_in_buffer - 1:
                                # Send new_decoder_input_ids AND new_decoder_input_ids
                                pipeline_state.run_communication()
                                pipeline_state.run_communication()

                    else:
                        assert isinstance(sharded_logits, TensorPointer)

                all_new_decoder_input_ids_and_mask: Iterable[
                    Tuple[Union[torch.LongTensor, TensorPointer], Union[torch.BoolTensor, TensorPointer]]
                ]
                if is_decoder_input_rank:
                    # We receive the tensor from other ranks unless `decoder_input_rank` == `decoder_logit_rank` in which case `all_new_decoder_input_ids` is already populated.
                    if decoder_input_rank == decoder_logit_rank:
                        # `all_new_decoder_input_ids_and_mask_same_rank` is already populated. Since `decoder_input_rank` and `decoder_logit_rank` are the same, there's no need to communicate as we can just store the new input_ids in a list.
                        assert len(all_new_decoder_input_ids_and_mask_same_rank) == number_states_in_buffer
                        all_new_decoder_input_ids_and_mask = all_new_decoder_input_ids_and_mask_same_rank
                    else:

                        def generator():
                            for _ in range(number_states_in_buffer):
                                pipeline_state.register_recv_activation(from_rank=decoder_logit_rank, p2p=p2p)
                                pipeline_state.register_recv_activation(from_rank=decoder_logit_rank, p2p=p2p)
                                while len(pipeline_state.activations_buffer) < 2:
                                    pipeline_state.run_communication()
                                new_decoder_input_ids = pipeline_state.activations_buffer.popleft()
                                new_decoder_input_mask = pipeline_state.activations_buffer.popleft()
                                yield new_decoder_input_ids, new_decoder_input_mask

                        all_new_decoder_input_ids_and_mask = iter(generator())
                else:
                    all_new_decoder_input_ids_and_mask = (
                        (TensorPointer(group_rank=decoder_input_rank), TensorPointer(group_rank=decoder_input_rank))
                        for _ in range(number_states_in_buffer)
                    )

                # Create new decoder states
                decoder_states = (
                    GenerationStates(
                        new_input_ids=new_decoder_input_ids_and_mask[0],
                        new_input_mask=new_decoder_input_ids_and_mask[1],
                        store=state.store,
                        generation_ids=state.generation_ids + [new_decoder_input_ids_and_mask[0]],
                        generation_mask=state.generation_mask + [new_decoder_input_ids_and_mask[1]],
                    )
                    for state, new_decoder_input_ids_and_mask in zip(
                        new_decoder_states, all_new_decoder_input_ids_and_mask
                    )
                )

            if is_bench:
                # Compute throughput (tok/s/gpu). Note that the first generation is done with full seq_len, so we don't count it.
                torch.cuda.synchronize()
                total_time_sec = time.perf_counter() - start_time - elapsed_time_first_iteration
                # We generate 1 token per iteration per batch (batch=microbatch)
                # Number of tokens generated every iteration: gbs/iteration_time
                global_batch_size = len(batches) * parallel_context.dp_pg.size()
                tokens_per_sec = global_batch_size * max_new_tokens / total_time_sec

                model_tflops, hardware_tflops = model.get_flops_per_sec(
                    iteration_time_in_sec=total_time_sec,
                    sequence_length=max_new_tokens,
                    global_batch_size=global_batch_size,
                )

                bench_config = BenchArgs(
                    model_name=model.config._name_or_path,
                    sequence_length=max_new_tokens,
                    micro_batch_size=max_micro_batch_size,
                    batch_accumulation_per_replica=1,
                    benchmark_csv_path="benchmark.csv",
                )

                model_size = sum(
                    [p.numel() * p.data.element_size() for p in chain(model.parameters(), model.buffers())]
                )

                log_throughput(
                    bench_config,
                    parallel_context,
                    model_tflops,
                    hardware_tflops,
                    tokens_per_sec,
                    bandwidth=model_size * tokens_per_sec / 1e9,
                )

            # Flush communication
            for _ in range(
                max(
                    len(pipeline_state.microbatches_activations_to_send),
                    len(pipeline_state.microbatches_activations_to_recv),
                )
            ):
                pipeline_state.run_communication()
            assert len(pipeline_state.microbatches_activations_to_send) == 0
            assert len(pipeline_state.microbatches_activations_to_recv) == 0

            # Yield result
            decoder_states = list(decoder_states)
            for state, batch in zip(decoder_states, batches):
                if is_decoder_input_rank:
                    assert all(isinstance(elt, torch.Tensor) for elt in state.generation_ids)
                    batch_generated_ids = torch.cat(state.generation_ids, dim=-1)
                    batch_generated_mask = torch.cat(state.generation_mask, dim=-1)
                else:
                    assert all(isinstance(elt, TensorPointer) for elt in state.generation_ids)
                    batch_generated_ids = TensorPointer(group_rank=decoder_input_rank)
                    batch_generated_mask = TensorPointer(group_rank=decoder_input_rank)

                # Broadcast all data
                batch_generated_ids, batch_generated_mask = broadcast_tensors(
                    [batch_generated_ids, batch_generated_mask],
                    group_src=decoder_input_rank,
                    group=parallel_context.pp_pg,
                )
                batch.input_ids, batch.input_masks = broadcast_tensors(
                    [batch.input_ids, batch.input_masks], group_src=decoder_input_rank, group=parallel_context.pp_pg
                )

                # Flush the store to release memory
                state.store.flush()
                assert len(state.store) == 0

                if dist.get_rank(parallel_context.pp_pg) == decoder_input_rank:
                    assert (
                        batch_generated_ids.shape[0] == batch.input_ids.shape[0]
                    ), f"Batch size needs to match {batch_generated_ids.shape[0]} != {batch.input_ids.shape[0]}"
                    assert (
                        batch_generated_mask.shape[0] == batch.input_ids.shape[0]
                    ), f"Batch size needs to match {batch_generated_mask.shape[0]} != {batch.input_ids.shape[0]}"
                    assert (
                        batch_generated_ids.shape[1] == batch_generated_mask.shape[1]
                    ), f"Sequence length needs to match {batch_generated_ids.shape[1]} != {batch_generated_mask.shape[0]}"

                for i, (generated_ids, generated_mask) in enumerate(zip(batch_generated_ids, batch_generated_mask)):
                    # TODO @thomasw21: We could actually have all ranks return the output, since it's been already broadcasted
                    if dist.get_rank(parallel_context.pp_pg) == decoder_input_rank:
                        input_ids = batch.input_ids[i]
                        input_mask = batch.input_masks[i]
                        yield GenerationOutput(
                            input_ids=input_ids[input_mask],
                            generation_ids=generated_ids[generated_mask],
                        )
                    else:
                        yield GenerationOutput(
                            input_ids=TensorPointer(group_rank=decoder_input_rank),
                            generation_ids=TensorPointer(group_rank=decoder_input_rank),
                        )


def adjust_padding(input_ids, attention_mask, pad_token_id=128001, pad_to_multiple_of=4, padding_left=True):
    """
    This function pad or remove paddiings to make the sequence length a multiple of "pad_to_multiple_of"
    two modes: padding on the left or right.
    Use with decode_text_simple for 1M tokens context length.
    """
    # Count the number of padding tokens on the left side
    padding_tokens_count = (input_ids == pad_token_id).sum(dim=1).item()
    # padding_tokens_count = (input_ids == pad_token_id).sum(dim=1).item()
    input_length = input_ids.size(1)

    # Calculate the required padding to make the length a multiple of pad_to_multiple_of
    padding_length = (pad_to_multiple_of - input_length % pad_to_multiple_of) % pad_to_multiple_of

    if (
        padding_tokens_count >= pad_to_multiple_of - padding_length
    ):  # pad k tokens = remove 8-k tokens, if pad_to_multiple_of = 8
        # Remove excessive padding tokens to make the length a multiple of pad_to_multiple_of
        excess_padding = pad_to_multiple_of - padding_length
        ## padding is on the left or right.
        if padding_left:
            input_ids = input_ids[:, excess_padding:]
            attention_mask = attention_mask[:, excess_padding:]
        else:
            input_ids = input_ids[:, :-excess_padding]
            attention_mask = attention_mask[:, :-excess_padding]
    else:
        # Add padding tokens to the left side
        if padding_length > 0:
            padding = torch.full((input_ids.size(0), padding_length), pad_token_id, dtype=torch.long).to(
                "cuda"
            )  # [batch_size, padding_length]
            if padding_left:
                input_ids = torch.cat([padding, input_ids], dim=1)
                attention_mask = torch.cat([torch.zeros_like(padding), attention_mask], dim=1)
            else:
                input_ids = torch.cat([input_ids, padding], dim=1)
                attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)], dim=1)
    return input_ids, attention_mask


@torch.inference_mode()
def decode_text_simple(
    input_texts: List[str],
    tokenizer: "PreTrainedTokenizer",
    model: LlamaModel,
    parallel_context: ParallelContext,
    max_new_tokens: int,
    pad_to_multiple_of: Optional[int] = 8,
    padding_left: Optional[bool] = False,
):
    """
    Use this function only when dealing with long context.
    A simpler decode text function without PP aims to support long context(e.g. 1M tokens) inference by adjust padding.
    For sequence length = 1M, TP=8(reduce scatter mode) and SP=2 is needed. For sequence length <= 512K, TP=8(reduce scatter mode) and SP=1 is enough.
    Specifically, SP need padding on the right for now(faster with flash_attn_func and easier to implement).
    TODO: Only tested when batch size = 1.  KV cache is not implemented yet. For faster inference, consider using decode_text.
    """
    assert parallel_context.sp_pg.size() == 1 or (
        parallel_context.sp_pg.size() > 1 and not padding_left
    ), "For SP, should use padding right for inference."
    output_texts = []
    pad_token_id = tokenizer.pad_token_id
    sampler = GreedySampler(
        pg=parallel_context.tp_pg
    )  # Notice that the logics for gathering sharded logits(last dim) are implemented in the sampler.
    for input_text in input_texts:
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        num_input_tokens = attention_mask.sum().item()
        sp_world_size = dist.get_world_size(parallel_context.sp_pg)
        sp_rank = dist.get_rank(parallel_context.sp_pg)
        # Generate output
        for i in range(max_new_tokens):
            input_ids, attention_mask = adjust_padding(
                input_ids,
                attention_mask,
                pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_left=padding_left,
            )
            if parallel_context.sp_pg.size() > 1:
                assert (
                    input_ids.size(1) % (2 * sp_world_size) == 0
                ), "The sequence length should be a multiple of 2*world_size. Check padding function"
                chunk_size = input_ids.size(1) // (2 * sp_world_size)  # chunk size
            local_outputs = model(input_ids=input_ids, input_mask=attention_mask).transpose(
                0, 1
            )  # batch_size x seq_length/sp x vocab_size/tp
            # I don't gather the output along the sequnce level for SP. Because it takes way too much memory.
            if parallel_context.sp_pg.size() > 1:
                last_token = attention_mask.sum(dim=1).item() - 1  # the last token before padding
                last_token_chunk = last_token // chunk_size  # the chunk index of the last token
                GPU_idx = (
                    last_token_chunk if last_token_chunk < sp_world_size else 2 * sp_world_size - 1 - last_token_chunk
                )  # the GPU index of the last token among the SP group
                last_token_idx = (
                    last_token % chunk_size + chunk_size
                    if last_token_chunk >= sp_world_size
                    else last_token % chunk_size
                )  # the token index in the GPU
                if sp_rank == GPU_idx:
                    next_token_id = sampler(sharded_logits=local_outputs[:, last_token_idx, :])
                else:
                    next_token_id = torch.zeros((input_ids.size(0), 1), dtype=torch.long).to("cuda")
                dist.barrier(group=parallel_context.sp_pg)  # all process have to wait for the sampler to finish
                dist.broadcast(
                    next_token_id,
                    src=dist.get_global_rank(parallel_context.sp_pg, GPU_idx),
                    group=parallel_context.sp_pg,
                )  # broadcast the generated token to all process.  Source rank on global process group (regardless of group argument).
                # replace the padding with generated token(padding on the right side)
                if last_token + 1 < input_ids.size(1):  # replace the padding with generated token
                    input_ids[:, last_token + 1] = next_token_id
                    attention_mask[:, last_token + 1] = 1
                else:  # insert the generated token when there is no padding to replace
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)
            # without SP. We can just use sampler to generate the next token.
            else:
                if padding_left:
                    last_token = -1
                    next_token_id = sampler(sharded_logits=local_outputs[:, last_token, :])  # batch_size x 1
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)
                else:
                    last_token = attention_mask.sum(dim=1).item() - 1  # the last token before padding
                    next_token_id = sampler(sharded_logits=local_outputs[:, last_token, :])
                    if last_token + 1 < input_ids.size(1):  # replace the padding with generated token
                        input_ids[:, last_token + 1] = next_token_id
                        attention_mask[:, last_token + 1] = 1
                    else:  # insert the generated token when there is no padding to replace
                        input_ids = torch.cat([input_ids, next_token_id], dim=1)
                        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)

        generation = tokenizer.decode(input_ids[0, num_input_tokens:], skip_special_tokens=True)
        output_texts.append({"prompt": input_text, "generation": generation})
    return output_texts


@torch.inference_mode()
def decode_tokenized(
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    model: LlamaModel,
    parallel_context: ParallelContext,
    generation_config: GenerationArgs,
    max_micro_batch_size: int,
    max_new_tokens: int,
    returns_logits: Optional[bool] = False,
) -> Generator[GenerationOutput, None, None]:
    """We assume the following:
    - Everyone receives ALL the input text. # TODO @thomasw21: technically only specific ranks need to receive input.
    - Only a specific rank will output the generated text_ids as `torch.Tensor`, the others return a `TensorPointer`. # TODO @thomasw21: Maybe all ranks should return the text.
    - We assume that within a model replica, the inputs are already synchronized.
    """
    if returns_logits:
        raise NotImplementedError("return_logits is not implemented yet")

    if generation_config:
        if isinstance(generation_config.sampler, str):
            sampler_type = SamplerType(generation_config.sampler.upper())
        else:
            sampler_type = generation_config.sampler
    else:
        sampler_type = SamplerType.GREEDY

    decoder_input_rank, decoder_logit_rank = get_min_max_rank(module=model)

    # Compute flag
    is_decoder_input_rank = dist.get_rank(parallel_context.pp_pg) == decoder_input_rank
    is_decoder_logit_rank = dist.get_rank(parallel_context.pp_pg) == decoder_logit_rank
    max_nb_microbatches = decoder_logit_rank - decoder_input_rank + 1

    # TODO @thomasw21: Fix this as we shouldn't get P2P like that
    p2p = model.p2p

    # That's annoying but I need this as soon as there's a change communication "cross"
    pipeline_state = PipelineEvalBatchState()
    with attach_pipeline_state_to_model(model=model, pipeline_state=pipeline_state):
        # We query the first `pipeline_size` batches
        for batches in chunks(
            iterable=micro_splitter(
                input_ids,
                input_mask,
                max_micro_batch_size=max_micro_batch_size,
                parallel_context=parallel_context,
                input_rank=decoder_input_rank,
            ),
            chunk_size=max_nb_microbatches,
        ):
            if len(batches) == 0:
                # It means we're out of element
                return

            # Number of micro batches
            number_states_in_buffer = len(batches)
            # Otherwise the pipelining doesn't work
            assert number_states_in_buffer <= max_nb_microbatches
            is_max_nb_microbatches = number_states_in_buffer == max_nb_microbatches

            # Initialize decoder states
            decoder_states: Iterable[GenerationStates] = (
                GenerationStates(
                    new_input_ids=batch.input_ids,
                    new_input_mask=batch.input_masks,
                    store=Store(),
                    generation_ids=[batch.input_ids],
                    generation_mask=[batch.input_masks],
                )
                for batch in batches
            )

            for generation_iter in range(max_new_tokens):
                all_new_decoder_input_ids_and_mask_same_rank: List[
                    Tuple[Union[torch.LongTensor, TensorPointer], Union[torch.BoolTensor, TensorPointer]]
                ] = []
                new_decoder_states: List[GenerationStates] = []
                for state_id, state in enumerate(decoder_states):
                    new_decoder_states.append(state)
                    # Get the new logits
                    with attach_store(model=model, store=state.store):
                        # transpose: [sequence_length, batch_size, vocab_size] -> [batch_size, sequence_length, vocab_size]
                        sharded_logits = model(
                            input_ids=state.new_input_ids,
                            input_mask=state.new_input_mask,
                        )
                        if isinstance(sharded_logits, torch.Tensor):
                            sharded_logits = sharded_logits.transpose(0, 1)

                    # Communicate
                    # TODO @thomasw21: Make a diagram to show how this works
                    nb_send: int = 0
                    if is_decoder_input_rank:
                        if is_max_nb_microbatches:
                            if generation_iter == 0:
                                if state_id == number_states_in_buffer - 1:
                                    # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                    nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                                else:
                                    # Send everything
                                    nb_send = len(pipeline_state.microbatches_activations_to_send)
                            else:
                                # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                        else:
                            if number_states_in_buffer - 1 == state_id or generation_iter == 0:
                                # Send everything
                                nb_send = len(pipeline_state.microbatches_activations_to_send)
                            else:
                                # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                    else:
                        if state_id == number_states_in_buffer - 1:
                            if not is_max_nb_microbatches:
                                nb_send = len(pipeline_state.microbatches_activations_to_send)
                    for _ in range(nb_send):
                        pipeline_state.run_communication()

                    if is_decoder_logit_rank:
                        assert isinstance(sharded_logits, torch.Tensor)

                        # run a logit chooser.
                        if sampler_type == SamplerType.GREEDY:
                            sampler = GreedySampler(pg=parallel_context.tp_pg)
                        elif sampler_type == SamplerType.TOP_K:
                            sampler = TopKSampler(
                                pg=parallel_context.tp_pg,
                                k=generation_config.top_k,
                                temperature=generation_config.temperature,
                            )
                        elif sampler_type == SamplerType.TOP_P:
                            sampler = TopPSampler(
                                pg=parallel_context.tp_pg,
                                p=generation_config.top_p,
                                temperature=generation_config.temperature,
                            )
                        elif sampler_type == SamplerType.BASIC:
                            sampler = BasicSampler(pg=parallel_context.tp_pg)
                        else:
                            raise NotImplementedError(f"Sampler type {sampler_type} is not implemented")

                        new_decoder_input_ids = sampler(sharded_logits=sharded_logits[:, -1, :])

                        # TODO @thomasw21: Handle this correctly, ie from some point after <eos> this should only generate masked tokens
                        # TODO @thomasw21: Actually I can probably build this thing on the next device directly. Will save some communication
                        new_decoder_input_mask = torch.ones(
                            size=(new_decoder_input_ids.shape[0], 1),
                            dtype=torch.bool,
                            device=new_decoder_input_ids.device,
                        )

                        # TODO @thomasw21: We need to have stop condition.

                        # broadcast new_tokens to everyone
                        if decoder_input_rank == decoder_logit_rank:
                            # It's the same rank so no need to do anything too fancy
                            all_new_decoder_input_ids_and_mask_same_rank.append(
                                (new_decoder_input_ids, new_decoder_input_mask)
                            )
                        else:
                            pipeline_state.register_send_activation(
                                new_decoder_input_ids, to_rank=decoder_input_rank, p2p=p2p
                            )
                            pipeline_state.register_send_activation(
                                new_decoder_input_mask, to_rank=decoder_input_rank, p2p=p2p
                            )
                            if not is_max_nb_microbatches and state_id == number_states_in_buffer - 1:
                                # Send new_decoder_input_ids AND new_decoder_input_ids
                                pipeline_state.run_communication()
                                pipeline_state.run_communication()

                    else:
                        assert isinstance(sharded_logits, TensorPointer)

                all_new_decoder_input_ids_and_mask: Iterable[
                    Tuple[Union[torch.LongTensor, TensorPointer], Union[torch.BoolTensor, TensorPointer]]
                ]
                if is_decoder_input_rank:
                    # We receive the tensor from other ranks unless `decoder_input_rank` == `decoder_logit_rank` in which case `all_new_decoder_input_ids` is already populated.
                    if decoder_input_rank == decoder_logit_rank:
                        # `all_new_decoder_input_ids_and_mask_same_rank` is already populated. Since `decoder_input_rank` and `decoder_logit_rank` are the same, there's no need to communicate as we can just store the new input_ids in a list.
                        assert len(all_new_decoder_input_ids_and_mask_same_rank) == number_states_in_buffer
                        all_new_decoder_input_ids_and_mask = all_new_decoder_input_ids_and_mask_same_rank
                    else:

                        def generator():
                            for _ in range(number_states_in_buffer):
                                pipeline_state.register_recv_activation(from_rank=decoder_logit_rank, p2p=p2p)
                                pipeline_state.register_recv_activation(from_rank=decoder_logit_rank, p2p=p2p)
                                while len(pipeline_state.activations_buffer) < 2:
                                    pipeline_state.run_communication()
                                new_decoder_input_ids = pipeline_state.activations_buffer.popleft()
                                new_decoder_input_mask = pipeline_state.activations_buffer.popleft()
                                yield new_decoder_input_ids, new_decoder_input_mask

                        all_new_decoder_input_ids_and_mask = iter(generator())
                else:
                    all_new_decoder_input_ids_and_mask = (
                        (TensorPointer(group_rank=decoder_input_rank), TensorPointer(group_rank=decoder_input_rank))
                        for _ in range(number_states_in_buffer)
                    )

                # Create new decoder states
                decoder_states = (
                    GenerationStates(
                        new_input_ids=new_decoder_input_ids_and_mask[0],
                        new_input_mask=new_decoder_input_ids_and_mask[1],
                        store=state.store,
                        generation_ids=state.generation_ids + [new_decoder_input_ids_and_mask[0]],
                        generation_mask=state.generation_mask + [new_decoder_input_ids_and_mask[1]],
                    )
                    for state, new_decoder_input_ids_and_mask in zip(
                        new_decoder_states, all_new_decoder_input_ids_and_mask
                    )
                )

            # Flush communication
            for _ in range(
                max(
                    len(pipeline_state.microbatches_activations_to_send),
                    len(pipeline_state.microbatches_activations_to_recv),
                )
            ):
                pipeline_state.run_communication()
            assert len(pipeline_state.microbatches_activations_to_send) == 0
            assert len(pipeline_state.microbatches_activations_to_recv) == 0

            # Yield result
            decoder_states = list(decoder_states)
            for state, batch in zip(decoder_states, batches):
                if is_decoder_input_rank:
                    assert all(isinstance(elt, torch.Tensor) for elt in state.generation_ids)
                    batch_generated_ids = torch.cat(state.generation_ids, dim=-1)
                    batch_generated_mask = torch.cat(state.generation_mask, dim=-1)
                else:
                    assert all(isinstance(elt, TensorPointer) for elt in state.generation_ids)
                    batch_generated_ids = TensorPointer(group_rank=decoder_input_rank)
                    batch_generated_mask = TensorPointer(group_rank=decoder_input_rank)

                # Broadcast all data
                batch_generated_ids, batch_generated_mask = broadcast_tensors(
                    [batch_generated_ids, batch_generated_mask],
                    group_src=decoder_input_rank,
                    group=parallel_context.pp_pg,
                )
                batch.input_ids, batch.input_masks = broadcast_tensors(
                    [batch.input_ids, batch.input_masks], group_src=decoder_input_rank, group=parallel_context.pp_pg
                )

                # Flush the store to release memory
                state.store.flush()
                assert len(state.store) == 0

                if dist.get_rank(parallel_context.pp_pg) == decoder_input_rank:
                    assert (
                        batch_generated_ids.shape[0] == batch.input_ids.shape[0]
                    ), f"Batch size needs to match {batch_generated_ids.shape[0]} != {batch.input_ids.shape[0]}"
                    assert (
                        batch_generated_mask.shape[0] == batch.input_ids.shape[0]
                    ), f"Batch size needs to match {batch_generated_mask.shape[0]} != {batch.input_ids.shape[0]}"
                    assert (
                        batch_generated_ids.shape[1] == batch_generated_mask.shape[1]
                    ), f"Sequence length needs to match {batch_generated_ids.shape[1]} != {batch_generated_mask.shape[0]}"

                for i, (generated_ids, generated_mask) in enumerate(zip(batch_generated_ids, batch_generated_mask)):
                    # TODO @thomasw21: We could actually have all ranks return the output, since it's been already broadcasted
                    if dist.get_rank(parallel_context.pp_pg) == decoder_input_rank:
                        input_ids = batch.input_ids[i]
                        input_mask = batch.input_masks[i]
                        yield GenerationOutput(
                            input_ids=input_ids[input_mask],
                            generation_ids=generated_ids[generated_mask],
                        )
                    else:
                        yield GenerationOutput(
                            input_ids=TensorPointer(group_rank=decoder_input_rank),
                            generation_ids=TensorPointer(group_rank=decoder_input_rank),
                        )


# Distributed utilities
def broadcast_tensors(
    tensors: List[Union[torch.Tensor, TensorPointer]], group_src: int, group: Optional[ProcessGroup] = None
) -> List[torch.Tensor]:
    result = []
    for tensor in tensors:
        if dist.get_rank(group) == group_src:
            assert isinstance(tensor, torch.Tensor)
            meta = [
                [
                    tensor.dtype,
                    tensor.requires_grad,
                    tensor.shape,
                    get_untyped_storage(tensor).size(),
                    tensor.stride(),
                    tensor.is_contiguous(),
                    tensor.storage_offset(),
                ]
            ]
        else:
            assert isinstance(tensor, TensorPointer)
            meta = [None]
        dist.broadcast_object_list(meta, src=get_global_rank(group_rank=group_src, group=group), group=group)
        dtype, requires_grad, shape, untyped_storage_size, stride, is_contiguous, storage_offset = meta[0]
        meta = P2PTensorMetaData(
            dtype=dtype,
            requires_grad=requires_grad,
            shape=shape,
            untyped_storage_size=untyped_storage_size,
            stride=stride,
            is_contiguous=is_contiguous,
            storage_offset=storage_offset,
        )
        if dist.get_rank(group) != group_src:
            tensor = meta.create_empty_storage(device=torch.device("cuda"))
        else:
            tensor = view_as_contiguous(tensor)
        dist.broadcast(tensor, src=get_global_rank(group_rank=group_src, group=group), group=group)
        # Set shape and stride
        tensor = tensor.as_strided(size=tuple(meta.shape), stride=tuple(meta.stride))
        result.append(tensor)
    return result
