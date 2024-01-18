from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Union

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import log_rank
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.data_parallel.utils import ddp_trigger_sync_in_bwd
from nanotron.parallel.pipeline_parallel.context_manager import attach_pipeline_state_to_model
from nanotron.parallel.pipeline_parallel.state import PipelineTrainBatchState
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.utils import ContextManagers
from torch import nn as torch_nn
from torch.nn.parallel import DistributedDataParallel

logger = logging.get_logger(__name__)


class PipelineEngine(ABC):
    def __init__(self):
        self.nb_microbatches: Optional[int] = None
        pass

    def forward(
        self,
        context: ContextManagers,
        state: PipelineTrainBatchState,
        micro_batch: Dict[str, Union[torch.Tensor, TensorPointer]],
        model: torch_nn.Module,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Increment the number of backwards
        state.nb_forwards += 1
        log_rank(
            f"Forward micro batch id: {state.nb_forwards}",
            logger=logger,
            level=logging.DEBUG,
        )

        # IMPORTANT as it's basically the context manager storing all the intermediary activations
        state.new_micro_batch_forward()
        with context:
            output = model(**micro_batch)

        # We make `output` a dict
        if not isinstance(output, dict):
            output = {"loss": output}

        # We normalize our loss
        if not isinstance(output["loss"], TensorPointer):
            output["loss"] = output["loss"] / self.nb_microbatches

        # Add output as activations that require backward pass
        if not isinstance(output["loss"], TensorPointer):
            assert output["loss"].requires_grad
            state.register_activation_requiring_backward(output["loss"])
        return output

    @staticmethod
    def _get_fwd_context(model: torch_nn.Module):
        is_ddp = isinstance(model, DistributedDataParallel)
        # We never to trigger a DDP sync in the next backward pass
        context = ContextManagers([model.no_sync()] if is_ddp else [])
        return context

    def backward(
        self, context: ContextManagers, state: PipelineTrainBatchState, grad_accumulator: Optional[GradientAccumulator]
    ):
        # Increment the number of backwards
        state.nb_backwards += 1
        log_rank(
            f"Backward micro batch id: {state.nb_forwards}",
            logger=logger,
            level=logging.DEBUG,
        )
        # Go backward entirely
        activations = state.pop_last_activations_requiring_backward()
        if len(activations) == 0:
            return

        with context:
            if grad_accumulator is None:
                sum(activations).backward()
            else:
                grad_accumulator.backward(sum(activations))

        # TODO @nouamane: this fixes interleaved afab but makes 1f1b hang
        # with context:
        #     if grad_accumulator is None:
        #         for activation in reversed(activations): #TODO @nouamane: need to bwd only 2nd chunk
        #             activation.backward()
        #     else:
        #         for activation in reversed(activations):
        #             grad_accumulator.backward(activation)

    def _get_bwd_context(
        self,
        model: torch_nn.Module,
        nb_backwards: int,
        grad_accumulator: Optional[GradientAccumulator],
    ):
        assert (
            self.nb_microbatches is not None
        ), "You must call `train_batch_iter` first and set `self.nb_microbatches`"
        is_ddp = isinstance(model, DistributedDataParallel)
        context_list = []
        if is_ddp:
            if grad_accumulator is not None and nb_backwards < self.nb_microbatches - 1:
                context_list.append(grad_accumulator.no_sync())  # Prevents grad accumulator from syncing
            if nb_backwards == self.nb_microbatches - 1:
                # Triggers DDP to sync gradients in the next backward pass
                context_list.append(ddp_trigger_sync_in_bwd(model_ddp=model))
        context = ContextManagers(context_list)
        return context

    @abstractmethod
    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        """If model returns tensor, we use it as a loss to backpropagate. If model returns a dict, we assume that the key "loss" is the loss to backpropagate."""
        ...

    @torch.inference_mode()
    def validate_batch_iter(
        self,
        model: torch_nn.Module,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        # Assign a new state for the current batch
        state = PipelineTrainBatchState()  # TODO: do i need state?
        self.nb_microbatches = nb_microbatches

        outputs = []

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            # All forward
            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)
                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    # Execute
                    send_activation()

                # We make `output` a dict
                if not isinstance(output, dict):
                    output = {"loss": output}

                # Store the loss for each microbatch
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

        return outputs


class AllForwardAllBackwardPipelineEngine(PipelineEngine):
    def __init__(self):
        super().__init__()

    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        # Assign a new state for the current batch
        state = PipelineTrainBatchState()
        self.nb_microbatches = nb_microbatches

        outputs = []

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            # All forward
            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)
                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    # Execute
                    send_activation()

                # We make `output` a dict
                if not isinstance(output, dict):
                    output = {"loss": output}

                # Store the loss for each microbatch
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

            # All backward
            for _ in range(len(state.microbatches_activations_requiring_backward)):
                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

                for _ in range(len(state.microbatches_grads_to_send)):
                    send_grads = state.microbatches_grads_to_send.popleft()
                    # Execute
                    send_grads()
            # Make sure that micro batches are all fully consumed
            state.check_buffers_empty()

            return outputs


class OneForwardOneBackwardPipelineEngine(PipelineEngine):
    def __init__(self):
        super().__init__()

    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        """Check https://arxiv.org/abs/2104.04473 for diagrams for the pipeline engine"""
        self.nb_microbatches = nb_microbatches
        assert (
            self.nb_microbatches >= pg.size() - 1
        ), f"Number of microbatches ({self.nb_microbatches}) must be at least PP_SIZE-1={pg.size() - 1} when using the OneForwardOneBackwardPipelineEngine"

        state = PipelineTrainBatchState()

        outputs = []
        batch = iter(batch)

        current_pp_rank = dist.get_rank(pg)

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            # Init
            for _ in range(pg.size() - current_pp_rank - 1):
                micro_batch = next(batch)
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)

                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    # Execute
                    send_activation()

                # We make `output` a dict
                if not isinstance(output, dict):
                    output = {"loss": output}

                # Send tensors
                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    # Execute
                    send_activation()

                # Store the loss for each microbatch
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)

                # We make `output` a dict
                if not isinstance(output, dict):
                    output = {"loss": output}

                # Store the loss for each microbatch
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

                # One backward
                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

            # Check figure in paper: The remain blocks are all backward and there is only `pg.size() - current_pp_rank - 1` blocks left
            assert len(state.microbatches_activations_requiring_backward) == pg.size() - current_pp_rank - 1
            # No more activation to send/recv
            assert (
                len(state.microbatches_activations_to_send) == 0
            ), f"There are activations left for me to send still: {len(state.microbatches_activations_to_send)}"
            assert (
                len(state.microbatches_activations_to_recv) == 0
            ), f"There are activations left for me to recv still: {len(state.microbatches_activations_to_recv)}"

            # Close: compute backward for the rest
            # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
            for _ in range(len(state.microbatches_grads_to_send)):
                send_grads = state.microbatches_grads_to_send.popleft()
                # Execute
                send_grads()
            for _ in range(len(state.microbatches_activations_requiring_backward)):
                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_grads_to_send)):
                    send_grads = state.microbatches_grads_to_send.popleft()
                    # Execute
                    send_grads()

            # Make sure that micro batches are all fully consumed
            state.check_buffers_empty()

        return outputs
