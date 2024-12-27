import collections
import dataclasses
from abc import ABC, abstractmethod
from typing import List

import torch

from nanotron import distributed as dist
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.p2p import P2P, P2PCommunicationError

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class SendActivation:
    activation: torch.Tensor
    to_rank: int
    p2p: P2P

    def __call__(self):
        self.p2p.send_tensors([self.activation], to_rank=self.to_rank)


@dataclasses.dataclass
class RecvActivation:
    from_rank: int
    p2p: P2P

    def __call__(self) -> torch.Tensor:
        return self.p2p.recv_tensors(num_tensors=1, from_rank=self.from_rank)[0]


@dataclasses.dataclass
class SendGrad:
    grad: torch.Tensor
    to_rank: int
    p2p: P2P

    def __call__(self):
        """Send gradient and wait for acknowledgment"""
        current_rank = dist.get_rank(self.p2p.pg)
        grad_id = self.p2p.grad_id_counter
        self.p2p.grad_id_counter += 1

        logger.debug(
            f"[SEND GRAD] Rank {current_rank}: Sending grad (id={grad_id}) to rank {self.to_rank} "
            f"shape={self.grad.shape}, dtype={self.grad.dtype}"
        )

        # Send the gradient using existing method
        self.p2p.send_tensors([self.grad], to_rank=self.to_rank)
        logger.debug(f"[SEND GRAD] Rank {current_rank}: Completed grad send for grad_id {grad_id}")

        # # Wait for acknowledgment
        # if not self.p2p.wait_for_ack(self.to_rank, grad_id):
        #     logger.error(
        #         f"[SEND GRAD] Rank {current_rank}: Failed to receive ack for grad_id {grad_id} "
        #         f"from rank {self.to_rank}"
        #     )
        #     raise P2PCommunicationError(f"No acknowledgment received for grad_id {grad_id} from rank {self.to_rank}")

        # logger.debug(f"[SEND GRAD] Rank {current_rank}: Completed grad send and ack for grad_id {grad_id}")


@dataclasses.dataclass
class RecvGrad:
    from_rank: int
    p2p: P2P

    def __call__(self) -> torch.Tensor:
        """Receive gradient and send acknowledgment"""
        current_rank = dist.get_rank(self.p2p.pg)
        grad_id = self.p2p.grad_id_counter

        logger.debug(f"[RECV GRAD] Rank {current_rank}: Starting grad receive from rank {self.from_rank}")

        # Receive the gradient using existing method
        grad = self.p2p.recv_tensors(num_tensors=1, from_rank=self.from_rank)[0]

        logger.debug(
            f"[RECV GRAD] Rank {current_rank}: Received grad from rank {self.from_rank} "
            f"shape={grad.shape}, dtype={grad.dtype}"
        )
        logger.debug(f"[RECV GRAD] Rank {current_rank}: Completed grad receive for grad_id {grad_id}")

        # Send acknowledgment
        # self.p2p.send_ack(self.from_rank, grad_id)

        # logger.debug(f"[RECV GRAD] Rank {current_rank}: Completed grad receive and ack send for grad_id {grad_id}")
        return grad


class PipelineBatchState(ABC):
    activations_buffer = collections.deque()

    @abstractmethod
    def register_activation_requiring_backward(self, activation: torch.Tensor):
        ...

    @abstractmethod
    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        ...

    @abstractmethod
    def register_recv_activation(self, from_rank: int, p2p: P2P):
        ...

    @abstractmethod
    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        ...

    @abstractmethod
    def register_recv_grad(self, from_rank: int, p2p: P2P):
        ...

    @abstractmethod
    def run_communication(self, send_only_activation: bool = False):
        ...

    @abstractmethod
    def new_micro_batch_forward(self):
        ...

    @abstractmethod
    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        ...


@dataclasses.dataclass
class PipelineTrainBatchState(PipelineBatchState):
    microbatches_activations_to_send = collections.deque()
    microbatches_activations_to_recv = collections.deque()
    microbatches_grads_to_send = collections.deque()
    microbatches_grads_to_recv = collections.deque()
    grads_buffer = collections.deque()

    # List of list, first index represent micro_batch_id, second index represent activations that needs to be popped
    microbatches_activations_requiring_backward = collections.deque()

    # Reinitialise counter
    nb_backwards = 0
    nb_forwards = 0
    # F1 B1 F2 B2 F3 B3 F4
    str_history = []

    def register_activation_requiring_backward(self, activation: torch.Tensor):
        # Register the activation to last microbatch
        self.microbatches_activations_requiring_backward[-1].append(activation)

    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        # TODO @thomasw21: We assume that each rank has a single contiguous list of blocks. This also means that we only send activations from higher ranks
        log_rank(
            f"Registering send activation to rank {to_rank}",
            logger=logger,
            level=logging.DEBUG,
        )
        self.microbatches_activations_to_send.append(SendActivation(activation=activation, to_rank=to_rank, p2p=p2p))

    def register_recv_activation(self, from_rank: int, p2p: P2P):
        # TODO @thomasw21: We assume that each rank has a single contiguous list of blocks. This also means that we only recv activations from lower ranks
        log_rank(
            f"Registering receive activation from rank {from_rank}",
            logger=logger,
            level=logging.DEBUG,
        )
        self.microbatches_activations_to_recv.append(RecvActivation(from_rank=from_rank, p2p=p2p))

    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        # TODO @thomasw21: We assume that each rank has a single contiguous list of blocks. This also means that we only send gradients to lower ranks
        log_rank(
            f"Registering send gradient to rank {to_rank}",
            logger=logger,
            level=logging.DEBUG,
        )
        self.microbatches_grads_to_send.append(SendGrad(grad=grad, to_rank=to_rank, p2p=p2p))

    def register_recv_grad(self, from_rank: int, p2p: P2P):
        # TODO @thomasw21: We assume that each rank has a single contiguous list of blocks. This also means that we only recv gradients from higher ranks
        log_rank(
            f"Registering receive gradient from rank {from_rank}",
            logger=logger,
            level=logging.DEBUG,
        )
        self.microbatches_grads_to_recv.append(RecvGrad(from_rank=from_rank, p2p=p2p))

    def run_communication(self, send_only_activation: bool = False):
        """Run communication in a specific order: send activation, recv activation, send grad, recv grad.
        Retries failed communications up to 5 times."""
        log_rank(
            f"before run_comm: activation_to_send: {len(self.microbatches_activations_to_send)} | "
            f"activation_to_recv: {len(self.microbatches_activations_to_recv)} | "
            f"grads_to_send: {len(self.microbatches_grads_to_send)} | "
            f"grads_to_recv: {len(self.microbatches_grads_to_recv)} | "
            f"activation_buffer: {len(self.activations_buffer)} | "
            f"grads_buffer: {len(self.grads_buffer)}",
            logger=logger,
            level=logging.DEBUG,
        )

        try:
            num_retries = 5
            retry_count = 0

            while retry_count < num_retries:
                made_progress = False

                # Pop one of each type of operation
                send_activation = None
                recv_activation = None
                send_grad = None
                recv_grad = None
                activation_send_requires_grad = False

                if len(self.microbatches_activations_to_send) > 0:
                    send_activation = self.microbatches_activations_to_send.popleft()
                if not send_only_activation:
                    if len(self.microbatches_activations_to_recv) > 0:
                        recv_activation = self.microbatches_activations_to_recv.popleft()
                    if len(self.microbatches_grads_to_send) > 0:
                        send_grad = self.microbatches_grads_to_send.popleft()
                    if len(self.microbatches_grads_to_recv) > 0:
                        recv_grad = self.microbatches_grads_to_recv.popleft()

                # If nothing to do, break
                if not any([send_activation, recv_activation, send_grad, recv_grad]):
                    break

                # Try to execute each operation
                if send_activation:
                    try:
                        activation_send_requires_grad = send_activation.activation.requires_grad
                        send_activation()
                        made_progress = True
                        if send_only_activation:
                            return
                    except P2PCommunicationError as e:
                        self.microbatches_activations_to_send.appendleft(send_activation)
                        logger.warning(f"send_activation failed, will retry later. Error: {str(e)}")

                if recv_activation:
                    try:
                        recv_activation_tensor = recv_activation()
                        self.activations_buffer.append(recv_activation_tensor)
                        made_progress = True
                        if recv_activation_tensor.requires_grad is False:
                            if send_grad:
                                self.microbatches_grads_to_send.appendleft(send_grad)
                            if recv_grad:
                                self.microbatches_grads_to_recv.appendleft(recv_grad)
                            return
                    except P2PCommunicationError as e:
                        self.microbatches_activations_to_recv.appendleft(recv_activation)
                        logger.warning(f"recv_activation failed, will retry later. Error: {str(e)}")

                if send_grad:
                    try:
                        send_grad()
                        made_progress = True
                    except P2PCommunicationError as e:
                        self.microbatches_grads_to_send.appendleft(send_grad)
                        logger.warning(f"send_grad failed, will retry later. Error: {str(e)}")

                if recv_grad:
                    try:
                        # First try to send any remaining activations that require grad
                        while len(self.microbatches_activations_to_send) > 0 and not activation_send_requires_grad:
                            try:
                                next_send_activation = self.microbatches_activations_to_send.popleft()
                                activation_send_requires_grad = next_send_activation.activation.requires_grad
                                next_send_activation()
                                made_progress = True
                            except P2PCommunicationError as e:
                                self.microbatches_activations_to_send.appendleft(next_send_activation)
                                logger.warning(f"additional send_activation failed, will retry later. Error: {str(e)}")
                                break

                        # Now try to receive the gradient
                        grad_tensor = recv_grad()
                        self.grads_buffer.append(grad_tensor)
                        made_progress = True
                    except P2PCommunicationError as e:
                        self.microbatches_grads_to_recv.appendleft(recv_grad)
                        logger.warning(f"recv_grad failed, will retry later. Error: {str(e)}")

                # If no progress was made, increment retry counter
                if not made_progress:
                    retry_count += 1
                    logger.warning(f"No progress made in communication iteration, attempt {retry_count}/{num_retries}")
                    if retry_count >= num_retries:
                        logger.error("Max retries reached, giving up on communication")
                        break

        finally:
            log_rank(
                f"after run_comm: activation_to_send: {len(self.microbatches_activations_to_send)} | "
                f"activation_to_recv: {len(self.microbatches_activations_to_recv)} | "
                f"grads_to_send: {len(self.microbatches_grads_to_send)} | "
                f"grads_to_recv: {len(self.microbatches_grads_to_recv)} | "
                f"activation_buffer: {len(self.activations_buffer)} | "
                f"grads_buffer: {len(self.grads_buffer)}",
                logger=logger,
                level=logging.DEBUG,
            )
        # TODO @thomasw21: I need some mechanism to point to whatever is now sorted in a buffer, typically some id that would point to the correct tensor in our buffer instead of relying on the sorted list.

    def new_micro_batch_forward(self):
        self.microbatches_activations_requiring_backward.append(collections.deque())

    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        return self.microbatches_activations_requiring_backward.popleft()

    def check_buffers_empty(self):
        assert (
            len(self.microbatches_activations_requiring_backward) == 0
        ), f"There are still activations that require backward: {len(self.microbatches_activations_requiring_backward)}"
        assert (
            len(self.microbatches_activations_to_send) == 0
        ), f"There are activations left for me to send still: {len(self.microbatches_activations_to_send)}"
        assert (
            len(self.microbatches_activations_to_recv) == 0
        ), f"There are activations left for me to recv still: {len(self.microbatches_activations_to_recv)}"
        assert (
            len(self.microbatches_grads_to_send) == 0
        ), f"There are gradients left for me to send still: {len(self.microbatches_grads_to_send)}"
        assert (
            len(self.microbatches_grads_to_recv) == 0
        ), f"There are gradients left for me to recv still: {len(self.microbatches_grads_to_recv)}"


@dataclasses.dataclass
class PipelineEvalBatchState(PipelineBatchState):
    microbatches_activations_to_send = collections.deque()
    microbatches_activations_to_recv = collections.deque()
    activations_buffer = collections.deque()

    def register_activation_requiring_backward(self, activation: torch.Tensor):
        pass

    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        self.microbatches_activations_to_send.append(SendActivation(activation=activation, to_rank=to_rank, p2p=p2p))

        # There's a cross communication
        if len(self.microbatches_activations_to_recv) > 0 and len(self.microbatches_activations_to_recv) > 0:
            self.run_communication()

    def register_recv_activation(self, from_rank: int, p2p: P2P):
        self.microbatches_activations_to_recv.append(RecvActivation(from_rank=from_rank, p2p=p2p))

        # There's a cross communication
        if len(self.microbatches_activations_to_recv) > 0 and len(self.microbatches_activations_to_recv) > 0:
            self.run_communication()

    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        raise NotImplementedError("You can't register a send grad in pipeline eval mode")

    def register_recv_grad(self, from_rank: int, p2p: P2P):
        raise NotImplementedError("You can't register a recv grad in pipeline eval mode")

    def new_micro_batch_forward(self):
        pass

    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        pass

    def run_communication(self, send_only_activation: bool = False):
        # four cases:
        #  - you receive from higher rank and you send to higher rank
        #  - You receive from higher rank and you send to lower rank
        #  - you receive from lower rank and you send to higher rank
        #  - you receive from lower rank and you send to lower rank

        send_activation = None
        # Pop all send activation
        for _ in range(min(1, len(self.microbatches_activations_to_send))):
            send_activation = self.microbatches_activations_to_send.popleft()

        # Pop all recv activation
        recv_activation = None
        for _ in range(min(1, len(self.microbatches_activations_to_recv))):
            recv_activation = self.microbatches_activations_to_recv.popleft()

        if send_activation is None:
            if recv_activation is None:
                raise ValueError("Why the hell do we communicate when there's nothing to communicate?")
            self.activations_buffer.append(recv_activation())
        else:
            if recv_activation is None:
                send_activation()
            else:
                # Define in which order to we do it.
                # Actually we can't do any heuristics as you need global information in order to define clear ordering.
                # We make a BIG assumption that only ONE rank receives from higher rank and sends to higher rank.
                # In this case we find the "lowest" rank, send first
                # All the other ranks receive first and send after
                # Lowest rank receives.
                # If we knew who was involved in the cycle, we could just randomly choose one rank to first send then recv, however it's not clear who's involved
                p2p = send_activation.p2p
                assert p2p == recv_activation.p2p
                is_lowest = send_activation.to_rank > dist.get_rank(
                    p2p.pg
                ) and recv_activation.from_rank > dist.get_rank(p2p.pg)
                if is_lowest:
                    send_activation()
                    self.activations_buffer.append(recv_activation())
                else:
                    self.activations_buffer.append(recv_activation())
                    send_activation()

    def check_buffers_empty(self):
        assert (
            len(self.microbatches_activations_to_send) == 0
        ), f"There are activations left for me to send still: {len(self.microbatches_activations_to_send)}"
        assert (
            len(self.microbatches_activations_to_recv) == 0
        ), f"There are activations left for me to recv still: {len(self.microbatches_activations_to_recv)}"
        assert (
            len(self.activations_buffer) == 0
        ), f"There are activations left in the buffer: {len(self.activations_buffer)}"
