import dataclasses
import os
import time
from enum import Enum
from typing import List, Sequence, Tuple

import torch

from nanotron import distributed as dist
from nanotron import logging
from nanotron.utils import get_untyped_storage, tensor_from_untyped_storage

logger = logging.get_logger(__name__)

FIRST_METADATA_SIZE = 7
SECOND_METADATA_SIZE = 1024
ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]
DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

ID_TO_REQUIRES_GRAD = [True, False]
REQUIRES_GRAD_TO_ID = {value: id_ for id_, value in enumerate(ID_TO_REQUIRES_GRAD)}
ID_TO_IS_CONTIGUOUS = [True, False]
IS_CONTIGUOUS_TO_ID = {value: id_ for id_, value in enumerate(ID_TO_IS_CONTIGUOUS)}

COMMS_TIMEOUT = 60 * 20  # 20 minutes timeout


class P2PCommunicationError(Exception):
    """Raised when P2P communication fails"""

    pass


@dataclasses.dataclass
class P2PTensorMetaData:
    shape: Sequence[int]
    stride: Sequence[int]
    is_contiguous: bool
    untyped_storage_size: int
    storage_offset: int
    dtype: torch.dtype
    requires_grad: bool

    def create_empty_storage(self, device: torch.device) -> torch.Tensor:
        buffer = torch.empty(
            size=(self.untyped_storage_size,),
            requires_grad=False,
            dtype=torch.int8,
            device=device,
            memory_format=torch.contiguous_format,
        ).view(dtype=self.dtype)
        buffer.requires_grad = self.requires_grad

        if self.is_contiguous:
            buffer = buffer.as_strided(
                size=tuple(self.shape), stride=tuple(self.stride), storage_offset=self.storage_offset
            )

        # Complex needs to be viewed as real first
        # TODO @thomasw21: Find the issue with send/recv complex tensors
        buffer = torch.view_as_real(buffer) if self.dtype.is_complex else buffer

        return buffer

    def reshape(self, buffer):
        """Changes the way we view buffer in order to fit metadata"""
        # TODO @thomasw21: Find the issue with send/recv complex tensors
        buffer = torch.view_as_complex(buffer) if self.dtype.is_complex else buffer

        # Set shape and stride
        if not self.is_contiguous:
            buffer = buffer.as_strided(
                size=tuple(self.shape), stride=tuple(self.stride), storage_offset=self.storage_offset
            )

        return buffer

    @staticmethod
    def to_first_metadata(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        # TODO @nouamane: avoid having two metadata comms, and preallocate shape/stride instead
        return torch.tensor(
            [
                len(tensor.shape),
                len(tensor.stride()),
                IS_CONTIGUOUS_TO_ID[tensor.is_contiguous()],
                get_untyped_storage(tensor).size(),
                tensor.storage_offset(),
                DTYPE_TO_ID[tensor.dtype],
                REQUIRES_GRAD_TO_ID[tensor.requires_grad],
            ],
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def to_second_metadata(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.tensor(tensor.shape + tensor.stride(), dtype=torch.long, device=device)

    @classmethod
    def from_metadata(cls, first_metadata: List[int], second_metadata: List[int]):
        shape_and_stride = second_metadata
        (
            num_shape,
            num_stride,
            is_contiguous,
            untyped_storage_size,
            storage_offset,
            dtype_id,
            requires_grad_id,
        ) = first_metadata
        return cls(
            shape=shape_and_stride[: len(shape_and_stride) // 2],
            stride=shape_and_stride[len(shape_and_stride) // 2 :],
            is_contiguous=ID_TO_IS_CONTIGUOUS[is_contiguous],
            untyped_storage_size=untyped_storage_size,
            storage_offset=storage_offset,
            dtype=ID_TO_DTYPE[dtype_id],
            requires_grad=ID_TO_REQUIRES_GRAD[requires_grad_id],
        )


def view_as_contiguous(tensor: torch.Tensor):
    """Given a tensor, we want to view the tensor as a contiguous storage"""
    tensor_numel = tensor.numel()
    tensor_element_size = tensor.element_size()
    untyped_storage = get_untyped_storage(tensor)
    untyped_storage_size = untyped_storage.size()
    untyped_element_size = untyped_storage.element_size()
    assert (
        tensor_numel * tensor_element_size >= untyped_storage_size * untyped_element_size
    ), "Expect storage_size to be smaller than tensor size. It might not be true, when you use slicing for example though. We probably don't want to support it in our P2P system"
    buffer = tensor_from_untyped_storage(untyped_storage=untyped_storage, dtype=tensor.dtype)
    return buffer


class MessageType(Enum):
    GRAD = 0
    ACK = 1


class P2P:
    def __init__(self, pg: dist.ProcessGroup, device: torch.device):
        super().__init__()
        self.pg = pg
        self.device = device

        # Add history tracking
        self.history = []
        self.send_counter = 0
        self.recv_counter = 0

        # Log communication configuration
        self.p2p_enabled = os.environ.get("NCCL_P2P_DISABLE", "0") != "1"
        logger.info(
            f"[INIT] P2P Communication: {'enabled' if self.p2p_enabled else 'disabled'}, "
            f"Device: {device}, Group: {pg}"
        )

        self.first_metadata = torch.empty(FIRST_METADATA_SIZE, dtype=torch.long, device=self.device)
        self.second_metadata = torch.empty(SECOND_METADATA_SIZE, dtype=torch.long, device=self.device)
        self.ack_metadata = torch.empty(2, dtype=torch.long, device=self.device)  # [message_type, grad_id]
        self.grad_id_counter = 0

    def _record_operation(self, op_type: str):
        """Record a send or receive operation in history"""
        if op_type == "S":
            self.send_counter += 1
            counter = self.send_counter
        elif op_type == "R":
            self.recv_counter += 1
            counter = self.recv_counter
        self.history.append(f"{op_type}{counter}")
        logger.debug(f"P2P History: {' '.join(self.history)}")

    def _send_first_metadata_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        first_metadata = P2PTensorMetaData.to_first_metadata(tensor=tensor, device=self.device)
        raise Exception("Experimental: Don't use this")
        self._record_operation("S")
        return dist.P2POp(
            op=dist.isend,
            tensor=first_metadata,
            peer=to_rank,
            group=self.pg,
            tag=tag,
        )

    def _recv_first_metadata_p2p_op(self, from_rank: int, tag: int = 0) -> Tuple[torch.Tensor, dist.P2POp]:
        self._record_operation("R")
        raise Exception("Experimental: Don't use this")
        return self.first_metadata, dist.P2POp(
            op=dist.irecv,
            tensor=self.first_metadata,
            peer=from_rank,
            group=self.pg,
            tag=tag,
        )

    def _send_second_metadata_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        second_metadata = P2PTensorMetaData.to_second_metadata(tensor=tensor, device=self.device)
        self._record_operation("S")
        raise Exception("Experimental: Don't use this")
        return dist.P2POp(
            op=dist.isend,
            tensor=second_metadata,
            peer=to_rank,
            group=self.pg,
            tag=tag,
        )

    def _recv_second_metadata_p2p_op(
        self, shape_length: int, stride_length: int, from_rank: int, tag: int = 0
    ) -> Tuple[torch.Tensor, dist.P2POp]:
        self._record_operation("R")
        raise Exception("Experimental: Don't use this")
        return self.second_metadata, dist.P2POp(
            op=dist.irecv,
            tensor=self.second_metadata,
            peer=from_rank,
            group=self.pg,
            tag=tag,
        )

    def _send_data_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        self._record_operation("S")
        raise Exception("Experimental: Don't use this")
        return dist.P2POp(
            op=dist.isend,
            tensor=tensor,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            # group=self.pg,
            # tag=tag,
        )

    def _recv_data_p2p_op(
        self, tensor_metadata: P2PTensorMetaData, from_rank: int, tag: int = 0
    ) -> Tuple[torch.Tensor, dist.P2POp]:
        tensor_buffer = tensor_metadata.create_empty_storage(self.device)
        raise Exception("Experimental: Don't use this")
        self._record_operation("R")
        return tensor_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=tensor_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            # group=self.pg,
            # tag=tag,
        )

    def _send_meta(self, tensor: torch.Tensor, to_rank: int, tag: int, timeout: float = COMMS_TIMEOUT):
        """Send tensor metadata with timeout"""
        current_rank = dist.get_rank(self.pg)
        start_time = time.time()

        logger.debug(f"[SEND META] Rank {current_rank}: Starting metadata send to {to_rank} ")

        try:
            # Send first metadata
            cpu_tensor = torch.tensor(
                [
                    len(tensor.shape),
                    len(tensor.stride()),
                    IS_CONTIGUOUS_TO_ID[tensor.is_contiguous()],
                    get_untyped_storage(tensor).size(),
                    tensor.storage_offset(),
                    DTYPE_TO_ID[tensor.dtype],
                    REQUIRES_GRAD_TO_ID[tensor.requires_grad],
                ],
                dtype=torch.long,
            )
            self.first_metadata.copy_(cpu_tensor)

            # Send with timeout check
            self._record_operation("S")
            work = dist.isend(
                self.first_metadata,
                dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
                # group=self.pg,
                # tag=tag,
            )

            while not work.is_completed():
                if time.time() - start_time > timeout:
                    logger.warning(
                        f"[SEND META] Rank {current_rank}: First metadata send to rank {to_rank} "
                        f"timed out after {timeout}s"
                    )
                    work.abort()
                    raise P2PCommunicationError(f"First metadata send timed out to rank {to_rank}")
                time.sleep(0.1)

            elapsed = time.time() - start_time
            logger.debug(
                f"[SEND META] Rank {current_rank}: First metadata sent in {elapsed:.3f}s " f"to rank {to_rank}"
            )

            # Prepare and send second metadata
            second_metadata = tensor.shape + tensor.stride()
            assert len(tensor.shape) == self.first_metadata[0]
            assert len(tensor.stride()) == self.first_metadata[1]

            # increase buffer size if needed
            if len(second_metadata) > len(self.second_metadata):
                self.second_metadata = torch.empty(len(second_metadata), dtype=torch.long, device=self.device)

            self.second_metadata[: len(second_metadata)].copy_(torch.tensor(second_metadata, dtype=torch.long))

            # Send second metadata with timeout check
            start_time = time.time()
            self._record_operation("S")
            work = dist.isend(
                self.second_metadata[: len(second_metadata)],
                dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
                # group=self.pg,
                # tag=tag,
            )

            while not work.is_completed():
                if time.time() - start_time > timeout:
                    logger.warning(
                        f"[SEND META] Rank {current_rank}: Second metadata send to rank {to_rank} "
                        f"timed out after {timeout}s"
                    )
                    work.abort()
                    raise P2PCommunicationError(f"Second metadata send timed out to rank {to_rank}")
                time.sleep(0.1)

            total_elapsed = time.time() - start_time
            logger.debug(
                f"[SEND META] Rank {current_rank}: All metadata sent in {total_elapsed:.3f}s " f"to rank {to_rank}"
            )

        except Exception as e:
            logger.error(f"[SEND META] Rank {current_rank}: Failed to send metadata to rank {to_rank}: {str(e)}")
            raise P2PCommunicationError(f"Failed to send metadata to rank {to_rank}: {str(e)}")

    def _recv_meta(self, from_rank: int, tag: int):
        current_rank = dist.get_rank(self.pg)
        timeout = COMMS_TIMEOUT  # seconds timeout

        logger.debug(f"[RECV META] Rank {current_rank}: Starting receive from {from_rank} ")

        try:
            start_time = time.time()

            # Add work handle to allow timeout checking
            self._record_operation("R")
            work = dist.irecv(
                self.first_metadata,
                src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
                # group=self.pg,
                # tag=tag,
            )

            # Wait with timeout
            while not work.is_completed():
                if time.time() - start_time > timeout:
                    logger.warning(
                        f"[RECV META] Rank {current_rank}: First metadata receive from rank {from_rank} "
                        f"timed out after {timeout}s"
                    )
                    # Try to abort the hanging communication
                    work.abort()
                    raise P2PCommunicationError(f"First metadata receive timed out from rank {from_rank}")
                time.sleep(0.1)  # Small sleep to prevent tight loop

            elapsed = time.time() - start_time
            logger.debug(
                f"[RECV META] Rank {current_rank}: Received first metadata in {elapsed:.3f}s " f"from rank {from_rank}"
            )

            (
                num_shape,
                num_stride,
                is_contiguous,
                untyped_storage_size,
                storage_offset,
                dtype_id,
                requires_grad_id,
            ) = self.first_metadata.tolist()

            second_metadata_num_elements = num_shape + num_stride
            logger.debug(
                f"[RECV META] Rank {current_rank}: Expecting second metadata of size {second_metadata_num_elements}"
            )

            start_time = time.time()
            self._record_operation("R")
            work = dist.irecv(
                self.second_metadata[:second_metadata_num_elements],
                src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
                # group=self.pg,
                # tag=tag,
            )
            while not work.is_completed():
                if time.time() - start_time > timeout:
                    logger.warning(
                        f"[RECV META] Rank {current_rank}: Second metadata receive from rank {from_rank} "
                        f"timed out after {timeout}s"
                    )
                    work.abort()
                    raise P2PCommunicationError(f"Second metadata receive timed out from rank {from_rank}")
                time.sleep(0.1)
            logger.debug(f"[RECV META] Rank {current_rank}: Second metadata received successfully")

            shape = self.second_metadata[:num_shape].tolist()
            stride = self.second_metadata[num_shape:second_metadata_num_elements].tolist()

            logger.debug(
                f"[RECV META] Rank {current_rank}: Metadata complete - "
                f"shape={shape}, stride={stride}, dtype={ID_TO_DTYPE[dtype_id]}, "
                f"requires_grad={ID_TO_REQUIRES_GRAD[requires_grad_id]}"
            )

            return P2PTensorMetaData(
                shape=shape,
                stride=stride,
                is_contiguous=ID_TO_IS_CONTIGUOUS[is_contiguous],
                untyped_storage_size=untyped_storage_size,
                storage_offset=storage_offset,
                dtype=ID_TO_DTYPE[dtype_id],
                requires_grad=ID_TO_REQUIRES_GRAD[requires_grad_id],
            )
        except Exception as e:
            logger.error(
                f"[RECV META] Rank {current_rank}: Failed to receive metadata from rank {from_rank}: {str(e)}"
            )
            raise P2PCommunicationError(f"Failed to receive metadata from rank {from_rank}: {str(e)}")

    def isend_tensors(self, tensors: List[torch.Tensor], to_rank: int, tag: int = 0) -> List[dist.Work]:
        logger.debug(f"Starting send operation from rank {dist.get_rank(self.pg)} to {to_rank}")
        futures = []
        current_rank = dist.get_rank(self.pg)
        logger.debug(f"Current rank {current_rank} sending to rank {to_rank}. Nb_tensors: {len(tensors)}")
        for tensor in tensors:
            if to_rank != current_rank:
                self._send_meta(tensor, to_rank=to_rank, tag=tag)
                if tensor.is_contiguous():
                    buffer = tensor
                else:
                    # If the tensor is not contiguous we send the entire storage
                    buffer = view_as_contiguous(tensor)

                # TODO @thomasw21: Find the issue with send/recv complex tensors
                buffer = torch.view_as_real(buffer) if buffer.is_complex() else buffer
                self._record_operation("S")
                futures.append(
                    dist.isend(
                        buffer,
                        dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
                        # group=self.pg,
                        # tag=tag,
                    )
                )
            else:
                raise ValueError("Tried sending tensor to itself")
        logger.debug(f"Completed send operation from rank {dist.get_rank(self.pg)} to {to_rank}")
        return futures

    def irecv_tensors(
        self, num_tensors: int, from_rank: int, tag: int = 0
    ) -> Tuple[List[torch.Tensor], List[dist.Work]]:
        futures = []
        buffers = []
        current_rank = dist.get_rank(self.pg)
        logger.debug(f"Starting receive operation on rank {current_rank} from rank {from_rank}")
        logger.debug(f"Current rank {current_rank} receiving from rank {from_rank}. Nb_tensors: {num_tensors}")
        for i in range(num_tensors):
            if from_rank != current_rank:
                logger.debug(f"Receiving metadata for tensor {i+1}/{num_tensors}")
                meta = self._recv_meta(from_rank=from_rank, tag=tag)
                logger.debug(f"Metadata received: shape={meta.shape}, dtype={meta.dtype}")

                buffer = meta.create_empty_storage(device=self.device)
                logger.debug(f"Created receive buffer of size {buffer.numel()}")

                self._record_operation("R")
                future = dist.irecv(
                    buffer,
                    src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
                    # group=self.pg,
                    # tag=tag,
                )
                futures.append(future)
                logger.debug(f"Started async receive for tensor {i+1}")

                buffer = meta.reshape(buffer=buffer)
                buffers.append(buffer)

        logger.debug(f"Completed receive setup on rank {current_rank} from rank {from_rank}")
        return buffers, futures

    def send_tensors(self, tensors: List[torch.Tensor], to_rank: int, tag: int = 0):
        futures = self.isend_tensors(tensors=tensors, to_rank=to_rank, tag=tag)
        for future in futures:
            future.wait()

    def recv_tensors(
        self, num_tensors: int, from_rank: int, tag: int = 0, timeout: float = COMMS_TIMEOUT
    ) -> List[torch.Tensor]:
        buffers, futures = self.irecv_tensors(num_tensors=num_tensors, from_rank=from_rank, tag=tag)

        start_time = time.time()
        for i, future in enumerate(futures):
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                logger.error(f"Timeout while waiting for tensor {i+1}/{len(futures)} from rank {from_rank}")
                raise TimeoutError(f"Receive operation timed out after {timeout} seconds")

            if not future.wait(timeout=remaining_time):
                logger.error(f"Failed to receive tensor {i+1}/{len(futures)} from rank {from_rank}")
                raise RuntimeError(f"Receive operation failed for tensor {i+1}")

        return buffers

    def get_history_str(self) -> str:
        """Get the history of operations as a space-separated string"""
        return " ".join(self.history)

    def clear_history(self):
        """Clear the operation history"""
        self.history = []
        self.send_counter = 0
        self.recv_counter = 0


class BatchTensorSendRecvState:
    """
    This class is used to register send/recv batches of tensors, and
    then executes send/recv in `flush()` calls. This is useful for
    amortizing the cost of sending and receiving tensors over multiple
    iterations.
    """

    p2p: P2P
    first_metadata_p2p_ops: List[dist.P2POp]
    second_metadata_p2p_ops: List[dist.P2POp]
    data_p2p_ops: List[dist.P2POp]
    recv_first_metadata_buffers: List[torch.Tensor]
    recv_from_ranks: List[int]

    def __init__(self, p2p: P2P):
        self.p2p = p2p
        self._reset()

    def _reset(self):
        self.first_metadata_p2p_ops: List[dist.P2POp] = []
        self.second_metadata_p2p_ops: List[dist.P2POp] = []
        self.data_p2p_ops: List[dist.P2POp] = []
        self.recv_first_metadata_buffers: List[torch.Tensor] = []
        self.recv_from_ranks: List[int] = []

    def __str__(self):
        return f"BatchTensorSendRecvState(first_metadata_p2p_ops={len(self.first_metadata_p2p_ops)}, second_metadata_p2p_ops={len(self.second_metadata_p2p_ops)}, data_p2p_ops={len(self.data_p2p_ops)}, recv_first_metadata_buffers={len(self.recv_first_metadata_buffers)}, recv_from_ranks={self.recv_from_ranks})"

    def add_send(self, tensor: torch.Tensor, to_rank: int, tag: int = 0):
        current_rank = dist.get_rank(self.p2p.pg)
        logger.debug(f"[SEND] Rank {current_rank}: Adding send operation to rank {to_rank}")
        logger.debug(f"[SEND] Rank {current_rank}: Tensor info - shape={tensor.shape}, dtype={tensor.dtype}")

        try:
            self.first_metadata_p2p_ops.append(
                self.p2p._send_first_metadata_p2p_op(tensor=tensor, to_rank=to_rank, tag=tag)
            )
            self.second_metadata_p2p_ops.append(
                self.p2p._send_second_metadata_p2p_op(tensor=tensor, to_rank=to_rank, tag=tag)
            )
            self.data_p2p_ops.append(
                self.p2p._send_data_p2p_op(tensor=view_as_contiguous(tensor), to_rank=to_rank, tag=tag)
            )
            logger.debug(f"[SEND] Rank {current_rank}: Successfully added send operations")
        except Exception as e:
            logger.error(f"[SEND] Rank {current_rank}: Failed to add send operations: {str(e)}")
            raise

    def add_recv(self, from_rank: int, tag: int = 0) -> int:
        """
        Only add p2p ops for the first operation, as `_recv_second_metadata` and `_recv_data_p2p_op`
        require results from the first metadata to be transferred first.
        Return: index of the recv_buffer in `self.recv_first_metadata_buffers`
        """
        buffer, recv_op = self.p2p._recv_first_metadata_p2p_op(from_rank=from_rank, tag=tag)
        self.first_metadata_p2p_ops.append(recv_op)
        self.recv_first_metadata_buffers.append(buffer)
        self.recv_from_ranks.append(from_rank)
        return len(self.recv_first_metadata_buffers) - 1

    def _send_recv_first_metadata(self) -> List[List[int]]:
        # Send/Recv first metadata
        reqs = dist.batch_isend_irecv(self.first_metadata_p2p_ops)
        for req in reqs:
            req.wait()
        # We want an early cpu/gpu sync here as we are right after the wait so it's nearly free.
        # Removing the tolist call here delays the sync and will impact performance.
        # We need to instantiate it in a list because it is used twice
        first_metadatas = [tensor.tolist() for tensor in self.recv_first_metadata_buffers]
        return first_metadatas

    def _send_recv_second_metadata(self, first_metadata: List[List[int]]) -> List[List[int]]:
        # turn a list of tuple into a tuple of list
        recv_second_metadata_buffers, recv_second_metadata_ops = zip(
            *(
                self.p2p._recv_second_metadata_p2p_op(
                    shape_length=num_shape, stride_length=num_stride, from_rank=from_rank
                )
                for (num_shape, num_stride, *_), from_rank in zip(first_metadata, self.recv_from_ranks)
            )
        )
        recv_second_metadata_ops = list(recv_second_metadata_ops)
        # Send/Recv second metadata
        reqs = dist.batch_isend_irecv(self.second_metadata_p2p_ops + recv_second_metadata_ops)
        for req in reqs:
            req.wait()

        # We want an early cpu/gpu sync here as we are right after the wait so it's nearly free.
        # Removing the tolist call here delays the sync and will impact performance.
        second_metadatas = [tensor.tolist() for tensor in recv_second_metadata_buffers]
        return second_metadatas

    def _send_recv_data(self, tensor_metadatas: List[P2PTensorMetaData]) -> List[torch.Tensor]:
        # turn a list of tuples into a tuple of list
        recv_data_buffers, recv_data_ops = zip(
            *(
                self.p2p._recv_data_p2p_op(tensor_metadata=tensor_metadata, from_rank=from_rank)
                for tensor_metadata, from_rank in zip(tensor_metadatas, self.recv_from_ranks)
            )
        )
        recv_data_ops = list(recv_data_ops)
        # Send/Recv tensor data
        futures = dist.batch_isend_irecv(self.data_p2p_ops + recv_data_ops)
        for future in futures:
            if not future.wait(timeout=COMMS_TIMEOUT):  # 1 minute timeout
                raise TimeoutError("P2P communication timeout")

        # Format tensor by setting the stride
        return [
            recv_data_buffer.as_strided(size=tuple(tensor_metadata.shape), stride=tuple(tensor_metadata.stride))
            for recv_data_buffer, tensor_metadata in zip(recv_data_buffers, tensor_metadatas)
        ]

    def flush(self):
        current_rank = dist.get_rank(self.p2p.pg)
        # logger.debug(
        #     f"[FLUSH] Rank {current_rank}: Starting flush - "
        #     f"first_meta_ops={len(self.first_metadata_p2p_ops)}, "
        #     f"second_meta_ops={len(self.second_metadata_p2p_ops)}, "
        #     f"data_ops={len(self.data_p2p_ops)}, "
        #     f"recv_buffers={len(self.recv_first_metadata_buffers)}"
        # )

        if len(self.first_metadata_p2p_ops) == 0:
            # logger.debug(f"[FLUSH] Rank {current_rank}: No operations to flush")
            return []

        try:
            if len(self.recv_first_metadata_buffers) == 0:
                logger.debug(f"[FLUSH] Rank {current_rank}: Processing send-only operations")
                reqs = dist.batch_isend_irecv(
                    self.first_metadata_p2p_ops + self.second_metadata_p2p_ops + self.data_p2p_ops
                )
                for i, req in enumerate(reqs):
                    logger.debug(f"[FLUSH] Rank {current_rank}: Waiting for operation {i+1}/{len(reqs)}")
                    req.wait()
                logger.debug(f"[FLUSH] Rank {current_rank}: All send operations completed")
            else:
                logger.debug(f"[FLUSH] Rank {current_rank}: Processing send/receive operations")
                first_metadatas = self._send_recv_first_metadata()
                logger.debug(f"[FLUSH] Rank {current_rank}: First metadata exchange complete")

                second_metadatas = self._send_recv_second_metadata(first_metadatas)
                logger.debug(f"[FLUSH] Rank {current_rank}: Second metadata exchange complete")

                tensor_metadatas = [
                    P2PTensorMetaData.from_metadata(first_metadata, second_metadata)
                    for first_metadata, second_metadata in zip(first_metadatas, second_metadatas)
                ]

                recv_tensors = self._send_recv_data(tensor_metadatas)
                logger.debug(f"[FLUSH] Rank {current_rank}: Data exchange complete")

                self._reset()
                return recv_tensors
        except Exception as e:
            logger.error(f"[FLUSH] Rank {current_rank}: Failed during flush: {str(e)}")
            raise
        finally:
            logger.debug(f"[FLUSH] Rank {current_rank}: Flush operation complete")
            self._reset()
