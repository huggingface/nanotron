import dataclasses
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


class P2P:
    def __init__(self, pg: dist.ProcessGroup, device: torch.device):
        self.pg = pg
        self.device = device
        self.first_metadata = torch.empty(FIRST_METADATA_SIZE, dtype=torch.long, device=self.device)
        self.second_metadata = torch.empty(SECOND_METADATA_SIZE, dtype=torch.long, device=self.device)

    def _send_first_metadata_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        first_metadata = P2PTensorMetaData.to_first_metadata(tensor=tensor, device=self.device)
        return dist.P2POp(
            op=dist.isend,
            tensor=first_metadata,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_first_metadata_p2p_op(self, from_rank: int, tag: int = 0) -> Tuple[torch.Tensor, dist.P2POp]:
        first_metadata_buffer = torch.empty((FIRST_METADATA_SIZE,), dtype=torch.long, device=self.device)
        return first_metadata_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=first_metadata_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_second_metadata_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        second_metadata = P2PTensorMetaData.to_second_metadata(tensor=tensor, device=self.device)
        return dist.P2POp(
            op=dist.isend,
            tensor=second_metadata,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_second_metadata_p2p_op(
        self, shape_length: int, stride_length: int, from_rank: int, tag: int = 0
    ) -> Tuple[torch.Tensor, dist.P2POp]:
        second_metadata_buffer = torch.empty((shape_length + stride_length,), dtype=torch.long, device=self.device)
        return second_metadata_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=second_metadata_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_data_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        return dist.P2POp(
            op=dist.isend,
            tensor=tensor,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_data_p2p_op(
        self, tensor_metadata: P2PTensorMetaData, from_rank: int, tag: int = 0
    ) -> Tuple[torch.Tensor, dist.P2POp]:
        tensor_buffer = tensor_metadata.create_empty_storage(self.device)
        return tensor_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=tensor_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_meta(self, tensor: torch.Tensor, to_rank: int, tag: int):
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
        dist.send(
            self.first_metadata,
            dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

        second_metadata = tensor.shape + tensor.stride()
        assert len(tensor.shape) == self.first_metadata[0]
        assert len(tensor.stride()) == self.first_metadata[1]

        # increase buffer size
        if len(second_metadata) > len(self.second_metadata):
            self.second_metadata = torch.empty(len(second_metadata), dtype=torch.long, device=self.device)

        self.second_metadata[: len(second_metadata)].copy_(torch.tensor(second_metadata, dtype=torch.long))

        dist.send(
            self.second_metadata[: len(second_metadata)],
            dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_meta(self, from_rank: int, tag: int) -> P2PTensorMetaData:
        dist.recv(
            self.first_metadata,
            src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )
        (
            num_shape,
            num_stride,
            is_contiguous,
            untyped_storage_size,
            storage_offset,
            dtype_id,
            requires_grad_id,
        ) = self.first_metadata

        # self.pg.recv([second], from_rank, 0).wait() # more direct API
        second_metadata_num_elements = num_shape + num_stride

        # increase buffer size
        if second_metadata_num_elements > len(self.second_metadata):
            self.second_metadata = torch.empty(second_metadata_num_elements, dtype=torch.long, device=self.device)

        dist.recv(
            self.second_metadata[:second_metadata_num_elements],
            src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

        shape = self.second_metadata[:num_shape]
        stride = self.second_metadata[num_shape:second_metadata_num_elements]

        return P2PTensorMetaData(
            dtype=ID_TO_DTYPE[dtype_id],
            requires_grad=ID_TO_REQUIRES_GRAD[requires_grad_id],
            shape=shape,
            stride=stride,
            is_contiguous=ID_TO_IS_CONTIGUOUS[is_contiguous],
            untyped_storage_size=untyped_storage_size,
            storage_offset=storage_offset,
        )

    def isend_tensors(self, tensors: List[torch.Tensor], to_rank: int, tag: int = 0) -> List[dist.Work]:
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

                futures.append(
                    dist.isend(
                        buffer,
                        dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
                        group=self.pg,
                        tag=tag,
                    )
                )
            else:
                raise ValueError("Tried sending tensor to itself")
        return futures

    def irecv_tensors(
        self, num_tensors: int, from_rank: int, tag: int = 0
    ) -> Tuple[List[torch.Tensor], List[dist.Work]]:
        futures = []
        buffers = []
        current_rank = dist.get_rank(self.pg)
        logger.debug(f"Current rank {current_rank} receiving from rank {from_rank}. Nb_tensors: {num_tensors}")
        for _ in range(num_tensors):
            if from_rank != current_rank:
                meta = self._recv_meta(from_rank=from_rank, tag=tag)

                buffer = meta.create_empty_storage(device=self.device)

                futures.append(
                    dist.irecv(
                        buffer,
                        src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
                        group=self.pg,
                        tag=tag,
                    )
                )

                buffer = meta.reshape(buffer=buffer)

                # Add to the list
                buffers.append(buffer)
            else:
                raise ValueError("Tried receiving tensor from itself")
        return buffers, futures

    def send_tensors(self, tensors: List[torch.Tensor], to_rank: int, tag: int = 0):
        futures = self.isend_tensors(tensors=tensors, to_rank=to_rank, tag=tag)
        for future in futures:
            future.wait()

    def recv_tensors(self, num_tensors: int, from_rank: int, tag: int = 0) -> List[torch.Tensor]:
        buffers, futures = self.irecv_tensors(num_tensors=num_tensors, from_rank=from_rank, tag=tag)
        for future in futures:
            future.wait()
        return buffers


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
        self.first_metadata_p2p_ops.append(
            self.p2p._send_first_metadata_p2p_op(tensor=tensor, to_rank=to_rank, tag=tag)
        )
        self.second_metadata_p2p_ops.append(
            self.p2p._send_second_metadata_p2p_op(tensor=tensor, to_rank=to_rank, tag=tag)
        )
        self.data_p2p_ops.append(
            self.p2p._send_data_p2p_op(tensor=view_as_contiguous(tensor), to_rank=to_rank, tag=tag)
        )

    def add_recv(self, from_rank: int, tag: int = 0) -> int:
        """
        Only add p2p ops for the first operation, as `_recv_second_metadata` and `_recv_data_p2p_op`
        require results from the first metadata to be transfered first.
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
            future.wait()

        # Format tensor by setting the stride
        return [
            recv_data_buffer.as_strided(size=tuple(tensor_metadata.shape), stride=tuple(tensor_metadata.stride))
            for recv_data_buffer, tensor_metadata in zip(recv_data_buffers, tensor_metadatas)
        ]

    def flush(self) -> List[torch.Tensor]:
        """
        Run all communication in a batch.
        Return `torch.Tensor` in the case of recv.
        """
        assert len(self.recv_first_metadata_buffers) == len(
            self.recv_from_ranks
        ), f"len(self.recv_first_metadata_buffers)={len(self.recv_first_metadata_buffers)}, len(self.recv_from_ranks)={len(self.recv_from_ranks)} but should be equal."

        # If there is no communication, return
        if len(self.first_metadata_p2p_ops) == 0:
            return []

        # If there is no recv
        if len(self.recv_first_metadata_buffers) == 0:
            reqs = dist.batch_isend_irecv(
                self.first_metadata_p2p_ops + self.second_metadata_p2p_ops + self.data_p2p_ops
            )
            for req in reqs:
                req.wait()
            self._reset()
            return []

        # Send/Recv first metadata
        logger.debug(f"First metadata: {[p2pop.op for p2pop in self.first_metadata_p2p_ops]}")
        # TODO(kunhao): We could actually send all at once like the above no recv case. But I need to benchmark the performance.
        first_metadatas = self._send_recv_first_metadata()
        # Send/Recv second metadata
        second_metadatas = self._send_recv_second_metadata(first_metadatas)

        tensor_metadatas = [
            P2PTensorMetaData.from_metadata(first_metadata, second_metadata)
            for first_metadata, second_metadata in zip(first_metadatas, second_metadatas)
        ]

        recv_tensors = self._send_recv_data(tensor_metadatas)
        # Reset state
        self._reset()

        return recv_tensors
