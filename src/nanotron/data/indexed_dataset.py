import numpy
import torch

from nanotron import logging

logger = logging.get_logger(__name__)


class MMapIndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_to_mmap (str): Path to the mmap dataset

    """

    def __init__(self, path_to_mmap: str) -> None:
        super().__init__()
        self.path_to_mmap = path_to_mmap

        self.bin_buffer_mmap = numpy.memmap(self.path_to_mmap, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

    def get(self, idx: int, length: int, offset: int = None) -> numpy.ndarray:
        """
        Returns length + 1 tokens from the memmap dataset
        For CausalLM length == sequence_length

        Args:
            idx (int): The index into the memmap dataset
            length (int): The quantity of tokens to extract from the memmap dataset
            offset (int): The offset in the memmap dataset to extract tokens from
        """
        if offset is None:
            # dtype=uint16, 2 bytes per token
            offset = idx * length * 2

        sequence = numpy.frombuffer(self.bin_buffer, dtype=numpy.uint16, count=length + 1, offset=offset)
        return sequence

    def __getitem__(self, idx: int) -> numpy.ndarray:
        """
        Returns 1 token from the memmap dataset. Check get method

        Args:
            idx (int): The index of the token
        """
        # dtype=uint16, 2 bytes per token
        offset = idx * 2
        token = numpy.frombuffer(self.bin_buffer, dtype=numpy.uint16, count=1, offset=offset)
        return token

    def __del__(self) -> None:
        """Clean up the object"""
        if self.bin_buffer_mmap is not None:
            self.bin_buffer_mmap._mmap.close()
        del self.bin_buffer_mmap

    def __len__(self) -> int:
        """
        Returns the number of tokens in the file
        """
        # dtype=uint16, 2 bytes per token
        return int(len(self.bin_buffer) / 2)
