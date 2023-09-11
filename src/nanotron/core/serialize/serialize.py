import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from fsspec.implementations import local
from safetensors import safe_open as safetensors_safe_open
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from safetensors.torch import save_file as safetensors_save_file
from torch.serialization import MAP_LOCATION

from brrr.core.serialize.path import fs_open, get_filesystem_and_path


def save_file(
    filename: Union[str, Path],
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    fs, _ = get_filesystem_and_path(filename)
    if isinstance(fs, local.LocalFileSystem):
        safetensors_save_file(tensors=tensors, filename=filename, metadata=metadata)
    else:
        bytes = safetensors_save(tensors=tensors, metadata=metadata)
        with fs_open(file=filename, mode="wb") as fi:
            fi.write(bytes)


class fs_safetensors_safe_open:
    def __init__(self, f, device: str):
        # Download the entire file
        raw_data = f.read()

        # Parse data and get tensors in dictionary format
        self.object = safetensors_load(raw_data)
        keys = list(self.object.keys())
        for k in keys:
            self.object[k] = self.object[k].to(device)

        # Get metadata
        n_header = raw_data[:8]
        n = int.from_bytes(n_header, "little")
        metadata_bytes = raw_data[8 : 8 + n]
        header = json.loads(metadata_bytes)
        self._metadata = header.get("__metadata__", {})

    def metadata(self):
        return self._metadata

    def get_slice(self, embedding):
        return self.object[embedding]

    def get_tensor(self, embedding):
        return self.object[embedding]

    def keys(self):
        return self.object.keys()


@contextlib.contextmanager
def safe_open(
    path: Union[str, Path],
    framework: str,  # for example: "pt",
    device: str,  # for example: "cuda"
):
    fs, local_path = get_filesystem_and_path(path)
    if isinstance(fs, local.LocalFileSystem):
        with safetensors_safe_open(local_path, framework=framework, device=device) as fi:
            yield fi
    else:
        with fs_open(path, mode="rb") as fi:
            yield fs_safetensors_safe_open(fi, device=device)


def torch_save(
    obj: object,
    file: Union[str, Path],
) -> None:
    with fs_open(file=file, mode="wb") as fo:
        return torch.save(obj, fo)


def torch_load(
    file: Union[str, Path],
    map_location: MAP_LOCATION = None,
) -> Any:
    with fs_open(file=file, mode="rb") as fi:
        return torch.load(fi, map_location=map_location)
