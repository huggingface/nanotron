import contextlib
from pathlib import Path
from typing import Tuple, Union

import fsspec
from fsspec.implementations import local


def get_filesystem_and_path(path: Path, storage_options=None) -> Tuple[fsspec.AbstractFileSystem, str]:
    # Use supported filesystems in `fsspec`. If you need another one, please use `fsspec.registry.register_implementation`
    # DO NOT USE `mode` argument as it adds a suffix `0.part` when using `mode="w"`.
    fs, _, paths = fsspec.core.get_fs_token_paths(str(path), storage_options=storage_options)
    assert len(paths) == 1
    return fs, paths[0]


@contextlib.contextmanager
def fs_open(
    file: Union[str, Path],
    mode="r",
):
    # TODO @thomasw21: pass storage options.
    fs, path = get_filesystem_and_path(file)
    with fs.open(path, mode=mode) as f:
        yield f


def fs_copy(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
):
    """Copy file from input to output (possibly on s3/other fs)"""
    with fs_open(input_file, mode="rb") as fi, fs_open(output_file, mode="wb") as fo:
        fo.write(fi.read())


def check_path_is_local(path: Path, storage_options=None) -> bool:
    return isinstance(get_filesystem_and_path(path=path, storage_options=storage_options)[0], local.LocalFileSystem)
