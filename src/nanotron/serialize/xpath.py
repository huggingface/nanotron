import glob
import platform
from packaging import version
from dataclasses import dataclass, field, InitVar
import os
import posixpath
import contextlib
import re
import copy
import time
import warnings
from asyncio import TimeoutError
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

import fsspec
from aiohttp.client_exceptions import ClientError
from urllib.parse import urlparse

from nanotron import config, __version__, constants
from nanotron.logging import get_logger

try:
    from huggingface_hub import HfFolder
except ImportError:
    HfFolder = None

logger = get_logger(__name__)

SINGLE_SLASH_AFTER_PROTOCOL_PATTERN = re.compile(r"(?<!:):/")

MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL = {
    bytes.fromhex("504B0304"): "zip",
    bytes.fromhex("504B0506"): "zip",  # empty archive
    bytes.fromhex("504B0708"): "zip",  # spanned archive
    bytes.fromhex("425A68"): "bz2",
    bytes.fromhex("1F8B"): "gzip",
    bytes.fromhex("FD377A585A00"): "xz",
    bytes.fromhex("04224D18"): "lz4",
    bytes.fromhex("28B52FFD"): "zstd",
}
MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL = {
    b"Rar!": "rar",
}
MAGIC_NUMBER_MAX_LENGTH = max(
    len(magic_number)
    for magic_number in chain(MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL, MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL)
)

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
    # TODO @thomasw21: pass storage options
    fs, path = get_filesystem_and_path(file)
    with fs.open(path, mode=mode) as f:
        yield f


@dataclass
class DownloadConfig:
    """Configuration for our cached path manager.

    Attributes:
        cache_dir (`str` or `Path`, *optional*):
            Specify a cache directory to save the file to (overwrite the
            default cache dir).
        force_download (`bool`, defaults to `False`):
            If `True`, re-dowload the file even if it's already cached in
            the cache dir.
        resume_download (`bool`, defaults to `False`):
            If `True`, resume the download if an incompletely received file is
            found.
        proxies (`dict`, *optional*):
        user_agent (`str`, *optional*):
            Optional string or dict that will be appended to the user-agent on remote
            requests.
        extract_compressed_file (`bool`, defaults to `False`):
            If `True` and the path point to a zip or tar file,
            extract the compressed file in a folder along the archive.
        force_extract (`bool`, defaults to `False`):
            If `True` when `extract_compressed_file` is `True` and the archive
            was already extracted, re-extract the archive and override the folder where it was extracted.
        delete_extracted (`bool`, defaults to `False`):
            Whether to delete (or keep) the extracted files.
        use_etag (`bool`, defaults to `True`):
            Whether to use the ETag HTTP response header to validate the cached files.
        num_proc (`int`, *optional*):
            The number of processes to launch to download the files in parallel.
        max_retries (`int`, default to `1`):
            The number of times to retry an HTTP request if it fails.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token
            for remote files on the Datasets Hub. If `True`, or not specified, will get token from `~/.huggingface`.
        use_auth_token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token
            for remote files on the Datasets Hub. If `True`, or not specified, will get token from `~/.huggingface`.

            <Deprecated version="2.14.0">

            `use_auth_token` was deprecated in favor of `token` in version 2.14.0 and will be removed in 3.0.0.

            </Deprecated>

        ignore_url_params (`bool`, defaults to `False`):
            Whether to strip all query parameters and fragments from
            the download URL before using it for caching the file.
        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the dataset file-system backend, if any.
        download_desc (`str`, *optional*):
            A description to be displayed alongside with the progress bar while downloading the files.
    """

    cache_dir: Optional[Union[str, Path]] = None
    force_download: bool = False
    resume_download: bool = False
    local_files_only: bool = False
    proxies: Optional[Dict] = None
    user_agent: Optional[str] = None
    extract_compressed_file: bool = False
    force_extract: bool = False
    delete_extracted: bool = False
    use_etag: bool = True
    num_proc: Optional[int] = None
    max_retries: int = 1
    token: Optional[Union[str, bool]] = None
    use_auth_token: InitVar[Optional[Union[str, bool]]] = "deprecated"
    ignore_url_params: bool = False
    storage_options: Dict[str, Any] = field(default_factory=dict)
    download_desc: Optional[str] = None

    def __post_init__(self, use_auth_token):
        if use_auth_token != "deprecated":
            warnings.warn(
                "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'token={use_auth_token}' instead.",
                FutureWarning,
            )
            self.token = use_auth_token
        if "hf" not in self.storage_options:
            self.storage_options["hf"] = {"token": self.token, "endpoint": config.HF_ENDPOINT}

    def copy(self) -> "DownloadConfig":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def __setattr__(self, name, value):
        if name == "token" and getattr(self, "storage_options", None) is not None:
            if "hf" not in self.storage_options:
                self.storage_options["hf"] = {"token": value, "endpoint": config.HF_ENDPOINT}
            elif getattr(self.storage_options["hf"], "token", None) is None:
                self.storage_options["hf"]["token"] = value
        super().__setattr__(name, value)


class NonStreamableDatasetError(Exception):
    pass


def is_local_path(url_or_filename: str) -> bool:
    # On unix the scheme of a local path is empty (for both absolute and relative),
    # while on windows the scheme is the drive name (ex: "c") for absolute paths.
    # for details on the windows behavior, see https://bugs.python.org/issue42215
    return urlparse(url_or_filename).scheme == "" or os.path.ismount(urlparse(url_or_filename).scheme + ":/")


def get_nanotron_user_agent(user_agent: Optional[Union[str, dict]] = None) -> str:
    ua = f"nanotron/{__version__}"
    ua += f"; python/{constants.PY_VERSION}"
    if isinstance(user_agent, dict):
        ua += f"; {'; '.join(f'{k}/{v}' for k, v in user_agent.items())}"
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def get_authentication_headers_for_url(
    url: str, token: Optional[Union[str, bool]] = None, use_auth_token: Optional[Union[str, bool]] = "deprecated"
) -> dict:
    """Handle the HF authentication"""
    if use_auth_token != "deprecated":
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'token={use_auth_token}' instead.",
            FutureWarning,
        )
        token = use_auth_token
    headers = {}
    if url.startswith(constants.HF_ENDPOINT):
        if token is False:
            token = None
        elif isinstance(token, str):
            token = token
        else:
            token = HfFolder.get_token()

        if token:
            headers["authorization"] = f"Bearer {token}"
    return headers


def xjoin(a, *p):
    """
    This function extends os.path.join to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xjoin function allows you to apply the join on the first path of the chain.

    Example::

        >>> xjoin("zip://folder1::https://host.com/archive.zip", "file.txt")
        zip://folder1/file.txt::https://host.com/archive.zip
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.join(a, *p)
    else:
        a = posixpath.join(a, *p)
        return "::".join([a] + b)


def xdirname(a):
    """
    This function extends os.path.dirname to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xdirname function allows you to apply the dirname on the first path of the chain.

    Example::

        >>> xdirname("zip://folder1/file.txt::https://host.com/archive.zip")
        zip://folder1::https://host.com/archive.zip
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        a = os.path.dirname(Path(a).as_posix())
    else:
        a = posixpath.dirname(a)
    # if we end up at the root of the protocol, we get for example a = 'http:'
    # so we have to fix it by adding the '//' that was removed:
    if a.endswith(":"):
        a += "//"
    return "::".join([a] + b)


def xexists(urlpath: str, download_config: Optional[DownloadConfig] = None):
    """Extend `os.path.exists` function to support both local and remote files.

    Args:
        urlpath (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """

    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        return os.path.exists(main_hop)
    else:
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        fs, *_ = fsspec.get_fs_token_paths(urlpath, storage_options=storage_options)
        return fs.exists(main_hop)


def xbasename(a):
    """
    This function extends os.path.basename to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xbasename function allows you to apply the basename on the first path of the chain.

    Example::

        >>> xbasename("zip://folder1/file.txt::https://host.com/archive.zip")
        file.txt
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.basename(Path(a).as_posix())
    else:
        return posixpath.basename(a)


def xsplit(a):
    """
    This function extends os.path.split to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xsplit function allows you to apply the xsplit on the first path of the chain.

    Example::

        >>> xsplit("zip://folder1/file.txt::https://host.com/archive.zip")
        ('zip://folder1::https://host.com/archive.zip', 'file.txt')
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.split(Path(a).as_posix())
    else:
        a, tail = posixpath.split(a)
        return "::".join([a + "//" if a.endswith(":") else a] + b), tail


def xsplitext(a):
    """
    This function extends os.path.splitext to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xsplitext function allows you to apply the splitext on the first path of the chain.

    Example::

        >>> xsplitext("zip://folder1/file.txt::https://host.com/archive.zip")
        ('zip://folder1/file::https://host.com/archive.zip', '.txt')
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.splitext(Path(a).as_posix())
    else:
        a, ext = posixpath.splitext(a)
        return "::".join([a] + b), ext


def xisfile(path, download_config: Optional[DownloadConfig] = None) -> bool:
    """Extend `os.path.isfile` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.isfile(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        fs, *_ = fsspec.get_fs_token_paths(path, storage_options=storage_options)
        return fs.isfile(main_hop)


def xgetsize(path, download_config: Optional[DownloadConfig] = None) -> int:
    """Extend `os.path.getsize` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `int`: optional
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.getsize(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        fs, *_ = fsspec.get_fs_token_paths(path, storage_options=storage_options)
        size = fs.size(main_hop)
        if size is None:
            # use xopen instead of fs.open to make data fetching more robust
            with xopen(path, download_config=download_config) as f:
                size = len(f.read())
        return size


def xisdir(path, download_config: Optional[DownloadConfig] = None) -> bool:
    """Extend `os.path.isdir` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.isdir(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        fs, *_ = fsspec.get_fs_token_paths(path, storage_options=storage_options)
        inner_path = main_hop.split("://")[1]
        if not inner_path.strip("/"):
            return True
        return fs.isdir(inner_path)


def xrelpath(path, start=None):
    """Extend `os.path.relpath` function to support remote files.

    Args:
        path (`str`): URL path.
        start (`str`): Start URL directory path.

    Returns:
        `str`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.relpath(main_hop, start=start) if start else os.path.relpath(main_hop)
    else:
        return posixpath.relpath(main_hop, start=str(start).split("::")[0]) if start else os.path.relpath(main_hop)


def _add_retries_to_file_obj_read_method(file_obj):
    read = file_obj.read
    max_retries = config.STREAMING_READ_MAX_RETRIES

    def read_with_retries(*args, **kwargs):
        disconnect_err = None
        for retry in range(1, max_retries + 1):
            try:
                out = read(*args, **kwargs)
                break
            except (ClientError, TimeoutError) as err:
                disconnect_err = err
                logger.warning(
                    f"Got disconnected from remote data host. Retrying in {config.STREAMING_READ_RETRY_INTERVAL}sec [{retry}/{max_retries}]"
                )
                time.sleep(config.STREAMING_READ_RETRY_INTERVAL)
        else:
            raise ConnectionError("Server Disconnected") from disconnect_err
        return out

    file_obj.read = read_with_retries


def _prepare_path_and_storage_options(
    urlpath: str, download_config: Optional[DownloadConfig] = None
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    prepared_urlpath = []
    prepared_storage_options = {}
    for hop in urlpath.split("::"):
        hop, storage_options = _prepare_single_hop_path_and_storage_options(hop, download_config=download_config)
        prepared_urlpath.append(hop)
        prepared_storage_options.update(storage_options)
    return "::".join(prepared_urlpath), storage_options


def _prepare_single_hop_path_and_storage_options(
    urlpath: str, download_config: Optional[DownloadConfig] = None
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Prepare the URL and the kwargs that must be passed to the HttpFileSystem or to requests.get/head

    Storage options are formatted in the form {protocol: storage_options_for_protocol}
    """
    token = None if download_config is None else download_config.token
    protocol = urlpath.split("://")[0] if "://" in urlpath else "file"
    if download_config is not None and protocol in download_config.storage_options:
        storage_options = download_config.storage_options[protocol]
    elif download_config is not None and protocol not in download_config.storage_options:
        storage_options = {
            option_name: option_value
            for option_name, option_value in download_config.storage_options.items()
            if option_name not in fsspec.available_protocols()
        }
    else:
        storage_options = {}
    if storage_options:
        storage_options = {protocol: storage_options}
    if protocol in ["http", "https"]:
        storage_options[protocol] = {
            "headers": {
                **get_authentication_headers_for_url(urlpath, token=token),
                "user-agent": get_nanotron_user_agent(),
            },
            "client_kwargs": {"trust_env": True},  # Enable reading proxy env variables.
            **(storage_options.get(protocol, {})),
        }
        if urlpath.startswith("https://raw.githubusercontent.com/"):
            # Workaround for served data with gzip content-encoding: https://github.com/fsspec/filesystem_spec/issues/389
            storage_options[protocol]["headers"]["Accept-Encoding"] = "identity"
    elif protocol == "hf":
        storage_options[protocol] = {
            "token": token,
            "endpoint": config.HF_ENDPOINT,
            **storage_options.get(protocol, {}),
        }
    return urlpath, storage_options


def xopen(file: str, mode="r", *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    """Extend `open` function to support remote files using `fsspec`.

    It also has a retry mechanism in case connection fails.
    The `args` and `kwargs` are passed to `fsspec.open`, except `token` which is used for queries to private repos on huggingface.co

    Args:
        file (`str`): Path name of the file to be opened.
        mode (`str`, *optional*, default "r"): Mode in which the file is opened.
        *args: Arguments to be passed to `fsspec.open`.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs: Keyword arguments to be passed to `fsspec.open`.

    Returns:
        file object
    """
    # This works as well for `xopen(str(Path(...)))`
    file_str = _as_str(file)
    main_hop, *rest_hops = file_str.split("::")
    if is_local_path(main_hop):
        return open(main_hop, mode, *args, **kwargs)
    # add headers and cookies for authentication on the HF Hub and for Google Drive
    file, storage_options = _prepare_path_and_storage_options(file_str, download_config=download_config)
    kwargs = {**kwargs, **(storage_options or {})}
    try:
        file_obj = fsspec.open(file, mode=mode, *args, **kwargs).open()
    except ValueError as e:
        if str(e) == "Cannot seek streaming HTTP file":
            raise NonStreamableDatasetError(
                "Streaming is not possible for this dataset because data host server doesn't support HTTP range "
                "requests. You can still load this dataset in non-streaming mode by passing `streaming=False` (default)"
            ) from e
        else:
            raise
    except FileNotFoundError:
        if file.startswith(config.HF_ENDPOINT):
            raise FileNotFoundError(
                file + "\nIf the repo is private or gated, make sure to log in with `huggingface-cli login`."
            ) from None
        else:
            raise
    _add_retries_to_file_obj_read_method(file_obj)
    return file_obj


def xlistdir(path: str, download_config: Optional[DownloadConfig] = None) -> List[str]:
    """Extend `os.listdir` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `list` of `str`
    """
    main_hop, *rest_hops = _as_str(path).split("::")
    if is_local_path(main_hop):
        return os.listdir(path)
    else:
        # globbing inside a zip in a private repo requires authentication
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        fs, *_ = fsspec.get_fs_token_paths(path, storage_options=storage_options)
        inner_path = main_hop.split("://")[1]
        if inner_path.strip("/") and not fs.isdir(inner_path):
            raise FileNotFoundError(f"Directory doesn't exist: {path}")
        objects = fs.listdir(inner_path)
        return [os.path.basename(obj["name"]) for obj in objects]


def xglob(urlpath, *, recursive=False, download_config: Optional[DownloadConfig] = None):
    """Extend `glob.glob` function to support remote files.

    Args:
        urlpath (`str`): URL path with shell-style wildcard patterns.
        recursive (`bool`, default `False`): Whether to match the "**" pattern recursively to zero or more
            directories or subdirectories.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `list` of `str`
    """
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        return glob.glob(main_hop, recursive=recursive)
    else:
        # globbing inside a zip in a private repo requires authentication
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        fs, *_ = fsspec.get_fs_token_paths(urlpath, storage_options=storage_options)
        # - If there's no "*" in the pattern, get_fs_token_paths() doesn't do any pattern matching
        #   so to be able to glob patterns like "[0-9]", we have to call `fs.glob`.
        # - Also "*" in get_fs_token_paths() only matches files: we have to call `fs.glob` to match directories.
        # - If there is "**" in the pattern, `fs.glob` must be called anyway.
        inner_path = main_hop.split("://")[1]
        globbed_paths = fs.glob(inner_path)
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        return ["::".join([f"{protocol}://{globbed_path}"] + rest_hops) for globbed_path in globbed_paths]


def xwalk(urlpath, download_config: Optional[DownloadConfig] = None, **kwargs):
    """Extend `os.walk` function to support remote files.

    Args:
        urlpath (`str`): URL root path.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs: Additional keyword arguments forwarded to the underlying filesystem.


    Yields:
        `tuple`: 3-tuple (dirpath, dirnames, filenames).
    """
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        yield from os.walk(main_hop, **kwargs)
    else:
        # walking inside a zip in a private repo requires authentication
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        fs, *_ = fsspec.get_fs_token_paths(urlpath, storage_options=storage_options)
        inner_path = main_hop.split("://")[1]
        if inner_path.strip("/") and not fs.isdir(inner_path):
            return []
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        for dirpath, dirnames, filenames in fs.walk(inner_path, **kwargs):
            yield "::".join([f"{protocol}://{dirpath}"] + rest_hops), dirnames, filenames


class xPath(type(Path())):
    """Extension of `pathlib.Path` to support both local paths and remote URLs."""

    def __str__(self):
        path_str = super().__str__()
        main_hop, *rest_hops = path_str.split("::")
        if is_local_path(main_hop):
            return main_hop
        path_as_posix = path_str.replace("\\", "/")
        path_as_posix = SINGLE_SLASH_AFTER_PROTOCOL_PATTERN.sub("://", path_as_posix)
        path_as_posix += "//" if path_as_posix.endswith(":") else ""  # Add slashes to root of the protocol
        return path_as_posix

    def exists(self, download_config: Optional[DownloadConfig] = None):
        """Extend `pathlib.Path.exists` method to support both local and remote files.

        Args:
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Returns:
            `bool`
        """
        return xexists(str(self), download_config=download_config)

    def glob(self, pattern, download_config: Optional[DownloadConfig] = None):
        """Glob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Yields:
            [`xPath`]
        """
        posix_path = self.as_posix()
        main_hop, *rest_hops = posix_path.split("::")
        if is_local_path(main_hop):
            yield from Path(main_hop).glob(pattern)
        else:
            # globbing inside a zip in a private repo requires authentication
            if rest_hops:
                urlpath = rest_hops[0]
                urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
                storage_options = {urlpath.split("://")[0]: storage_options}
                posix_path = "::".join([main_hop, urlpath, *rest_hops[1:]])
            else:
                storage_options = None
            fs, *_ = fsspec.get_fs_token_paths(xjoin(posix_path, pattern), storage_options=storage_options)
            # - If there's no "*" in the pattern, get_fs_token_paths() doesn't do any pattern matching
            #   so to be able to glob patterns like "[0-9]", we have to call `fs.glob`.
            # - Also "*" in get_fs_token_paths() only matches files: we have to call `fs.glob` to match directories.
            # - If there is "**" in the pattern, `fs.glob` must be called anyway.
            globbed_paths = fs.glob(xjoin(main_hop, pattern))
            for globbed_path in globbed_paths:
                yield type(self)("::".join([f"{fs.protocol}://{globbed_path}"] + rest_hops))

    def rglob(self, pattern, **kwargs):
        """Rglob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.

        Yields:
            [`xPath`]
        """
        return self.glob("**/" + pattern, **kwargs)

    @property
    def parent(self) -> "xPath":
        """Name function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            [`xPath`]
        """
        return type(self)(xdirname(self.as_posix()))

    @property
    def name(self) -> str:
        """Name function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).name

    @property
    def stem(self) -> str:
        """Stem function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).stem

    @property
    def suffix(self) -> str:
        """Suffix function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).suffix

    def open(self, *args, **kwargs):
        """Extend :func:`xopen` to support argument of type :obj:`~pathlib.Path`.

        Args:
            **args: Arguments passed to :func:`fsspec.open`.
            **kwargs: Keyword arguments passed to :func:`fsspec.open`.

        Returns:
            `io.FileIO`: File-like object.
        """
        return xopen(str(self), *args, **kwargs)

    def joinpath(self, *p: Tuple[str, ...]) -> "xPath":
        """Extend :func:`xjoin` to support argument of type :obj:`~pathlib.Path`.

        Args:
            *p (`tuple` of `str`): Other path components.

        Returns:
            [`xPath`]
        """
        return type(self)(xjoin(self.as_posix(), *p))

    def __truediv__(self, p: str) -> "xPath":
        return self.joinpath(p)

    def with_suffix(self, suffix):
        main_hop, *rest_hops = str(self).split("::")
        if is_local_path(main_hop):
            return type(self)(str(super().with_suffix(suffix)))
        return type(self)("::".join([type(self)(PurePosixPath(main_hop).with_suffix(suffix)).as_posix()] + rest_hops))


def _as_str(path: Union[str, Path, xPath]):
    return str(path) if isinstance(path, xPath) else str(xPath(str(path)))
