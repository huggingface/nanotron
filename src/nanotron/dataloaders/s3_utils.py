import os
import re
from collections import deque
from fnmatch import fnmatch
from re import Pattern
from typing import Union

from pathlib import Path

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


# mostly borrowed from datatrove: https://github.com/huggingface/datatrove/blob/main/src/datatrove/io/cloud/s3.py


def _get_s3_path_components(s3_path: Union[str, Path]):
    s3_path = str(s3_path)
    bucket_name, _, prefix = s3_path[len("s3://") :].replace("//", "/").partition(os.sep)
    return bucket_name, prefix


def _get_s3_object(s3_path: Union[str, Path]):
    s3_path = str(s3_path)
    bucket_name, prefix = _get_s3_path_components(s3_path)
    s3_resource = boto3.resource("s3")
    s3_object = s3_resource.Object(bucket_name=bucket_name, key=prefix)
    return s3_object


def _stream_s3_file(s3_path: Union[str, Path], chunk_size, offset):
    s3_path = str(s3_path)
    s3_object = _get_s3_object(s3_path)
    return s3_object.get(Range=f"bytes={chunk_size * offset}-")["Body"].iter_chunks(chunk_size)


def _match_prefix(base_prefix, prefix, match_pattern=None):
    if not match_pattern:
        return True
    path_elements = os.path.relpath(prefix, base_prefix).split(os.path.sep)
    pattern_elements = os.path.normpath(match_pattern).split(os.path.sep)
    match_len = min(len(pattern_elements), len(path_elements))
    return fnmatch(os.path.sep.join(path_elements[:match_len]), os.path.sep.join(pattern_elements[:match_len]))


def _get_s3_file_list(
    s3_path: Union[str, Path], pattern: Union[str, Pattern] = None, recursive: bool = True, max_recursion: int = -1
):
    """Get list of relative paths to files in a cloud folder with a given (optional) pattern

    Args:
        s3_path: path to the cloud folder (e.g. s3://bucket/prefix)
        pattern: optional pattern to filter files (str or re.Pattern)
        recursive: whether to recursively search for files (default: True)
        max_recursion: how many levels to recursively search for files (-1 means no limit)
    """
    s3_path = str(s3_path)

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    s3_client = boto3.client("s3")
    bucket, main_prefix = _get_s3_path_components(s3_path)

    paginator = s3_client.get_paginator("list_objects_v2")
    objects = []
    prefixes = deque()

    prefixes.append((0, main_prefix))
    while prefixes:
        level, prefix = prefixes.popleft()
        for resp in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            if recursive and (max_recursion == -1 or level < max_recursion):
                prefixes.extend([(level + 1, next_prefix["Prefix"]) for next_prefix in resp.get("CommonPrefixes", [])])
            filtered_objects = [x for x in resp.get("Contents", []) if x["Key"] != prefix]
            if pattern is not None:
                filtered_objects = [
                    x for x in filtered_objects if pattern.fullmatch(os.path.relpath(x["Key"], main_prefix))
                ]
            objects.extend([f"s3://{bucket}/{x['Key']}" for x in filtered_objects])
    return sorted(objects)
