# import os
import random
import socket
from functools import partial
from typing import Callable

import pytest
import torch
import torch.multiprocessing as mp

# NOTE: because these tests run too slow in GitHub Actions
# skip_in_github_actions = pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Test skipped in GitHub Actions")
skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError as e:
            raise e


def spawn(func: Callable, world_size: int = 1, **kwargs):
    if kwargs.get("port") is None:
        port = find_free_port()
    else:
        port = kwargs["port"]
        kwargs.pop("port")

    wrapped_func = partial(func, world_size=world_size, port=port, **kwargs)

    mp.spawn(wrapped_func, nprocs=world_size)
