import functools
import inspect
import os
import random
import socket
from contextlib import ExitStack, contextmanager
from typing import ContextManager, List, Optional

import torch
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint

from nanotron import distributed as dist


class Singleton(type):
    """
    Singleton metaclass.
    Create objects using this class as the metaclass to enable singleton behaviour.
    For instance:
    ```
    class Logger(metaclass=Singleton):
      ...
    ```
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `transformers` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({[context_manager.gen.__qualname__ for context_manager in self.context_managers]})"


@contextmanager
def main_rank_first(group: Optional[dist.ProcessGroup] = None):
    """Context manager that executes the code in the context with the rank zero of the group going first."""
    is_main = dist.get_rank(group) == 0
    if is_main:
        yield

    dist.barrier(group)

    if not is_main:
        yield


@contextmanager
def local_ranks_zero_first(group: Optional[dist.ProcessGroup] = None):
    """Context manager that executes the code in the context with all the local rank zero of the group going first.
    Useful to run only once per node first (e.g. to create local files, etc)
    """
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        yield

    dist.barrier(group)

    if not is_main:
        yield


def checkpoint_method(attr_name: str):
    """Decorator to checkpoint a method of a class."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _self = args[0]
            checkpoint_activated = getattr(_self, attr_name)
            if checkpoint_activated:
                all_args = list(args)
                signature_params = inspect.signature(func).parameters
                # Parameters are ordered in the function definition order: https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
                for i, (arg_name, arg_value) in enumerate(signature_params.items()):
                    if arg_value.kind in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                        raise NotImplementedError(
                            "Checkpointing of functions with *args or **kwargs is not supported."
                        )
                    if i < len(args):
                        continue
                    if arg_name not in kwargs:
                        assert (
                            arg_value.default is not inspect.Parameter.empty
                        ), f"Missing argument {arg_name} from {kwargs} for {func.__name__}"
                        all_args.append(arg_value.default)
                    else:
                        all_args.append(kwargs[arg_name])
                assert len(all_args) == len(signature_params), f"Missing arguments for {func.__name__}"
                # TODO @nouamanetazi: we pass `self`(which is module) to checkpoint, so it's stored in `ctx.inputs` whereas some other methods create a custom fwd and pass only tensors without `self`. Need to investigate which is better
                return checkpoint(func, *all_args)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_parameter_and_parent_module(target: str, root_module: nn.Module):
    """Retrieve a parameter and its parent module from a dotted target path.

    Args:
        target: Dotted string path to the parameter (e.g. ``"layer.weight"``).
        root_module: The root ``nn.Module`` to traverse.

    Returns:
        A tuple of ``(parameter, parent_module, parameter_name)``.

    Raises:
        AttributeError: If the target attribute does not exist or is not an ``nn.Parameter``.
    """
    module_path, _, param_name = target.rpartition(".")

    mod: torch.nn.Module = root_module.get_submodule(module_path)

    if not hasattr(mod, param_name):
        raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

    param: torch.nn.Parameter = getattr(mod, param_name)

    if not isinstance(param, torch.nn.Parameter):
        raise AttributeError("`" + param_name + "` is not an " "nn.Parameter")

    return param, mod, param_name


def get_untyped_storage(tensor: torch.Tensor) -> torch.UntypedStorage:
    """Return the untyped storage backing a tensor, compatible with PyTorch >= 1.x and >= 2.0.

    Args:
        tensor: The input ``torch.Tensor``.

    Returns:
        The ``torch.UntypedStorage`` backing the tensor.
    """
    if version.parse(torch.__version__) >= version.parse("2.0"):
        return tensor.untyped_storage()
    else:
        return tensor.storage().untyped()


def tensor_from_untyped_storage(untyped_storage: torch.UntypedStorage, dtype: torch.dtype):
    """Construct a tensor from an untyped storage with the given dtype.

    Args:
        untyped_storage: The raw ``torch.UntypedStorage`` to wrap.
        dtype: The desired ``torch.dtype`` of the resulting tensor.

    Returns:
        A 0-dimensional ``torch.Tensor`` backed by the provided storage.
    """
    # TODO @thomasw21: Figure out what's the best Pytorch way of building a tensor from a storage.
    device = untyped_storage.device
    tensor = torch.empty([], dtype=dtype, device=device)
    tensor.set_(source=untyped_storage)
    return tensor


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    """Find a free TCP port on localhost within the given range.

    Randomly samples ports in ``[min_port, max_port]`` until one that is not
    in use is found.

    Args:
        min_port: Lower bound of the port range (inclusive). Defaults to 2000.
        max_port: Upper bound of the port range (inclusive). Defaults to 65000.

    Returns:
        An available port number.
    """
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue
