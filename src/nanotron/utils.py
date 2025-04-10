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
def main_rank_first(group: dist.ProcessGroup):
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
    module_path, _, param_name = target.rpartition(".")

    mod: torch.nn.Module = root_module.get_submodule(module_path)

    if not hasattr(mod, param_name):
        raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

    param: torch.nn.Parameter = getattr(mod, param_name)

    if not isinstance(param, torch.nn.Parameter):
        raise AttributeError("`" + param_name + "` is not an " "nn.Parameter")

    return param, mod, param_name


def get_untyped_storage(tensor: torch.Tensor) -> torch.UntypedStorage:
    if version.parse(torch.__version__) >= version.parse("2.0"):
        return tensor.untyped_storage()
    else:
        return tensor.storage().untyped()


def tensor_from_untyped_storage(untyped_storage: torch.UntypedStorage, dtype: torch.dtype):
    # TODO @thomasw21: Figure out what's the best Pytorch way of building a tensor from a storage.
    device = untyped_storage.device
    tensor = torch.empty([], dtype=dtype, device=device)
    tensor.set_(source=untyped_storage)
    return tensor


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue


def assert_tensor_equal_across_processes(tensor, process_group=None, rtol=1e-5, atol=1e-8):
    """
    Assert that a tensor has the same values across all processes in a distributed process group.

    Args:
        tensor (torch.Tensor): The tensor to check for equality across processes
        process_group: The process group to work on. If None, the default process group is used
        rtol (float): Relative tolerance for floating point comparison
        atol (float): Absolute tolerance for floating point comparison

    Raises:
        AssertionError: If tensors are not equal across processes
    """
    # if not dist.is_initialized():
    #     return

    # Get rank and world size
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    # Skip the check if we only have one process
    # if world_size == 1:
    #     return

    # # Move tensor to CPU for consistent behavior
    # tensor = tensor.cpu()

    # Gather all tensors to rank 0
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=process_group)

    # Each process compares its tensor with all others
    for i in range(world_size):
        is_close = torch.allclose(tensor, tensor_list[i], rtol=rtol, atol=atol)
        if not is_close:
            # Find the first element that's different
            mismatch_mask = ~torch.isclose(tensor, tensor_list[i], rtol=rtol, atol=atol)
            first_mismatch_idx = torch.nonzero(mismatch_mask, as_tuple=True)
            if len(first_mismatch_idx[0]) > 0:
                idx = tuple(dim[0].item() for dim in first_mismatch_idx)
                raise AssertionError(
                    f"Tensor not equal across processes. Process {rank} has tensor value "
                    f"{tensor[idx].item()} at index {idx}, while process {i} has value "
                    f"{tensor_list[i][idx].item()} at the same index."
                )
            else:
                # This case shouldn't typically happen since is_close was False
                raise AssertionError(f"Tensor not equal between processes {rank} and {i}")

    # Synchronize all processes after check
    dist.barrier(group=process_group)
    assert 1 == 1
