import functools
import inspect
from contextlib import ExitStack, contextmanager
from typing import Callable, ContextManager, List, Optional

import torch
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint

from brrr.core import distributed as dist
from brrr.core.distributed import get_global_rank


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


# TODO @thomasw21: Should this option override user defined options? Maybe not ... right now it does.
@contextmanager
def init_on_device_and_dtype(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
):
    """
    A context manager under which models are initialized with all parameters on the specified device.
    Args:
        device (`torch.device` defaults to `cpu`):
            Device to initialize all parameters on.
        dtype (`torch.dtype` defaults to `torch.float`):
            Dtype to initialize all parameters on.
        include_buffers (`bool`, defaults to `False`):
            Whether or not to also default all buffers constructors given previous arguments.
    Example:
    ```python
    import torch.nn as nn
    from accelerate import init_on_device
    with init_on_device_and_dtype(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            if isinstance(param, DTypeInvariantTensor):
                # if param is DTypeInvariantTensor we should avoid updating it
                param.data = param.data.to(device)
            else:
                param.data = param.data.to(device, dtype)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            if isinstance(buffer, DTypeInvariantTensor):
                # if buffer is DTypeInvariantTensor we should avoid updating it
                buffer.data = buffer.data.to(device)
            else:
                module._buffers[name] = module._buffers[name].to(device, dtype)

    # Patch tensor creation
    tensor_constructors_to_patch = {
        torch_function_name: getattr(torch, torch_function_name)
        for torch_function_name in ["empty", "zeros", "ones", "full"]
    }

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            kwargs["dtype"] = dtype
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


class DTypeInvariantTensor(torch.Tensor):
    """DTypeInvariantTensor is a subclass of torch.Tensor that disallows modification of its dtype. Note that the data
    and other attributes of the tensor can still be modified."""

    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(cls, *args, **kwargs)
        return tensor

    def detach(self, *args, **kwargs):
        raise RuntimeError("Cannot detach an DTypeInvariantTensor")

    def to(self, *args, **kwargs):
        if "dtype" in kwargs or any(isinstance(arg, torch.dtype) for arg in args):
            raise RuntimeError("Cannot change the type of an DTypeInvariantTensor")
        else:
            return super().to(*args, **kwargs)

    def type(self, *args, **kwargs):
        raise RuntimeError("Cannot change the type of an DTypeInvariantTensor")

    def float(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to float")

    def double(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to double")

    def half(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to half")

    def long(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to long")

    def int(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to int")

    def short(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to short")

    def char(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to char")

    def byte(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to byte")

    def bool(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to bool")

    def bfloat16(self, *args, **kwargs):
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to bfloat16")


@contextmanager
def main_rank_first(group: dist.ProcessGroup):
    is_main = dist.get_rank(group) == 0
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


def assert_tensor_synced_across_pg(
    tensor: torch.Tensor,
    pg: dist.ProcessGroup,
    msg: Optional[Callable[[str], str]] = None,
    reference_rank: int = 0,
):
    """Assert that `tensor` is synced across `pg` with reference rank. Note that this always passes for reference rank"""
    if dist.get_rank(pg) == reference_rank:
        reference_tensor = tensor
    else:
        reference_tensor = torch.empty_like(tensor)
    dist.broadcast(
        reference_tensor,
        src=get_global_rank(group=pg, group_rank=reference_rank),
        group=pg,
    )

    # TODO @nouamane: Getting Greatest absolute difference: 4.6e-10 at large scale when syncing tied weights
    torch.testing.assert_close(tensor, reference_tensor, msg=msg)


# TODO @nouamanetazi: remove this with SANITY_CHECKS
@contextmanager
def assert_fail_except_rank_with(exception_class, rank_exception, pg):
    try:
        yield
    except exception_class:
        if rank_exception == dist.get_rank(pg):
            raise AssertionError(f"Expected rank {rank_exception} to not raise {exception_class}.")
        else:
            return

    except Exception as e:
        raise AssertionError(f"Expected {exception_class} to be raised, but got {type(e)} instead:\n{e}")
    if dist.get_rank(pg) != rank_exception:
        raise AssertionError(f"Expected {exception_class} to be raised, but no exception was raised.")


def get_untyped_storage(tensor: torch.Tensor) -> torch.UntypedStorage:
    if version.parse(torch.__version__) >= version.parse("2.0"):
        return tensor.untyped_storage()
    else:
        return tensor.storage().untyped()
