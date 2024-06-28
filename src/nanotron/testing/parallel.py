import os
import re
from inspect import signature
from typing import Callable

import torch.cuda
import torch.multiprocessing as mp
from nanotron.parallel import ParallelContext
from packaging import version


def global_wrapper(rank, func, tp, pp, dp, port, kwargs):
    def setup_dist_env(rank, world_size, port):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        # NOTE: since we do unit tests in a
        # single node => this is fine!
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp * pp * dp
    setup_dist_env(rank, world_size, port)
    parallel_context = ParallelContext(data_parallel_size=dp, pipeline_parallel_size=pp, tensor_parallel_size=tp)
    func(parallel_context, **kwargs)


def init_distributed(tp: int, dp: int, pp: int):
    def _init_distributed(func):
        def wrapper(**kwargs):
            from nanotron.utils import find_free_port

            world_size = tp * pp * dp
            port = find_free_port()

            # Note that kwargs needs to be passed as part of args in a way that can be unpacked
            args = (func, tp, pp, dp, port, kwargs)
            mp.spawn(global_wrapper, args=args, nprocs=world_size)

        return wrapper

    return _init_distributed


def rerun_if_address_is_in_use(max_try: int = 500):
    """
    This function reruns a wrapped function if "address already in use" occurs
    in testing spawned with torch.multiprocessing

    Credits: https://github.com/hpcaitech/ColossalAI/blob/adae123df3badfb15d044bd416f0cf29f250bc86/colossalai/testing/utils.py#L157

    Usage::

        @rerun_if_address_is_in_use()
        def test_something():
            ...

    """
    # check version
    torch_version = version.parse(torch.__version__)
    assert torch_version.major >= 1

    # only torch >= 1.8 has ProcessRaisedException
    if torch_version >= version.parse("1.8.0"):
        exception = torch.multiprocessing.ProcessRaisedException
    else:
        exception = Exception

    func_wrapper = rerun_on_exception(exception_type=exception, pattern=".*Address already in use.*", max_try=max_try)
    return func_wrapper


def rerun_on_exception(exception_type: Exception = Exception, pattern: str = None, max_try: int = 10) -> Callable:
    """
    A decorator on a function to re-run when an exception occurs.

    Credits: https://github.com/hpcaitech/ColossalAI/blob/adae123df3badfb15d044bd416f0cf29f250bc86/colossalai/testing/utils.py#L71

    Usage::

        # rerun for all kinds of exception
        @rerun_on_exception()
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for RuntimeError only
        @rerun_on_exception(exception_type=RuntimeError)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for maximum 10 times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, max_try=10)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for infinite times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, max_try=None)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun only the exception message is matched with pattern
        # for infinite times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, pattern="^Address.*$")
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

    Args:
        exception_type (Exception, Optional): The type of exception to detect for rerun
        pattern (str, Optional): The pattern to match the exception message.
            If the pattern is not None and matches the exception message,
            the exception will be detected for rerun
        max_try (int, Optional): Maximum reruns for this function. The default value is 5.
            If max_try is None, it will rerun forever if exception keeps occurring
    """

    def _match_lines(lines, pattern):
        for line in lines:
            if re.match(pattern, line):
                return True
        return False

    def _wrapper(func):
        def _run_until_success(*args, **kwargs):
            try_count = 0
            assert max_try is None or isinstance(
                max_try, int
            ), f"Expected max_try to be None or int, but got {type(max_try)}"

            while max_try is None or try_count < max_try:
                try:
                    try_count += 1
                    ret = func(*args, **kwargs)
                    return ret
                except exception_type as e:
                    error_lines = str(e).split("\n")
                    if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):

                        print("Exception is caught, retrying...")
                        # when pattern is not specified, we always skip the exception
                        # when pattern is specified, we only skip when pattern is matched
                        continue
                    else:
                        print("Maximum number of attempts is reached or pattern is not matched, no more retrying...")
                        raise e

        # Override signature
        # otherwise pytest.mark.parameterize will raise the following error:
        # function does not use argument xxx
        sig = signature(func)
        _run_until_success.__signature__ = sig

        return _run_until_success

    return _wrapper
