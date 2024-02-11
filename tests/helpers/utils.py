import contextlib
import os
import random
import re
import uuid
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.cuda
from nanotron.parallel import ParallelContext
from packaging import version
from torch.distributed.launcher import elastic_launch


def available_gpus():
    if not torch.cuda.is_available():
        return 0

    device_properties = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]

    # We filter out
    blacklisted_gpu_names = {"NVIDIA DGX Display"}
    device_properties = [property_ for property_ in device_properties if property_.name not in blacklisted_gpu_names]

    # TODO @thomasw21: Can we do this cross node
    return len(device_properties)


# from https://stackoverflow.com/a/34333710/9201239
@contextlib.contextmanager
def mock_os_environ(remove_keys: List[str] = None, update_key_values: Dict[str, Any] = None):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.
    The ``os.environ`` dictionary is updated in-place so that the modification is sure to work in all situations.
    Args:
      remove_keys: Environment variables to remove.
      update_key_values: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update_key_values = update_key_values or {}
    remove_keys = remove_keys or []

    update_keys = set(update_key_values.keys())
    remove_keys = set(remove_keys)
    assert remove_keys.isdisjoint(update_keys)

    stomped = (update_keys | remove_keys) & set(env.keys())
    reverse_change = {
        # Environment variables and values to restore on exit.
        **{k: env[k] for k in update_keys & stomped},
        # Environment variables and values to remove on exit.
        **{k: env[k] for k in remove_keys & stomped},
    }

    try:
        env.update(update_key_values)
        for k in remove_keys:
            env.pop(k, None)
        yield
    finally:
        env.update(reverse_change)


class init_process_and_run_func:
    """Initialize distributed process groups and run function."""

    def __init__(self, func, args, kwargs, tp: int, dp: int, pp: int):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.__name__ = self.__class__.__name__
        self.__qualname__ = self.__class__.__qualname__

    def __call__(self):
        with mock_os_environ(update_key_values={"WORLD_SIZE": f"{self.tp * self.dp * self.pp}"}):
            # NOTE: we use a different random RNG, so that each unit tests don't generate the same port
            # seed = random.randint(0, 9999)
            # with torch.random.fork_rng(devices=["cuda"]):
            # from nanotron.utils import find_free_port

            import time

            random.seed(time.time())

            # torch.manual_seed(seed)
            # port = find_free_port()
            parallel_context = ParallelContext(
                data_parallel_size=self.dp, pipeline_parallel_size=self.pp, tensor_parallel_size=self.tp
            )

            assert "parallel_context" not in self.kwargs
            self.kwargs["parallel_context"] = parallel_context

            self.func(*self.args, **self.kwargs)


def init_distributed(tp: int, dp: int, pp: int):
    def _init_distributed(func):
        """Wrapper to help initialize distributed nanotron.

        :param func: parallel function that runs on all the process, it requires one of its keyword argument to be "parallel_context"
        """
        nb_gpus = tp * dp * pp
        run_id = uuid.uuid4()

        config = torch.distributed.launcher.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=nb_gpus,
            rdzv_backend="c10d",
            rdzv_configs={"timeout": 60},
            # Setting port to `0` allows `torch` to randomly pick a port: https://pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker
            # Works only for single node workload.
            rdzv_endpoint="localhost:0",
            run_id=str(run_id),
            max_restarts=0,
            # TODO @thomasw21: Tune as we increase the number of tests
            monitor_interval=1,
            tee=torch.distributed.elastic.multiprocessing.Std(3),
        )

        def wrapper(*args, **kwargs):
            return elastic_launch(
                config=config,
                entrypoint=init_process_and_run_func(func, tp=tp, dp=dp, pp=pp, args=args, kwargs=kwargs),
            )()

        return wrapper

    return _init_distributed


def is_dict_equal(first: Dict, second: Dict, sub_paths: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """Returns True or False if the dictionaries match, and an additional message when it's False"""
    if sub_paths is None:
        sub_paths = []

    first_keys = set(first.keys())
    second_keys = set(second.keys())
    if first_keys != second_keys:
        return False, f"Keys don't match in {'.'.join(sub_paths)}.\nCur: {first_keys}\nRef: {second_keys}"
    for key in first_keys:
        first_elt = first[key]
        second_elt = second[key]

        if isinstance(first_elt, dict):
            if not isinstance(second_elt, dict):
                return (
                    False,
                    f"Object types don't match in {'.'.join(sub_paths +  [str(key)])}.\nCur: {first_elt}\nRef: {second_elt}",
                )
            match, msg = is_dict_equal(first_elt, second_elt, sub_paths=sub_paths + [str(key)])
            if match is False:
                return False, msg
        elif isinstance(first_elt, torch.Tensor):
            if not isinstance(second_elt, torch.Tensor):
                return (
                    False,
                    f"Object types don't match in {'.'.join(sub_paths +  [str(key)])}.\nCur: {first_elt}\nRef: {second_elt}",
                )
            try:
                torch.testing.assert_close(
                    first_elt,
                    second_elt,
                    atol=0.0,
                    rtol=0.0,
                    msg=lambda msg: f"Tensor at {'.'.join(sub_paths + [str(key)])} don't match.\nCur: {first_elt}\nRef: {second_elt}\n{msg}",
                )
            except AssertionError as error:
                return False, error.args[0]
        else:
            if first_elt != second_elt:
                return (
                    False,
                    f"Objects at key {'.'.join(sub_paths + [str(key)])} don't match.\nCur: {first_elt}\nRef: {second_elt}",
                )

    return True, None


def get_all_3d_configurations(gpus: int) -> List[Tuple[int, int, int]]:
    """Given a number of gpus, we want all 3d configurations possible such that pp * dp * tp = gpus"""
    result = []
    for tp in range(1, gpus + 1):
        if gpus % tp != 0:
            continue
        gpus_left_after_tp = gpus // tp
        for dp in range(1, gpus_left_after_tp + 1):
            if gpus_left_after_tp % dp != 0:
                continue
            gpus_left_after_dp = gpus_left_after_tp // dp
            for pp in range(1, gpus_left_after_dp + 1):
                if gpus_left_after_dp % pp != 0:
                    continue
                if tp * dp * pp == gpus:
                    result.append((pp, dp, tp))
    return result


def rerun_if_address_is_in_use(max_try: int = 200):
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
