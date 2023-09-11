import contextlib
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch.cuda
from torch.distributed.launcher import elastic_launch

from brrr.core.process_groups_initializer import get_process_groups


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
            dpg = get_process_groups(
                data_parallel_size=self.dp,
                pipeline_parallel_size=self.pp,
                tensor_parallel_size=self.tp,
            )

            assert "dpg" not in self.kwargs
            self.kwargs["dpg"] = dpg

            self.func(*self.args, **self.kwargs)


def init_distributed(tp: int, dp: int, pp: int):
    def _init_distributed(func):
        """Wrapper to help initialize distributed brrr.

        :param func: parallel function that runs on all the process, it requires one of its keyword argument to be "dpg"
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
