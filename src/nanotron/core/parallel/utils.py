import functools
import os


def assert_cuda_max_connections_set_to_1(func):
    flag_is_set_to_1 = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal flag_is_set_to_1
        if flag_is_set_to_1 is None:
            assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1"
            flag_is_set_to_1 = True
        return func(*args, **kwargs)

    return wrapper
