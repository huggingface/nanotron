import contextlib
import signal
from typing import Optional

from nanotron import distributed as dist


@contextlib.contextmanager
def assert_fail_with(exception_class, error_msg: Optional[str] = None):
    try:
        yield
    except exception_class as e:
        if error_msg is None:
            return
        if error_msg == str(e):
            return
        else:
            raise AssertionError(f'Expected message to be "{error_msg}", but got "{str(e)}" instead.')
    except Exception as e:
        raise AssertionError(f"Expected {exception_class} to be raised, but got: {type(e)} instead:\n{e}")
    raise AssertionError(f"Expected {exception_class} to be raised, but no exception was raised.")


@contextlib.contextmanager
def assert_fail_except_rank_with(
    exception_class, rank_exception: int, pg: dist.ProcessGroup, error_msg: Optional[str] = None
):
    try:
        yield
    except exception_class as e:
        if rank_exception == dist.get_rank(pg):
            raise AssertionError(f"Expected rank {rank_exception} to not raise {exception_class}.")
        else:
            if error_msg is None:
                return
            if error_msg == str(e):
                return
            else:
                raise AssertionError(f'Expected message to be "{error_msg}", but got "{str(e)}" instead.')

    except Exception as e:
        raise AssertionError(f"Expected {exception_class} to be raised, but got: {type(e)} instead:\n{e}")
    if dist.get_rank(pg) != rank_exception:
        raise AssertionError(f"Expected {exception_class} to be raised, but no exception was raised.")


@contextlib.contextmanager
def timeout_after(ms=500):
    """Timeout context manager."""

    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out after {ms} ms.")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, ms / 1000)
    try:
        yield
    finally:
        signal.alarm(0)
