from contextlib import contextmanager

import pytest


@contextmanager
def fail_if_expect_to_fail(expect_to_fail: bool):
    try:
        yield
    except AssertionError as e:
        if expect_to_fail is True:
            pytest.xfail("Failed successfully")
        else:
            raise e
