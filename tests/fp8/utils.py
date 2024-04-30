import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

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


def set_system_path():
    package = importlib.import_module("nanotron")
    # NOTE:  Path(package.__file__).parent = .../nanotron/src/nanotron
    # we want .../nanotron
    package_path = Path(package.__file__).parent.parent.parent
    sys.path.append(str(package_path))
