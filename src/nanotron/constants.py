import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("0.1")

PY_VERSION = parse(platform.python_version())
