import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.3")

PY_VERSION = parse(platform.python_version())


CHECKPOINT_FILE_NAME = "checkpoint_metadata.json"
