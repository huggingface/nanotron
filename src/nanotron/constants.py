import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.4")

PY_VERSION = parse(platform.python_version())

#### FOR SERIALIZATION ####

CHECKPOINT_FILE_NAME = "checkpoint_metadata.json"
MODEL_CONFIG_FILE_NAME = "model_config.json"


# TODO(xrsrke): remove this shit
ITERATION_STEP = 1
CONFIG = None
