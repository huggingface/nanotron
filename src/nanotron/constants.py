import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.4")

PY_VERSION = parse(platform.python_version())

#### FOR SERIALIZATION ####

CHECKPOINT_FILE_NAME = "checkpoint_metadata.json"
MODEL_CONFIG_FILE_NAME = "model_config.json"


# NOTE: hacky, remove after working
IS_FP8: bool = True

NN_STATES = None
CONFIG = None

TRACKING_FP8_PARAM = {}

PARAM_ID_TO_PARAM_NAMES = None

ITERATION_STEP: int = 1
