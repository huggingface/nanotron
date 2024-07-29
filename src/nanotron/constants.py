import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.4")

PY_VERSION = parse(platform.python_version())

#### FOR SERIALIZATION ####

CHECKPOINT_FILE_NAME = "checkpoint_metadata.json"
MODEL_CONFIG_FILE_NAME = "model_config.json"

GLOBAL_STEP = None
LOG_STATE_INTERVAL = 1
IS_RANK_TO_MONITOR = None
CONFIG = None

TRAINING_CONFIG = None


DEBUG_PATH = "./debug/nn_states_with_bs_2_and_transpose_qkv/acts/"

MONITOR_STATE_PATH = "/fsx/phuc/projects/nanotron/debug/runs"
