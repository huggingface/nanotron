import platform
from typing import Optional

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.2")

PY_VERSION = parse(platform.python_version())

# OPTIMIZER_CONFIG_FILE_NAME = "optimizer_config.json"
OPTIMIZER_CKP_PATH = "{}/optimizer/optimizer_config.json"

LR_SCHEDULER_CKP_PATH = "{}/lr_scheduler"
METADATA_CKP_PATH = "{}/checkpoint_metadata.json"

NEEDLE = None

GLOBAL_STEP: Optional[int] = None
LOG_STATE_INTERVAL = 2000
IS_RANK_TO_MONITOR = None
CONFIG = None

TRAINING_CONFIG = None


DEBUG_PATH = "./debug/nn_states_with_bs_2_and_transpose_qkv/acts/"

MONITOR_STATE_PATH = "/fsx/phuc/projects/nanotron/debug/runs"

BALANCE_FACTOR_STD = {}
