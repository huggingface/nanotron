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
LOG_STATE_INTERVAL = 500
CONFIG = None
