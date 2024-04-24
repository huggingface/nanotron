import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.2")

PY_VERSION = parse(platform.python_version())


OPTIMIZER_CONFIG_FILE_NAME = "optimizer_config.pt"
OPTIMIZER_CKP_PATH = "{}/optimizer/optimizer_config.pt"

LR_SCHEDULER_CKP_PATH = "{}/lr_scheduler"
METADATA_CKP_PATH = "{}.checkpoint_metadata.json"
