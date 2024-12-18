import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.4")

PY_VERSION = parse(platform.python_version())

#### FOR SERIALIZATION ####

CHECKPOINT_FILE_NAME = "checkpoint_metadata.json"
MODEL_CONFIG_FILE_NAME = "model_config.json"


# TODO(xrsrke): remove this shit
ITERATION_STEP = 1
# TODO(xrsrke): refactor to training stage,
# keep it in the same class as iteration_step
CONFIG = None

is_ready_to_log = False

# TODO(xrsrke): refactor
CPU_WEIGHTS = {}
ACCUM_GRADS = {}
