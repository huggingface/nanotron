import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.2")

PY_VERSION = parse(platform.python_version())
DEBUG_PATH = "./debug/nn_states_with_bs_2/acts/"
