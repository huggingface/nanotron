import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.4")

PY_VERSION = parse(platform.python_version())

#### FOR SERIALIZATION ####

CHECKPOINT_FILE_NAME = "checkpoint_metadata.json"
MODEL_CONFIG_FILE_NAME = "model_config.json"

# MoE specific
EXPERT_PARAM_NAMES = [
    # NOTE: nanotron's moe modeling
    "mlp.experts.merged_down_proj",
    "mlp.experts.merged_gate_up_proj"
    # NOTE: TE's moe modeling
    "experts.linear_fc1",
    "experts.linear_fc2",
]
