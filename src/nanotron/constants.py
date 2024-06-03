import platform
from typing import Optional

import torch
from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.4")

PY_VERSION = parse(platform.python_version())

#### FOR SERIALIZATION ####

CHECKPOINT_FILE_NAME = "checkpoint_metadata.json"
MODEL_CONFIG_FILE_NAME = "model_config.json"


# NOTE: hacky, remove after working
IS_FP8: bool = True

NN_STATES = None

TRACKING_FP8_PARAM = {}

PARAM_ID_TO_PARAM_NAMES = None

DEBUG_FP8_GRAD_OUTPUT: Optional[torch.Tensor] = None
DEBUG_FP8_GRAD_INPUT: Optional[torch.Tensor] = None
DEBUG_FP8_GRAD_WEIGHT: Optional[torch.Tensor] = None
DEBUG_FP8_GRAD_WEIGHT_BEFORE_RESHAPE: Optional[torch.Tensor] = None
DEBUG_FP8_GRAD_WEIGHT_AFTER_RESHAPE: Optional[torch.Tensor] = None

DEBUG_FP8_GRAD_BIAS: Optional[torch.Tensor] = None

REF_GRAD_OUTPUT: Optional[torch.Tensor] = None
REF_GRAD_INPUT: Optional[torch.Tensor] = None
REF_GRAD_WEIGHT: Optional[torch.Tensor] = None
REF_GRAD_BIAS: Optional[torch.Tensor] = None
