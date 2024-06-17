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

# TODO(xrsrke): delete all these shit after fixing the bug
DEBUG_FP8_INPUT: Optional[torch.Tensor] = None
DEBUG_FP8_INPUT_AFTER_QUANT: Optional[torch.Tensor] = None
DEBUG_FP8_WEIGHT: Optional[torch.Tensor] = None
DEBUG_FP8_BIAS: Optional[torch.Tensor] = None
DEBUG_FP8_OUTPUT: Optional[torch.Tensor] = None
DEBUG_FP8_OUTPUT_COPY = None
DEBUG_FP8_OUTPUT_AFTER_ALL_REDUCE: Optional[torch.Tensor] = None
DEBUG_FP8_OUTPUT_FOR_ACCUMULATION: Optional[torch.Tensor] = None


DEBUB_FP8_INPUT_THAT_WORK: Optional[torch.Tensor] = None
DEBUB_FP8_WEIGHT_THAT_WORK: Optional[torch.Tensor] = None
DEBUG_FP8_OUTPUT_DIRECTLY_FROM_FP8_THAT_WORK = None


DEBUG_FP8_GRAD_OUTPUT: Optional[torch.Tensor] = None
DEBUG_FP8_GRAD_INPUT: Optional[torch.Tensor] = None
DEBUG_FP8_GRAD_WEIGHT: Optional[torch.Tensor] = None
DEBUG_FP8_GRAD_WEIGHT_BEFORE_RESHAPE: Optional[torch.Tensor] = None
DEBUG_FP8_GRAD_WEIGHT_AFTER_RESHAPE: Optional[torch.Tensor] = None

DEBUG_FP8_GRAD_BIAS: Optional[torch.Tensor] = None

REF_INPUT: Optional[torch.Tensor] = None
REF_WEIGHT: Optional[torch.Tensor] = None
REF_BIAS: Optional[torch.Tensor] = None
REF_OUTPUT: Optional[torch.Tensor] = None
REF_MANUAL_OUTPUT: Optional[torch.Tensor] = None

REF_GRAD_OUTPUT: Optional[torch.Tensor] = None
REF_GRAD_INPUT: Optional[torch.Tensor] = None
REF_GRAD_WEIGHT: Optional[torch.Tensor] = None
REF_GRAD_BIAS: Optional[torch.Tensor] = None

ITERATION_STEP: int = 1
