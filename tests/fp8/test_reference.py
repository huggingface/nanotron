from copy import deepcopy

from torch import nn
from msamp.nn import LinearReplacer
from msamp.common.dtype import Dtypes as MS_Dtypes
from utils import convert_to_fp8_module

def test_backward():
    HIDDEN_SIZE = 16
    N_STEPS = 10
    
    ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    msamp_linear = deepcopy(ref_linear)
    msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)
    
    linear = deepcopy(ref_linear)
    linear = convert_to_fp8_module(linear)
    
