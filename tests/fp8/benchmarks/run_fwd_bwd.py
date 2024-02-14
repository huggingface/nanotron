from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam
from msamp.nn import LinearReplacer
from msamp.common.dtype import Dtypes as MS_Dtypes
from msamp.optim import LBAdam


HIDDEN_SIZE = 16
ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
msamp_linear = deepcopy(ref_linear)
msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)

input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device='cuda')

ref_output = ref_linear(input)
ref_output.sum().backward()

msamp_output = msamp_linear(input)
msamp_output.sum().backward()


torch.testing.assert_close(msamp_output.float(), ref_output, rtol=0, atol=0.1)
torch.testing.assert_close(msamp_linear.bias.grad, ref_linear.bias.grad, rtol=0, atol=0.1)
torch.testing.assert_close(msamp_linear.weight.grad.float(), ref_linear.weight.grad, rtol=0, atol=0.1)
