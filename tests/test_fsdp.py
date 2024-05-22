import torch
import torch.nn.functional as F
from helpers.utils import init_distributed
from nanotron.parallel import ParallelContext
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class DummyModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(10, 20, bias=False).to(dtype=dtype)
        self.fc2 = nn.Linear(20, 2, bias=False).to(dtype=dtype)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def _test_fsdp_with_afab(parallel_context: ParallelContext, accumulation_steps: int):

    model = DummyModel().to("cuda")
    FSDP(model, process_group=parallel_context.dp_pg)
    print()


if __name__ == "__main__":
    init_distributed(tp=2, dp=2, pp=1)(_test_fsdp_with_afab)(accumulation_steps=1)
