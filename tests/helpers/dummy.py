from math import ceil
from typing import Union

import torch
from nanotron import distributed as dist
from nanotron.models import init_on_device_and_dtype
from nanotron.optim.base import BaseOptimizer
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tied_parameters import tie_parameters
from nanotron.parallel.utils import initial_sync
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class DummyModel(nn.Module):
    def __init__(
        self,
        p2p: P2P,
    ):
        super().__init__()
        self.p2p = p2p
        self.mlp = nn.Sequential(
            *(
                nn.ModuleDict(
                    {
                        "linear": PipelineBlock(
                            p2p=p2p,
                            module_builder=nn.Linear,
                            module_kwargs={"in_features": 10, "out_features": 10},
                            module_input_keys={"input"},
                            module_output_keys={"output"},
                        ),
                        "activation": PipelineBlock(
                            p2p=p2p,
                            module_builder=nn.Sigmoid if pp_rank < p2p.pg.size() - 1 else nn.Identity,
                            module_kwargs={},
                            module_input_keys={"input"},
                            module_output_keys={"output"},
                        ),
                    }
                )
                for pp_rank in range(p2p.pg.size())
            )
        )

        self.loss = PipelineBlock(
            p2p=p2p,
            module_builder=lambda: lambda x: x.sum(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )

    def forward(self, x: Union[torch.Tensor, TensorPointer]):
        for non_linear in self.mlp:
            x = non_linear.linear(input=x)["output"]
            x = non_linear.activation(input=x)["output"]
        x = self.loss(x=x)["output"]
        return x


def init_dummy_model(parallel_context: ParallelContext, dtype: torch.dtype = torch.float) -> DummyModel:
    p2p = P2P(pg=parallel_context.pp_pg, device=torch.device("cuda"))
    model = DummyModel(p2p=p2p)

    # Build model using contiguous segments
    pipeline_blocks = [module for name, module in model.named_modules() if isinstance(module, PipelineBlock)]
    with init_on_device_and_dtype(device=torch.device("cuda"), dtype=dtype):
        contiguous_size = ceil(len(pipeline_blocks) / parallel_context.pp_pg.size())
        for i, block in enumerate(pipeline_blocks):
            rank = i // contiguous_size
            block.build_and_set_rank(rank)

    # Sync all parameters that have the same name and that are not sharded across TP.
    for name, param in model.named_parameters():
        if isinstance(param, NanotronParameter) and param.is_sharded:
            continue
        shared_weights = [
            (
                name,
                # sync across TP group
                tuple(sorted(dist.get_process_group_ranks(parallel_context.tp_pg))),
            )
        ]
        tie_parameters(
            root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=dist.ReduceOp.SUM
        )

    initial_sync(model=model, parallel_context=parallel_context)

    if len(list(model.named_parameters())) > 0:
        model = DistributedDataParallel(model, process_group=parallel_context.dp_pg)
    else:
        # No parameters, so no need to use DDP to sync parameters gradients
        model = model

    return model


def init_dummy_optimizer(model: nn.Module, parallel_context: ParallelContext) -> BaseOptimizer:
    optimizer = NamedOptimizer(
        named_params_or_groups=model.named_parameters(), optimizer_builder=lambda params: torch.optim.AdamW(params)
    )

    # Synchronize across dp: basic assumption, already done as nothing in optimizer initialization is stochastic

    return optimizer


def dummy_infinite_data_loader(pp_pg: dist.ProcessGroup, dtype=torch.float, input_pp_rank=0):
    micro_batch_size = 3
    # We assume the first linear is always built on the first rank.
    current_pp_rank = dist.get_rank(pp_pg)
    while True:
        yield {
            "x": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
            if current_pp_rank == input_pp_rank
            else TensorPointer(group_rank=input_pp_rank)
        }
