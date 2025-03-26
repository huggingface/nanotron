from dataclasses import dataclass
from typing import Optional

from nanotron.config.utils_config import (
    cast_str_to_pipeline_engine,
)
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode


@dataclass
class ParallelismArgs:
    """Arguments related to TP/PP/DP

    Args:
        dp: Number of DP replicas
        pp: Number of PP stages
        tp: Number of TP replicas
        expert_parallel_size: Number of expert parallel replicas (used only for MoEs)
        pp_engine: Pipeline engine to use between "1f1b" and "afab"
        tp_mode: TP mode to use between "all_reduce" and "reduce_scatter": all_reduce is normal, reduce_scatter activate sequence parallelism
        tp_linear_async_communication: Whether to use async communication in TP linear layers
        recompute_layer: Whether to recompute each Transformer layer to save memory.
    """

    dp: int
    pp: int
    tp: int
    pp_engine: Optional[PipelineEngine] = None
    tp_mode: Optional[TensorParallelLinearMode] = None
    tp_linear_async_communication: Optional[bool] = None
    recompute_layer: bool = False
    moe_layer_recompute: bool = False

    tp_recompute_allgather: bool = True

    expert_parallel_size: int = 1
    context_parallel_size: int = 1

    def __post_init__(self):
        # Conservative defaults
        if self.pp_engine is None:
            self.pp_engine = AllForwardAllBackwardPipelineEngine()
        if self.tp_mode is None:
            self.tp_mode = TensorParallelLinearMode.ALL_REDUCE
        if self.tp_linear_async_communication is None:
            self.tp_linear_async_communication = False

        if isinstance(self.pp_engine, str):
            self.pp_engine = cast_str_to_pipeline_engine(self.pp_engine)
        if isinstance(self.tp_mode, str):
            self.tp_mode = TensorParallelLinearMode[self.tp_mode.upper()]
