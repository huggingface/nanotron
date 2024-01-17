from contextlib import contextmanager

from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.state import PipelineBatchState
from torch import nn as torch_nn


@contextmanager
def attach_pipeline_state_to_model(model: torch_nn.Module, pipeline_state: PipelineBatchState):
    """Attach the pipeline state to all the PipelineBlocks within `model`"""
    old_pipeline_states = []

    # Set new
    for name, module in model.named_modules():
        if not isinstance(module, PipelineBlock):
            continue

        old_pipeline_state = module.pipeline_state
        assert old_pipeline_state is None, "We never replace an old pipeline engine, we just set one when there's none"

        old_pipeline_states.append((old_pipeline_state, module))

        module.set_pipeline_state(pipeline_state)

    try:
        yield
    finally:
        for old_pipeline_state, module in old_pipeline_states:
            module.set_pipeline_state(old_pipeline_state)
