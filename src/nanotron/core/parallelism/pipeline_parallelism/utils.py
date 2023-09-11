from torch import nn

from brrr.core.parallelism.pipeline_parallelism.block import PipelineBlock


def get_pp_rank_of(target: str, module: nn.Module):
    """Assuming a model with pipeline blocks, we want to know in which pp rank the module/parameter whose name is `target`"""
    if isinstance(module, PipelineBlock):
        return module.rank

    atoms = target.split(".")
    current_module = module
    for atom in atoms:
        if not hasattr(current_module, atom):
            raise AttributeError(f'{current_module._get_name()} has no attribute `"{atom}"`')

        current_module = getattr(current_module, atom)

        if isinstance(current_module, PipelineBlock):
            return current_module.rank

        if not isinstance(current_module, nn.Module):
            raise AttributeError(f'`"{atom}"` is not an nn.Module')

    raise ValueError(f'`"{target}" is not inside a PipelineBlock and thus does not have a pp_rank')
