from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from torch import nn

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.logging import log_rank
from nanotron.models.base import NanotronModel
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import TensorParallelRowLinear
from nanotron.utils import get_parameter_and_parent_module

logger = logging.get_logger(__name__)


def create_tied_parameter(
    parameter: nn.Parameter,
    name: str,
    global_ranks: Tuple[int, ...],
    reduce_op: Optional[dist.ReduceOp],
    root_module: nn.Module,
) -> NanotronParameter:
    if not isinstance(parameter, NanotronParameter):
        parameter = NanotronParameter(tensor=parameter)
    parameter.mark_as_tied(name=name, global_ranks=global_ranks, reduce_op=reduce_op, root_module=root_module)
    return parameter


def tie_parameters(
    root_module: nn.Module,
    ties: List[Tuple[str, Tuple[int, ...]]],
    parallel_context: ParallelContext,
    reduce_op: Optional[dist.ReduceOp],
):
    """
    Tie parameters.
    Within a single device, tied parameters are replaced with a single Parameter
    Across devices, we add metadata to Parameters that require extra synchronization.

    :param root_module: nn.Module
    :param ties: List[Tuple[str, Tuple[int, ...]]]: a tie is (param_target, global_ranks)
    :param parallel_context: ParallelContext
    :return:
    """
    if len(ties) < 1:
        raise ValueError("Can't tie nothing")

    # TODO @thomasw21: When we support Zero3 this isn't true anymore
    dp_ranks = tuple(
        sorted(
            {
                parallel_context.get_local_ranks(world_rank=global_rank)[1]
                for _, global_ranks in ties
                for global_rank in global_ranks
            }
        )
    )
    assert (
        len(dp_ranks) == 1
    ), f"Tying weights has to happen with a replica of a model. Got the ranks from the following replicas: {dp_ranks}"

    name = ties[0][0]
    global_ranks = tuple(sorted(set().union(*(tie[1] for tie in ties))))

    new_param = None
    world_rank = dist.get_rank(parallel_context.world_pg)
    for tie_target, tie_model_ranks in ties:
        if world_rank not in tie_model_ranks:
            continue

        param, parent_module, param_name = get_parameter_and_parent_module(target=tie_target, root_module=root_module)

        # If they are physically in the same device, then we tie them
        if new_param is None:
            new_param = create_tied_parameter(
                parameter=param, name=name, global_ranks=global_ranks, reduce_op=reduce_op, root_module=root_module
            )

        # Re-assign it to the original name. We assign the raw tensor instead of the parameter since we moved it already.
        setattr(parent_module, param_name, new_param)


def create_pg_for_tied_weights(root_module: nn.Module, parallel_context: ParallelContext):
    """Tied weights are tied across specific set of global ranks, we use this method to create process groups for each difference set of global ranks"""
    group_ranks = {
        param.get_tied_info().global_ranks
        for name, param in root_module.named_parameters()
        if isinstance(param, NanotronParameter) and param.is_tied
    }

    world_group_ranks = [None] * parallel_context.world_pg.size()
    dist.all_gather_object(world_group_ranks, group_ranks, group=parallel_context.world_pg)
    all_group_ranks = sorted(
        set().union(*world_group_ranks),
    )

    for global_ranks in all_group_ranks:
        if global_ranks not in parallel_context.world_ranks_to_pg:
            parallel_context.world_ranks_to_pg[global_ranks] = dist.new_group(global_ranks)


def get_tied_id_to_param(
    parameters: List[NanotronParameter], root_module: nn.Module
) -> Dict[Tuple[str, Tuple[int, ...]], NanotronParameter]:
    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in root_module.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(root_module)] = ""
    return {
        (
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix),
            param.get_tied_info().global_ranks,  # TODO @nouamane: merge groups which tie the same parameter
        ): param
        for param in parameters
        if param.is_tied
    }


def sync_tied_weights_gradients(
    module: nn.Module,  # TODO: NanotronModel
    parallel_context: ParallelContext,
    grad_accumulator: Optional[GradientAccumulator],
):
    tied_id_to_param = get_tied_id_to_param(
        parameters=[param for param in module.parameters() if param.requires_grad], root_module=module
    )

    # Only first and last rank should print the warning
    for rank in [0, parallel_context.world_pg.size() - 1]:
        log_rank(
            f"[Debug Tied Weights] Syncing the following tied weights: {tied_id_to_param.keys()}",
            logger=logger,
            level=logging.DEBUG,
            group=parallel_context.world_pg,
            rank=rank,
        )

    # Group tensors to reduce by process groups
    # Important to use ordered dict in order to be synchronized across all ranks
    group_ranks_and_reduce_op_to_tensors_to_reduce = OrderedDict()
    for (name, group_ranks), tied_param in sorted(tied_id_to_param.items(), key=lambda x: x[0]):
        tied_info = tied_param.get_tied_info()
        # Some weights don't require any syncing, because they are by design synchronised
        if tied_info.reduce_op is None:
            continue

        if grad_accumulator is not None:
            tied_grad = grad_accumulator.get_grad_buffer(name=name)
        else:
            tied_grad = tied_param.grad
        log_rank(
            f"Syncing tied weights {name} across ranks {group_ranks} ...",
            logger=logger,
            level=logging.DEBUG,
            group=parallel_context.world_ranks_to_pg[group_ranks],
            rank=0,
        )
        key = (group_ranks, tied_info.reduce_op)
        if key in group_ranks_and_reduce_op_to_tensors_to_reduce:
            group_ranks_and_reduce_op_to_tensors_to_reduce[(group_ranks, tied_info.reduce_op)].append(tied_grad)
        else:
            group_ranks_and_reduce_op_to_tensors_to_reduce[(group_ranks, tied_info.reduce_op)] = [tied_grad]

    for (group_ranks, reduce_op), tensors in group_ranks_and_reduce_op_to_tensors_to_reduce.items():
        dist.all_reduce_coalesced(tensors=tensors, op=reduce_op, group=parallel_context.world_ranks_to_pg[group_ranks])


def mark_unsharded_params_as_tied_across_tp(
    model: NanotronModel, parallel_context: ParallelContext, parallel_config: ParallelismArgs
):
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(param, NanotronParameter):
                # We skip tying if param already tied or sharded along tp
                if param.is_tied:
                    continue

                if param.is_sharded:
                    sharded_info = param.get_sharded_info()
                    if sharded_info.is_tp_sharded(parallel_context=parallel_context):
                        continue

            if isinstance(module, TensorParallelRowLinear) and "bias" == param_name:
                # bias for TensorParallelRowLinear only exists on TP=0 so we don't need to tie it
                continue

            shared_weights = [
                (
                    name,
                    # sync across TP group
                    tuple(sorted(dist.get_process_group_ranks(parallel_context.tp_pg))),
                )
            ]

            if parallel_config is None or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
                # We add `reduce_op=None` in order to signal that the weight are synced by design without needing to reduce
                # when TP=2 we have LN that is duplicated across TP, so by design it's tied
                reduce_op = None
            else:
                reduce_op = dist.ReduceOp.SUM

            tie_parameters(
                root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=reduce_op
            )


def mark_unsharded_params_as_tied_across_expert(
    model: NanotronModel, parallel_context: ParallelContext, parallel_config: ParallelismArgs
):
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(param, NanotronParameter):
                # We skip tying if param already tied or sharded along expert
                if param.is_tied:
                    continue

                if param.is_sharded:
                    sharded_info = param.get_sharded_info()
                    if sharded_info.is_expert_sharded(parallel_context):
                        continue

            shared_weights = [
                (
                    name,
                    # sync across expert group
                    tuple(sorted(dist.get_process_group_ranks(parallel_context.expert_pg))),
                )
            ]

            # Besides MoE block which sees shards tokens, the rest of the model sees the full tokens
            # so we don't need to reduce the gradients across expert group
            reduce_op = None

            tie_parameters(
                root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=reduce_op
            )
