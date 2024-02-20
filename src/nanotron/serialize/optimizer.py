import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from nanotron import distributed as dist
from nanotron import optim
from nanotron.optim.zero import (
    ZeroDistributedOptimizer,
    extract_parallel_ranks_from_shard_path,
    find_optim_index_from_param_name,
    get_sliced_tensor,
    merge_dp_shard_in_zero1_optimizer,
)
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.sanity_checks import check_optim_state_in_sync
from nanotron.serialize.metadata import TensorMetadata
from nanotron.serialize.utils import ObjectType, merge_and_shard_tp_tensors


# TODO(xrsrke): take rank instead of parallel_context
def optimizer_filename(parallel_context: ParallelContext, is_zero: bool):
    if is_zero is True:
        return f"{ObjectType.OPTIMIZER.value}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_exp-{dist.get_rank(parallel_context.expert_pg)}-of-{parallel_context.expert_parallel_size}.pt"
    else:
        return f"{ObjectType.OPTIMIZER.value}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}_tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_exp-{dist.get_rank(parallel_context.expert_pg)}-of-{parallel_context.expert_parallel_size}.pt"


def lr_scheduler_filename():
    """The lr_scheduler is the same for all processes."""
    return f"{ObjectType.LR_SCHEDULER.value}.pt"


def save_optimizer(
    optimizer: optim.BaseOptimizer,
    parallel_context: ParallelContext,
    root_folder: Path,
):
    """Saves optimizer states
    - If Zero-0 is used, optimizer states are replicated across all DPs. Only DP-0 saves the states
    - If Zero-1 is used, optimizer states are sharded across all DPs. Each DP saves its own states
    """
    if (not optimizer.inherit_from(optim.ZeroDistributedOptimizer)) and dist.get_rank(parallel_context.dp_pg) > 0:
        # this is Zero-0, so only DP-0 saves the optimizer states
        return

    # TODO @thomasw21: Figure out if I need to save param groups. Right now I'm assuming no as we only store what's trainable
    # TODO @thomasw21: We can probably "rotate" so that every process stores something (maybe doesn't matter if we're I/O bound)
    root_folder = root_folder / "optimizer"
    root_folder.mkdir(exist_ok=True, parents=True)

    if dist.get_rank(parallel_context.world_pg) == 0:
        with open(root_folder / "optimizer_config.json", "w") as fo:
            tp_size = parallel_context.tp_pg.size()
            pp_size = parallel_context.pp_pg.size()
            dp_size = parallel_context.dp_pg.size()
            expert_parallel_size = parallel_context.expert_parallel_size

            config = {
                "type": str(optimizer.__class__.__name__),
                "parallelism": {
                    "tp_size": str(tp_size),
                    "dp_size": str(dp_size),
                    "pp_size": str(pp_size),
                    "expert_parallel_size": str(expert_parallel_size),
                },
                "configs": {},
            }

            if isinstance(optimizer, ZeroDistributedOptimizer):
                # NOTE: in order to serialize, we must save all keys and values as strings
                def convert_to_string(input_item):
                    if isinstance(input_item, dict):
                        return {str(key): convert_to_string(value) for key, value in input_item.items()}
                    elif isinstance(input_item, list):
                        return [convert_to_string(element) for element in input_item]
                    elif isinstance(input_item, tuple):
                        return tuple(convert_to_string(element) for element in input_item)
                    else:
                        return str(input_item)

                # NOTE: if it's a ZeRO-1 optimzier, then we save how the parameters are sharded
                # across data parallel dimension, so that we can reconstruct the optimizer states
                assert optimizer.param_name_to_dp_rank_offsets is not None, "param_name_to_dp_rank_offsets is required"
                config["configs"]["param_name_to_dp_rank_offsets"] = convert_to_string(
                    optimizer.param_name_to_dp_rank_offsets
                )
                # NOTE: since tp sharded params are flattened, so we need to save the original param shapes
                # so that we can recontruct the original shapes => reconstruct the unsharded params in tensor parallel dimension
                config["configs"]["orig_param_shapes"] = convert_to_string(optimizer._orig_param_shapes)

            json.dump(config, fo)

    # We dump the optimizer state using `torch.save`
    torch.save(
        optimizer.state_dict(),
        root_folder
        / optimizer_filename(parallel_context, is_zero=optimizer.inherit_from(optim.ZeroDistributedOptimizer)),
    )


def save_lr_scheduler(
    lr_scheduler,
    parallel_context: ParallelContext,
    root_folder: Path,
):
    """Saves lr scheduler states"""
    if dist.get_rank(parallel_context.world_pg) > 0:
        # Only WORLD-RANK 0 saves the lr scheduler state
        return

    root_folder = root_folder / "lr_scheduler"
    root_folder.mkdir(exist_ok=True, parents=True)

    # We dump the optimizer state using `torch.save`
    torch.save(
        lr_scheduler.state_dict(),
        root_folder / lr_scheduler_filename(),
    )


@torch.no_grad()
def load_optimizer(
    optimizer: optim.BaseOptimizer,
    parallel_context: ParallelContext,
    root_folder: Path,
    map_location: Optional[str] = None,
    param_shard_metadata: Tuple[Tuple[int, int], TensorMetadata] = None,  # (pp_rank, tp_rank) -> TensorMetadata
    model: Optional[nn.Module] = None,
):
    root_folder = root_folder / "optimizer"
    # `load_state_dict` copies the state dict which can be very large in case of Zero-0 so we load to cpu and then move to the right device
    map_location = "cpu" if not optimizer.inherit_from(optim.ZeroDistributedOptimizer) else map_location
    ckp_optimizer_config_path = root_folder / "optimizer_config.json"
    with open(ckp_optimizer_config_path, "r") as file:
        ckp_optimizer_config = json.load(file)

    ckp_pp_size = ckp_optimizer_config["parallelism"]["pp_size"]
    ckp_tp_size = ckp_optimizer_config["parallelism"]["tp_size"]
    ckp_dp_size = ckp_optimizer_config["parallelism"]["dp_size"]
    ckpt_expert_parallel_size = ckp_optimizer_config["parallelism"]["expert_parallel_size"]

    if int(ckp_tp_size) != int(parallel_context.tp_pg.size()) or int(ckp_pp_size) != int(
        parallel_context.pp_pg.size()
    ):
        assert (
            param_shard_metadata is not None
        ), f"You have to pass how the original parameters are sharded in order to resume in a different tensor parallel size, ckp_tp_size: {ckp_tp_size}, current tp_size: {parallel_context.tp_pg.size()}"
        assert (
            model is not None
        ), "You have to pass the model in order to adjust the optimizer states according to how the current parameters are sharded"

        def get_checkpoint_state_metadata(param_name: str, pp_rank: int, tp_rank: int) -> TensorMetadata:
            return param_shard_metadata[param_name.replace("module.", "")][(str(pp_rank), str(tp_rank))]

        ckp_optim_type = ckp_optimizer_config["type"]

        if ckp_optim_type == ZeroDistributedOptimizer.__name__:
            # NOTE: if the checkpoint is from a Zero-1 optimizer, then we need to merge the shards
            # across data parallel dimension, before merging the shards across tensor parallel dimension
            shard_paths = list(
                root_folder.glob(
                    f"{ObjectType.OPTIMIZER.value}_pp-*-of-{ckp_pp_size}_dp-*-of-{ckp_dp_size}_tp-*-of-{ckp_tp_size}-exp-*-of-{ckpt_expert_parallel_size}.pt"
                )
            )
            ckp_sharded_optim_states = merge_dp_shard_in_zero1_optimizer(
                model, ckp_optimizer_config, shard_paths, parallel_context, map_location
            )
        else:
            # NOTE: if the checkpoint is from a Zero-0 optimizer, then we don't need to merge the shards
            # across data parallel dimension, just directly load the checkpoints
            shard_paths = list(
                root_folder.glob(f"{ObjectType.OPTIMIZER.value}_pp-*-of-{ckp_pp_size}_tp-*-of-{ckp_tp_size}.pt")
            )

            ckp_sharded_optim_states = {}
            for shard_path in shard_paths:
                pp_rank, tp_rank = extract_parallel_ranks_from_shard_path(shard_path, is_zero1=False)
                ckp_sharded_optim_states[(pp_rank, tp_rank)] = torch.load(shard_path, map_location=map_location)

        model_state_dict = model.state_dict()
        new_optim_state_dict = optimizer.state_dict()
        # TODO: this does not handle the edge case of different pipeline parallel optimizer state shards saving different state keys
        OPTIMIZER_STATE_NAMES = sorted(ckp_sharded_optim_states[(0, 0)]["state"][0].keys() - ["step"])
        # NOTE: because we can only resume training with the same optimizer type
        # (0, 0) = (pp_rank, tp_rank)
        # NOTE: also we don't merge "step" because it's just a scalar
        param_names = list(model_state_dict.keys())
        new_optim_state_param_names = {}
        # NOTE: iterates through all model parameters in the local pipeline parallel rank (hence, might not be the full model).
        # Since model parameters and optimizer states are aligned, loads only the optimizer states for these parameters from the checkpoint shards.
        for param_index, param_name in tqdm(
            enumerate(param_names),
            disable=dist.get_rank(parallel_context.world_pg) != 0,
            desc="Topology-agnostic optimizer loading",
        ):
            try:
                param = model.get_parameter(param_name)
            except AttributeError:
                param = None

            if not isinstance(param, NanotronParameter):
                raise NotImplementedError("Parameters are required to be NanotronParameter")

            # NOTE: for tied parameters, the metadata is stored using the parameter name,
            # while the data is stored using the name of the main tied parameter,
            # which may be different (e.g. `model.token_position_embeddings.pp_block.token_embedding.weight`
            # for `model.lm_head.pp_block.weight`).
            base_name = param.get_tied_info().name if param.is_tied else param_name
            if param_name != base_name:
                # NOTE: skip tied parameter if main tied parameter has already been loaded
                # (not always the case if pipeline parallel)
                if base_name in new_optim_state_param_names.values():
                    continue
            new_optim_state_param_names[param_index] = base_name

            if param.is_sharded:
                # NOTE: optimizer states's shape is equal to the parameter's shape
                # NOTE: sometines an unsharded parameter's shape differ
                # from an unsharded optimizer state's shape
                new_shard_metadata = param.get_sharded_info()
                new_unshared_shape = new_shard_metadata.unsharded_shape
                new_optim_state_dict["state"][param_index] = {}
                # NOTE: restore each state tensor (e.g. exg_avg) by iterating through
                # the optimizer state shards saved using the previous topology
                for state_key in OPTIMIZER_STATE_NAMES:
                    # TODO(xrsrke): free the memory of the shards that isn't
                    # corresponding to the current rank
                    buffer = torch.zeros_like(param, device="cuda")
                    unsharded_buffer = torch.empty(new_unshared_shape, device="cuda")

                    for (pp_rank, tp_rank), ckp_optim_state in ckp_sharded_optim_states.items():
                        old_optim_state_index = find_optim_index_from_param_name(
                            base_name, ckp_sharded_optim_states, is_zero1=False, pp_rank=pp_rank
                        )
                        if old_optim_state_index is None:
                            continue  # NOTE: param is not in this pp shard
                        ckp_shard_data = ckp_optim_state["state"][old_optim_state_index][state_key]
                        # NOTE: the metadata for the main parameter of a tied parameter might be in a
                        # different pipeline parallel shard.
                        if param.is_tied:
                            metadata_pp_rank = next(
                                iter(param_shard_metadata[param_name.replace("module.", "")].keys())
                            )[0]
                        else:
                            metadata_pp_rank = pp_rank
                        ckp_shard_metadata = get_checkpoint_state_metadata(param_name, metadata_pp_rank, tp_rank)

                        # NOTE: if the checkpoint is from a Zero-1 optimizer,
                        # so it's flattened, so we need to reshape it
                        if ckp_optim_type == ZeroDistributedOptimizer.__name__:
                            # NOTE: this is the original shape of the parameter before being flattened
                            orig_shape = ckp_optimizer_config["configs"]["orig_param_shapes"][param_name]
                            orig_shape = [int(dim) for dim in orig_shape]
                            ckp_shard_data = ckp_shard_data.view(orig_shape)

                        new_optim_state_dict["state"][param_index][state_key] = merge_and_shard_tp_tensors(
                            buffer,
                            unsharded_buffer,
                            [
                                (ckp_shard_data, ckp_shard_metadata.local_global_slices_pairs),
                            ],
                            new_shard_metadata,
                        )

                        if ckp_optim_type == ZeroDistributedOptimizer.__name__:
                            # NOTE: flatten the optimizer states
                            new_optim_state_dict["state"][param_index][state_key] = new_optim_state_dict["state"][
                                param_index
                            ][state_key].flatten()
                        # NOTE: a bit awkward, but while we're already reading this (pp,tp) shard for whatever state_key,
                        # try to get the step value as well.
                        step = ckp_optim_state["state"][old_optim_state_index].get("step")
                        if step is not None:
                            new_optim_state_dict["state"][param_index]["step"] = step

        new_optim_state_dict["names"] = new_optim_state_param_names
        state_dict = new_optim_state_dict
    else:
        # TODO @thomasw21: Load optimizer type and check that it's compatible otherwise we might be be loading something else completely
        state_dict = torch.load(
            root_folder
            / optimizer_filename(parallel_context, is_zero=optimizer.inherit_from(optim.ZeroDistributedOptimizer)),
            map_location=map_location,
        )

    if isinstance(optimizer, ZeroDistributedOptimizer):
        # NOTE: only reshard after merging tp shards
        # or we get a new dp_Size
        if int(ckp_tp_size) != parallel_context.tp_pg.size() or int(ckp_dp_size) != parallel_context.dp_pg.size():
            # NOTE: if the optimizer is ZeRO-1, now we shard the optimizer states across data parallel dimension
            current_dp_rank = dist.get_rank(parallel_context.dp_pg)
            OPTIMIZER_STATE_NAMES = state_dict["state"][0].keys() - ["step"]
            for param_index in state_dict["state"]:
                param_name = [name for idx, name in state_dict["names"].items() if idx == param_index][0]
                for state_name in OPTIMIZER_STATE_NAMES:
                    sliced_tensor = get_sliced_tensor(
                        param=state_dict["state"][param_index][state_name],
                        start_offset=optimizer.param_name_to_dp_rank_offsets[param_name][current_dp_rank][0],
                        end_offset=optimizer.param_name_to_dp_rank_offsets[param_name][current_dp_rank][1],
                    )
                    state_dict["state"][param_index][state_name] = sliced_tensor

    optimizer.load_state_dict(state_dict)

    if not optimizer.inherit_from(optim.ZeroDistributedOptimizer):
        check_optim_state_in_sync(optimizer, parallel_context.dp_pg)


def load_lr_scheduler(
    lr_scheduler,
    root_folder: Path,
):
    root_folder = root_folder / "lr_scheduler"

    state_dict = torch.load(root_folder / lr_scheduler_filename())
    lr_scheduler.load_state_dict(state_dict)
