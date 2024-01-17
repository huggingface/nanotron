import json
from pathlib import Path
from typing import Optional

import torch
from nanotron.core import distributed as dist
from nanotron.core import optim as optim
from nanotron.core.optim.zero import ZeroDistributedOptimizer
from nanotron.distributed import ParallelContext
from nanotron.serialize.utils import ObjectType
from torch import nn
from tqdm import tqdm


# TODO(xrsrke): take rank instead of parallel_context
def optimizer_filename(parallel_context: ParallelContext, is_zero: bool):
    if is_zero is True:
        return f"{ObjectType.OPTIMIZER.value}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}.pt"
    else:
        return f"{ObjectType.OPTIMIZER.value}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}_tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}.pt"


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
    # TODO @thomasw21: Figure out if I need to save param groups. Right now I'm assuming no as we only store what's trainable
    # TODO @thomasw21: We can probably "rotate" so that every process stores something (maybe doesn't matter if we're I/O bound)
    root_folder = root_folder / "optimizer"
    root_folder.mkdir(exist_ok=True, parents=True)

    if dist.get_rank(parallel_context.world_pg) == 0:
        with open(root_folder / "optimizer_config.json", "w") as fo:
            tp_size = parallel_context.tp_pg.size()
            pp_size = parallel_context.pp_pg.size()
            dp_size = parallel_context.dp_pg.size()

            # NOTE: remove str(x) here, do convert_to_string at the end of the config
            config = {
                "type": str(optimizer.__class__.__name__),
                "parallelism": {
                    "tp_size": str(tp_size),
                    "dp_size": str(dp_size),
                    "pp_size": str(pp_size),
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

            json.dump(config, fo)

    if (not optimizer.inherit_from(optim.ZeroDistributedOptimizer)) and dist.get_rank(parallel_context.dp_pg) > 0:
        # this is Zero-0, so only DP-0 saves the optimizer states
        return

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


def find_optim_index_from_param_name(optimizer, key_dict, param_name, ckp_sharded_optim_states):
    param_name = param_name.replace("module.", "")
    if key_dict == "state":
        # NOTE: since all shards have the same optim state names
        # so we take the first shard
        if isinstance(optimizer, ZeroDistributedOptimizer):
            # NOTE: (pp_rank, dp_rank, tp_rank)
            OPTIM_STATE_INDEX_TO_PARAM_NAME = ckp_sharded_optim_states[(0, 0, 0)]["names"]
        else:
            # NOTE: (pp_rank, tp_rank)
            OPTIM_STATE_INDEX_TO_PARAM_NAME = ckp_sharded_optim_states[(0, 0)]["names"]

        return next((k for k, v in OPTIM_STATE_INDEX_TO_PARAM_NAME.items() if v == param_name), None)
    else:
        return param_name


def _merge_dp_shard_in_zero1_optimizer(
    model,
    optimizer,
    optimizer_config,
    shard_paths,
    parallel_context: ParallelContext,
    map_location,
):
    import itertools

    def extract_pp_dp_tp_rank_from_shard_path(shard_path: Path):
        import re

        # TODO(xrsrke): use the same pattern as weight checkpoints
        # in weight checkpoints, we do pp-rank-.... but here we only do pp-...
        # TODO(xrsrke): don't hardcode this
        pattern = r"optimizer_pp-(\d+)-of-\d+_dp-(\d+)-of-\d+_tp-(\d+)-of-\d+\.pt"
        match = re.search(pattern, str(shard_path))
        if match:
            pp_rank, dp_rank, tp_rank = match.groups()
            return int(pp_rank), int(dp_rank), int(tp_rank)
        else:
            raise ValueError("Pattern not found in shard path")

    assert (
        optimizer_config["configs"]["param_name_to_dp_rank_offsets"] is not None
    ), "param_name_to_dp_rank_offsets is required"

    checkpoint_pp_size = optimizer_config["parallelism"]["pp_size"]
    checkpoint_tp_size = optimizer_config["parallelism"]["tp_size"]

    ckp_sharded_optim_states = {}
    for shard_path in shard_paths:
        pp_rank, dp_rank, tp_rank = extract_pp_dp_tp_rank_from_shard_path(shard_path)
        ckp_sharded_optim_states[(pp_rank, dp_rank, tp_rank)] = torch.load(shard_path, map_location=map_location)

    param_name_to_dp_rank_offsets = optimizer_config["configs"]["param_name_to_dp_rank_offsets"]

    def get_numel_of_unsharded_dp_param(param_name):
        dp_offsets = param_name_to_dp_rank_offsets[param_name]
        return max(int(value) for values in dp_offsets.values() for value in values)

    def assign_shard_to_buffer(buffer, offset, value):
        offset_start, offset_end = map(int, offset)
        buffer[offset_start:offset_end] = value
        return buffer

    param_names = sorted(model.state_dict().keys(), key=lambda x: x)
    ckp_merged_dp_shards_optim_states = {}

    for pp_rank, tp_rank in tqdm(
        list(itertools.product(range(int(checkpoint_pp_size)), range(int(checkpoint_tp_size)))),
        disable=dist.get_rank(parallel_context.world_pg) != 0,
        desc="Merging ZeRO-1's shards across data parallel dimension",
    ):
        # NOTE: filter only the shards that correspond to the current pp_rank and tp_rank
        filtered_ckp_sharded_optim_states = {}
        for (pp, dp, tp), ckp_optim_state in ckp_sharded_optim_states.items():
            if pp == pp_rank and tp == tp_rank:
                filtered_ckp_sharded_optim_states[dp] = ckp_optim_state

        # NOTE: now merge the shards across data parallel dimension
        # for each parameter, we need to merge all shards across data parallel dimension
        merged_dp_shards_optim_states = {}

        device = next(model.parameters()).device
        for param_name in param_names:
            merged_dp_shards_optim_states[param_name] = {}
            unshard_dp_size = get_numel_of_unsharded_dp_param(param_name)

            # TODO(xrsrke): detect if we need to merge gradient_accumulator states
            for key_dict in ["state", "gradient_accumulator"]:
                if key_dict == "state":
                    merged_dp_shards_optim_states[param_name][key_dict] = {}

                    # TODO(xrsrke): don't hardcore optimizer states
                    for state_name in ["exp_avg", "exp_avg_sq"]:
                        unsharded_dp_buffer = torch.zeros(unshard_dp_size, device=device)

                        # NOTE: now merge all the params across data parallel dimension
                        for dp_rank, ckp_optim_state in filtered_ckp_sharded_optim_states.items():
                            # NOTE: extract the optimizer state of the current parameter
                            optim_state_index = find_optim_index_from_param_name(
                                optimizer,
                                key_dict=key_dict,
                                param_name=param_name,
                                ckp_sharded_optim_states=ckp_sharded_optim_states,
                            )
                            ckp_optim_state = ckp_optim_state[key_dict][optim_state_index]
                            ckp_offset = param_name_to_dp_rank_offsets[param_name][str(dp_rank)]
                            assign_shard_to_buffer(unsharded_dp_buffer, ckp_offset, ckp_optim_state[state_name])

                        merged_dp_shards_optim_states[param_name][key_dict][state_name] = unsharded_dp_buffer
                else:
                    unsharded_dp_buffer = torch.zeros(unshard_dp_size, device=device)
                    for dp_rank, ckp_optim_state in filtered_ckp_sharded_optim_states.items():
                        # NOTE: extract the optimizer state of the current parameter
                        optim_state_index = find_optim_index_from_param_name(
                            optimizer,
                            key_dict=key_dict,
                            param_name=param_name,
                            ckp_sharded_optim_states=ckp_sharded_optim_states,
                        )
                        ckp_optim_state = ckp_optim_state[key_dict][optim_state_index]
                        ckp_offset = param_name_to_dp_rank_offsets[param_name][str(dp_rank)]
                        assign_shard_to_buffer(unsharded_dp_buffer, ckp_offset, ckp_optim_state)

                    merged_dp_shards_optim_states[param_name][key_dict] = unsharded_dp_buffer

        ckp_merged_dp_shards_optim_states[(pp_rank, tp_rank)] = merged_dp_shards_optim_states

    assert len(ckp_merged_dp_shards_optim_states) == checkpoint_pp_size * checkpoint_tp_size
    return ckp_merged_dp_shards_optim_states


@torch.no_grad()
def load_optimizer(
    optimizer: optim.BaseOptimizer,
    parallel_context: ParallelContext,
    root_folder: Path,
    map_location: Optional[str] = None,
    param_shard_metadata=None,
    model: Optional[nn.Module] = None,
):
    root_folder = root_folder / "optimizer"
    # `load_state_dict` copies the state dict which can be very large in case of Zero-0 so we load to cpu and then move to the right device
    map_location = "cpu" if not optimizer.inherit_from(optim.ZeroDistributedOptimizer) else map_location
    ckp_optimizer_config_path = root_folder / "optimizer_config.json"
    ckp_optimizer_config = json.load(ckp_optimizer_config_path.open("r"))

    # NOTE: for OptimizerFromGradientAccumulator only
    if ckp_optimizer_config["parallelism"]["tp_size"] == parallel_context.tp_pg.size():
        # TODO @thomasw21: Load optimizer type and check that it's compatible otherwise we might be be loading something else completely
        state_dict = torch.load(
            root_folder
            / optimizer_filename(parallel_context, is_zero=optimizer.inherit_from(optim.ZeroDistributedOptimizer)),
            map_location=map_location,
        )
    else:
        assert (
            param_shard_metadata is not None
        ), "You have to pass how the original parameters are sharded in order to resume in a different tensor parallel size"
        # NOTE: load checkpoint from a different tensor parallel size
        from nanotron.core.parallel.parameters import NanotronParameter

        def extract_tp_pp_rank_from_shard_path(shard_path: Path):
            import re

            # TODO(xrsrke): use the same pattern as weight checkpoints
            # in weight checkpoints, we do pp-rank-.... but here we only do pp-...
            pattern = r"pp-(\d+)-of-\d+_tp-(\d+)-of-\d+"
            match = re.search(pattern, str(shard_path))
            pp_rank, tp_rank = match.groups()
            return int(pp_rank), int(tp_rank)

        def merge_and_shard(buffer, unsharded_buffer, ckp_shard_data, current_shard_metadata, ckp_shard_metadata):
            for slices_pair in ckp_shard_metadata.local_global_slices_pairs:
                local_slices = slices_pair.local_slices
                global_slices = slices_pair.global_slices
                unsharded_buffer[global_slices] = ckp_shard_data[local_slices]

            for slices_pair in current_shard_metadata.local_global_slices_pairs:
                local_slices = slices_pair.local_slices
                global_slices = slices_pair.global_slices
                buffer[local_slices] = unsharded_buffer[global_slices]

            return buffer

        def get_checkpoint_state_metadata(param_name, pp_rank, tp_rank):
            return param_shard_metadata[param_name.replace("module.", "")][(str(pp_rank), str(tp_rank))]

        checkpoint_pp_size = ckp_optimizer_config["parallelism"]["pp_size"]
        checkpoint_tp_size = ckp_optimizer_config["parallelism"]["tp_size"]
        ckp_optim_type = ckp_optimizer_config["type"]

        # NOTE: if the checkpoint is from a Zero-1 optimizer, then we need to merge the shards
        # across data parallel dimension, before merging the shards across tensor parallel dimension
        if ckp_optim_type == ZeroDistributedOptimizer.__name__:
            checkpoint_dp_size = ckp_optimizer_config["parallelism"]["dp_size"]
            shard_paths = list(
                root_folder.glob(
                    f"{ObjectType.OPTIMIZER.value}_pp-*-of-{checkpoint_pp_size}_dp-*-of-{checkpoint_dp_size}_tp-*-of-{checkpoint_tp_size}.pt"
                )
            )
            ckp_sharded_optim_states = _merge_dp_shard_in_zero1_optimizer(
                model, optimizer, ckp_optimizer_config, shard_paths, parallel_context, map_location
            )
        else:
            # NOTE: if the checkpoint is from a Zero-0 optimizer, then we don't need to merge the shards
            # across data parallel dimension, just directly load the checkpoints
            shard_paths = list(
                root_folder.glob(
                    f"{ObjectType.OPTIMIZER.value}_pp-*-of-{checkpoint_pp_size}_tp-*-of-{checkpoint_tp_size}.pt"
                )
            )

            ckp_sharded_optim_states = {}
            for shard_path in shard_paths:
                pp_rank, tp_rank = extract_tp_pp_rank_from_shard_path(shard_path)
                ckp_sharded_optim_states[(pp_rank, tp_rank)] = torch.load(shard_path, map_location=map_location)

        model_state_dict = model.state_dict()
        new_optim_state_dict = optimizer.state_dict()
        for param_name, param_or_buffer in tqdm(
            sorted(model_state_dict.items(), key=lambda x: x[0]),
            disable=dist.get_rank(parallel_context.world_pg) != 0,
            desc="Topology-agnostic optimizer loading",
        ):
            try:
                param = model.get_parameter(param_name)
            except AttributeError:
                param = None

            if isinstance(param, NanotronParameter):
                if param.is_sharded:
                    # NOTE: optimizer states's shape is equal to the parameter's shape
                    # NOTE: sometines an unsharded parameter's shape differ
                    # from an unsharded optimizer state's shape
                    new_shard_metadata = param.get_sharded_info()
                    new_unshared_shape = new_shard_metadata.unsharded_shape

                    # TODO(xrsrke): find a better name than "dict_key"
                    for key_dict in ["state", "gradient_accumulator"]:
                        # NOTE: merging optimizer states
                        optim_state_index = find_optim_index_from_param_name(
                            optimizer, key_dict, param_name, ckp_sharded_optim_states
                        )

                        if key_dict == "state":
                            state_keys = ["exp_avg", "exp_avg_sq"]
                            new_optim_state_dict[key_dict][optim_state_index] = {}
                            for state_key in state_keys:
                                # TODO(xrsrke): free the memory of the shards that isn't
                                # corresponding to the current rank
                                unsharded_buffer = torch.empty(new_unshared_shape, device=param.device)
                                buffer = torch.zeros_like(param)

                                for (pp_rank, tp_rank), ckp_optim_state in ckp_sharded_optim_states.items():
                                    ckp_shard_metadata = get_checkpoint_state_metadata(param_name, pp_rank, tp_rank)
                                    ckp_shard_data = ckp_optim_state[key_dict][optim_state_index][state_key]
                                    new_optim_state_dict[key_dict][optim_state_index][state_key] = merge_and_shard(
                                        buffer,
                                        unsharded_buffer,
                                        ckp_shard_data,
                                        new_shard_metadata,
                                        ckp_shard_metadata,
                                    )

                            new_optim_state_dict[key_dict][optim_state_index]["step"] = ckp_optim_state[key_dict][
                                optim_state_index
                            ]["step"]
                        elif key_dict == "gradient_accumulator":
                            buffer = new_optim_state_dict[key_dict][optim_state_index]
                            unsharded_buffer = torch.empty(new_unshared_shape, device=param_or_buffer.device)

                            for (pp_rank, tp_rank), ckp_optim_state in ckp_sharded_optim_states.items():
                                ckp_shard_metadata = get_checkpoint_state_metadata(param_name, pp_rank, tp_rank)
                                ckp_shard_data = ckp_optim_state[key_dict][optim_state_index]
                                merge_and_shard(
                                    buffer,
                                    unsharded_buffer,
                                    ckp_shard_data,
                                    new_shard_metadata,
                                    ckp_shard_metadata,
                                )
            else:
                raise NotImplementedError("Parameters are required to be NanotronParameter")

        # NOTE: since all shards have the same optim state names
        # so we take the first shard
        new_optim_state_dict["names"] = ckp_sharded_optim_states[(0, 0)]["names"]
        state_dict = new_optim_state_dict
        # elif ckp_optim_type == ZeroDistributedOptimizer.__name__:
        #     # checkpoint_dp_size = ckp_optimizer_config["parallelism"]["dp_size"]
        #     # shard_paths = list(
        #     #     root_folder.glob(
        #     #         f"{ObjectType.OPTIMIZER.value}_pp-*-of-{checkpoint_pp_size}_dp-*-of-{checkpoint_dp_size}_tp-*-of-{checkpoint_tp_size}.pt"
        #     #     )
        #     # )

        #     # _merge_dp_shard_in_zero1_optimizer(
        #     #     model, optimizer, ckp_optimizer_config, shard_paths, parallel_context, map_location
        #     # )

        #     # for param_name, param_or_buffer in tqdm(
        #     #     sorted(model_state_dict.items(), key=lambda x: x[0]),
        #     #     disable=dist.get_rank(parallel_context.world_pg) != 0,
        #     #     desc="Topology-agnostic optimizer loading",
        #     # ):
        #     #     unshard_dp_size = get_numel_of_unsharded_dp_param(param_name)
        #     #     unsharded_dp_buffer = torch.zeros(unshard_dp_size, device=param_or_buffer.device)
        #     #     # NOTE: merge all checkpoints across data parallel dimension

        #     #     # TODO(xrsrke): support merging optimizer states across pipeline parallel dimension
        #     #     for (pp_rank, dp_rank, tp_rank), ckp_optim_state in ckp_sharded_optim_states.items():
        #     #         # extract the optimizer state for the current parameter
        #     #         optim_state_index = find_optim_index_from_param_name(
        #     #             optimizer,
        #     #             key_dict="state", param_name=param_name, ckp_sharded_optim_states=ckp_sharded_optim_states
        #     #         )

        #     #         ckp_optim_state = ckp_optim_state["state"][optim_state_index]
        #     #         ckp_offset = param_name_to_dp_rank_offsets[param_name][str(dp_rank)]
        #     #         # TODO(xrsrke): support other optimizer states
        #     #         apply_slice(unsharded_dp_buffer, ckp_offset, ckp_optim_state["exp_avg"])
        #     #         ckp_merged_dp_shards_optim_states[(pp_rank, tp_rank)] = {}
        #     #         ckp_merged_dp_shards_optim_states[(pp_rank, tp_rank)]["exp_avg"] = ckp_optim_state["exp_avg"]

        #     #     # NOTE: split optimizer states across tensor parallel dimension
        #     #     # NOTE: split optimizer states across data parallel dimension
        #     pass
        # else:
        #     raise NotImplementedError(f"{ckp_optim_type} is not supported yet")

    optimizer.load_state_dict(state_dict)


def load_lr_scheduler(
    lr_scheduler,
    root_folder: Path,
):
    root_folder = root_folder / "lr_scheduler"

    state_dict = torch.load(root_folder / lr_scheduler_filename())
    lr_scheduler.load_state_dict(state_dict)
