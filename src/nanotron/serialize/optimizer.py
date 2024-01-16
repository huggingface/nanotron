import json
from pathlib import Path
from typing import Optional

import torch
from nanotron.core import distributed as dist
from nanotron.core import optim as optim
from nanotron.distributed import ParallelContext
from nanotron.serialize.metadata import load_meta
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
            json.dump({"type": optimizer.__class__.__name__}, fo)

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


def load_optimizer(
    optimizer: optim.BaseOptimizer,
    parallel_context: ParallelContext,
    root_folder: Path,
    map_location: Optional[str] = None,
):
    root_folder = root_folder / "optimizer"
    # `load_state_dict` copies the state dict which can be very large in case of Zero-0 so we load to cpu and then move to the right device
    map_location = "cpu" if not optimizer.inherit_from(optim.ZeroDistributedOptimizer) else map_location

    # TODO @thomasw21: Load optimizer type and check that it's compatible otherwise we might be be loading something else completely
    state_dict = torch.load(
        root_folder
        / optimizer_filename(parallel_context, is_zero=optimizer.inherit_from(optim.ZeroDistributedOptimizer)),
        map_location=map_location,
    )
    optimizer.load_state_dict(state_dict)


@torch.no_grad()
def load_optimizer_topology_agnostic(
    # TODO(xrsrke): add typing
    param_shard_metadata,
    model: nn.Module,
    optimizer: optim.BaseOptimizer,
    parallel_context: ParallelContext,
    root_folder: Path,
    map_location: Optional[str] = None,
):
    checkpoint_metadata = load_meta(parallel_context=parallel_context, root_folder=root_folder)
    root_folder = root_folder / "optimizer"
    # `load_state_dict` copies the state dict which can be very large in case of Zero-0 so we load to cpu and then move to the right device
    map_location = "cpu" if not optimizer.inherit_from(optim.ZeroDistributedOptimizer) else map_location

    # NOTE: for OptimizerFromGradientAccumulator only
    if checkpoint_metadata.tp == parallel_context.tp_pg.size():
        # TODO @thomasw21: Load optimizer type and check that it's compatible otherwise we might be be loading something else completely
        state_dict = torch.load(
            root_folder
            / optimizer_filename(parallel_context, is_zero=optimizer.inherit_from(optim.ZeroDistributedOptimizer)),
            map_location=map_location,
        )
    else:
        from nanotron.core.parallel.parameters import NanotronParameter

        new_optim_state_dict = optimizer.state_dict()

        # TODO(xrsrke): make seria always save pp size even it's 1
        checkpoint_pp_size = getattr(checkpoint_metadata, "pp", 1)
        checkpoint_tp_size = checkpoint_metadata.tp
        shard_paths = list(
            root_folder.glob(
                f"{ObjectType.OPTIMIZER.value}_pp-*-of-{checkpoint_pp_size}_tp-*-of-{checkpoint_tp_size}.pt"
            )
        )

        # NOTE: load checkpoint from a different tensor parallel size
        # NOTE: gather the full optimizer states for each parameter
        # and take the local
        def extract_tp_pp_rank_from_shard_path(shard_path: Path):
            import re

            # TODO(xrsrke): use the same pattern as weight checkpoints
            # in weight checkpoints, we do pp-rank-.... but here we only do pp-...
            pattern = r"pp-(\d+)-of-\d+_tp-(\d+)-of-\d+"
            match = re.search(pattern, str(shard_path))
            pp_rank, tp_rank = match.groups()
            return pp_rank, tp_rank

        ckp_sharded_optim_states = {}
        for shard_path in shard_paths:
            pp_rank, tp_rank = extract_tp_pp_rank_from_shard_path(shard_path)
            ckp_sharded_optim_states[(pp_rank, tp_rank)] = torch.load(shard_path, map_location=map_location)

        def find_optim_index_from_param_name(key_dict, param_name):
            param_name = param_name.replace("module.", "")
            if key_dict == "state":
                OPTIM_STATE_INDEX_TO_PARAM_NAME = ckp_sharded_optim_states[("0", "0")]["names"]
                return next((k for k, v in OPTIM_STATE_INDEX_TO_PARAM_NAME.items() if v == param_name), None)
            else:
                return param_name

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

        model_state_dict = model.state_dict()
        # for param_name, param_or_buffer in tqdm(
        #     model_state_dict.items(),
        #     disable=dist.get_rank(parallel_context.world_pg) != 0,
        #     desc="Merging and sharding optimizer states",
        # ):
        assert 1 == 1
        for param_name, param_or_buffer in tqdm(
            sorted(model_state_dict.items(), key=lambda x: x[0]),
            disable=dist.get_rank(parallel_context.world_pg) != 0,
            desc="Merging and sharding optimizer states",
        ):
            try:
                param = model.get_parameter(param_name)
            except AttributeError:
                param = None
                raise ValueError(f"Parameter {param_name} is not found in the model")

            def get_checkpoint_state_metadata(param_name, pp_rank, tp_rank):
                # TODO(xrsrke): convert pp_rank, tp_rank to ingeter when saving
                return param_shard_metadata[param_name.replace("module.", "")][(str(pp_rank), str(tp_rank))]

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
                        optim_state_index = find_optim_index_from_param_name(key_dict, param_name)

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
                                # ckp_shard_data = ckp_optim_state[key_dict][optim_state_index][state_key]
                                # new_optim_state_dict[key_dict][optim_state_index][state_key] = merge_and_shard(
                                #     buffer,
                                #     unsharded_buffer,
                                #     ckp_shard_data,
                                #     new_shard_metadata,
                                #     ckp_shard_metadata,
                                # )
                                ckp_shard_data = ckp_optim_state[key_dict][optim_state_index]
                                merge_and_shard(
                                    buffer,
                                    unsharded_buffer,
                                    ckp_shard_data,
                                    new_shard_metadata,
                                    ckp_shard_metadata,
                                )

        new_optim_state_dict["names"] = ckp_sharded_optim_states[("0", "0")]["names"]
        state_dict = new_optim_state_dict

    # TODO(xrsrke): For ZeRO-1

    optimizer.load_state_dict(state_dict)


def load_lr_scheduler(
    lr_scheduler,
    root_folder: Path,
):
    root_folder = root_folder / "lr_scheduler"

    state_dict = torch.load(root_folder / lr_scheduler_filename())
    lr_scheduler.load_state_dict(state_dict)
