from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dacite
import torch
from packaging.version import Version
from safetensors.torch import safe_open, save_file
from torch import nn
from tqdm import tqdm

from nanotron import distributed as dist
from nanotron import logging
from nanotron.constants import CHECKPOINT_VERSION
from nanotron.distributed import get_global_rank
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter, ShardedInfo, SlicesPair
from nanotron.serialize.metadata import CheckpointMetadata, TensorMetadata, load_meta
from nanotron.serialize.utils import (
    ObjectType,
    extract_tp_pp_rank_from_shard_path,
    get_path,
    get_tp_and_pp_rank_and_size_from,
    merge_and_shard_tp_tensors,
)

logger = logging.get_logger(__name__)


def save_weights(model: nn.Module, parallel_context: ParallelContext, root_folder: Path):
    root_folder = root_folder / "model"

    # We save only `dist.get_rank(parallel_context.dp_pg) == 0`
    # TODO @thomasw21: Figure how this works with Zero-3
    if dist.get_rank(parallel_context.dp_pg) != 0:
        return

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    # We chunk everything by `tp_world_size` in order to make sure that we gather all the weights into a single device before saving it
    for name, param_or_buffer in tqdm(model.state_dict().items(), desc="Saving weights"):
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            # TODO @thomasw21: That's supposed to be slow. Can we try not calling `get_parameter()`?
            param = model.get_parameter(name)
        except AttributeError:
            # TODO @thomasw21: Handle buffers
            param = None

        if isinstance(param, NanotronParameter):
            metadata = {}
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
                group_ranks = tied_info.global_ranks
                group = parallel_context.world_ranks_to_pg[group_ranks]
                # Only the first rank of the group of the tied weights saves weights
                # TODO @thomasw21: We could rotate in order to balance the load.
                if dist.get_rank(group) != 0:
                    continue
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()
                group = parallel_context.world_ranks_to_pg[sharded_info.global_ranks]
                tp_and_pp_rank_and_size = get_tp_and_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=dist.get_rank(group)),
                    parallel_context=parallel_context,
                )
                metadata = TensorMetadata(
                    version=CHECKPOINT_VERSION,
                    local_global_slices_pairs=sharded_info.local_global_slices_pairs,
                    unsharded_shape=sharded_info.unsharded_shape,
                ).to_str_dict()

            else:
                tp_and_pp_rank_and_size = None

            path = root_folder.joinpath(
                *get_path(
                    base_name,
                    type=ObjectType.MODEL,
                    tp_and_pp_rank_and_size=tp_and_pp_rank_and_size,
                )
            )
            path.parent.mkdir(exist_ok=True, parents=True)
            try:
                save_file(tensors={"data": param_or_buffer}, filename=path, metadata=metadata)
            except Exception as e:
                log_rank(
                    f"Error saving {path} with {metadata}",
                    logger=logger,
                    level=logging.ERROR,
                    rank=0,
                )
                raise e
        else:
            raise NotImplementedError("Parameters are required to be NanotronParameter")


class CheckpointVersionFromShardFileException(Exception):
    """Raise when loading checkpoint version from shard file fails"""


def read_checkpoint_version_from_shard_file(param_save_path: Path) -> Version:
    try:
        with safe_open(param_save_path, framework="pt", device=str("cpu")) as fi:
            param_metadata = fi.metadata()
            param_metadata = TensorMetadata.from_str_dict(param_metadata)
            checkpoint_version = param_metadata.version
    except (dacite.exceptions.MissingValueError, dacite.exceptions.UnexpectedDataError):
        raise CheckpointVersionFromShardFileException()
    return checkpoint_version


def read_checkpoint_version_from_meta(parallel_context: ParallelContext, root_folder: Path) -> Version:
    checkpoint_metadata: CheckpointMetadata = load_meta(parallel_context=parallel_context, root_folder=root_folder)
    checkpoint_version = checkpoint_metadata.version
    return checkpoint_version


def get_checkpoint_version(parallel_context, root_folder, param_save_path: Path) -> Version:
    try:
        checkpoint_version = read_checkpoint_version_from_shard_file(param_save_path=param_save_path)
    except CheckpointVersionFromShardFileException:
        log_rank(
            f"Failed to read checkpoint version from shard file {param_save_path}, reading from meta file.",
            logger=logger,
            level=logging.ERROR,
            rank=0,
        )
        checkpoint_version = read_checkpoint_version_from_meta(
            parallel_context=parallel_context, root_folder=root_folder
        )
    return checkpoint_version


def load_sharded_param_latest(
    param_or_buffer: torch.Tensor,
    sharded_info: ShardedInfo,
    shards_path: List[Path],
    param_shard_metadata: Optional[Dict] = None,
):
    checkpoint_unsharded_shape = None
    shards_and_slices_maps: List[Tuple[torch.Tensor, Tuple[SlicesPair, ...]]] = []

    for shard_path in shards_path:
        with safe_open(shard_path, framework="pt", device=str(param_or_buffer.device)) as fi:
            # TODO @thomasw21: Choose only a slice if we switch the TP topology
            param_metadata = fi.metadata()
            param_metadata = TensorMetadata.from_str_dict(param_metadata)
            shards_and_slices_maps.append((fi.get_tensor("data"), param_metadata.local_global_slices_pairs))

            if checkpoint_unsharded_shape is None:
                checkpoint_unsharded_shape = param_metadata.unsharded_shape
            else:
                assert checkpoint_unsharded_shape == param_metadata.unsharded_shape

            if param_shard_metadata is not None:
                # NOTE: store how does model paramater are sharded
                # so that we can shard optimizer checkpoints in this way
                pp_rank, tp_rank = extract_tp_pp_rank_from_shard_path(shard_path)
                param_shard_metadata[(pp_rank, tp_rank)] = param_metadata

    assert checkpoint_unsharded_shape is not None
    # TODO @thomasw21: Interestingly enough we don't actually need to instantiate the entire model at all.
    unsharded_tensor = torch.empty(checkpoint_unsharded_shape, device=param_or_buffer.device)

    merge_and_shard_tp_tensors(
        buffer=param_or_buffer,
        unsharded_buffer=unsharded_tensor,
        shards_and_slices_maps=shards_and_slices_maps,
        shard_metadata=sharded_info,
    )

    return param_shard_metadata


def load_weights(
    model: nn.Module,
    parallel_context: ParallelContext,
    root_folder: Path,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    """Load weights from a checkpoint

    Args:
        model: model to load weights into
        parallel_context: distributed process groups
        root_folder: root folder of the checkpoint
        filtered_state_dict: state dict to load from (overrides model.state_dict()). if None, load from model.state_dict()
    """
    param_root_folder = root_folder / "model"

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    checkpoint_version: Optional[Version] = None

    filtered_state_dict = filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    param_shard_metadata = {}
    for name, param_or_buffer in tqdm(
        filtered_state_dict.items(), disable=dist.get_rank(parallel_context.world_pg) != 0, desc="Loading weights"
    ):
        # NOTE: extract how does the current model parameter are sharded
        # so that we can load optimizer checkpoints in this way
        param_shard_metadata[name] = {}
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None

        if isinstance(param, NanotronParameter):
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()

                if param.is_tied:
                    # When params are tied only the first rank of tied param group stores weights (see save_weights)
                    group = parallel_context.world_ranks_to_pg[tied_info.global_ranks]
                    group_rank = 0
                else:
                    group = parallel_context.world_ranks_to_pg[sharded_info.global_ranks]
                    group_rank = dist.get_rank(group)

                tp_and_pp_rank_and_size = get_tp_and_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=group_rank), parallel_context=parallel_context
                )
            else:
                tp_and_pp_rank_and_size = None

            path = param_root_folder.joinpath(
                *get_path(
                    base_name,
                    type=ObjectType.MODEL,
                    tp_and_pp_rank_and_size=tp_and_pp_rank_and_size,
                )
            )

            if path.exists():
                with safe_open(path, framework="pt", device=str(param.device)) as fi:
                    # TODO @thomasw21: Choose only a slice if we switch the TP topology
                    param_or_buffer[:] = fi.get_tensor("data")
            elif not path.parent.exists():
                raise ValueError(
                    f"Checkpoint is empty or checkpoint structure is not matching the model architecture."
                    f"Couldn't find folder {path.parent} in checkpoint at {root_folder}"
                )
            else:
                # Let's assume that the topology changed and the param is sharded.
                # We search for all the files from the shards, concatenate the "unsharded" tensor
                # and load the specific shard we're interested in.
                if not param.is_sharded:
                    raise ValueError(
                        f"`{name}` is not a sharded parameter. It's possible you were expecting {path} to exist."
                    )
                # TODO @thomasw21: Make so that we don't need to code this logic somewhere else than in `get_path`
                sharded_info = param.get_sharded_info()
                suffix = base_name.rsplit(".", 1)[-1]
                shards_path = list(path.parent.glob(f"{ObjectType.MODEL.value}_{suffix}*.safetensors"))
                if len(shards_path) <= 0:
                    raise ValueError(f"Could not find any shards in {path.parent}")

                if checkpoint_version is None:
                    checkpoint_version = get_checkpoint_version(
                        parallel_context, root_folder, param_save_path=shards_path[0]
                    )
                else:
                    current_checkpoint_version = None
                    try:
                        current_checkpoint_version = read_checkpoint_version_from_shard_file(
                            param_save_path=shards_path[0]
                        )
                    except CheckpointVersionFromShardFileException:
                        # The checkpoint version is read from the meta file
                        current_checkpoint_version = checkpoint_version
                    finally:
                        assert (
                            current_checkpoint_version == checkpoint_version
                        ), f"Checkpoint version mismatch at {shards_path[0]}."

                if checkpoint_version <= CHECKPOINT_VERSION:
                    load_sharded_param_latest(
                        param_or_buffer=param_or_buffer,
                        sharded_info=sharded_info,
                        shards_path=shards_path,
                        param_shard_metadata=param_shard_metadata[name],
                    )
                else:
                    raise ValueError(f"Unsupported checkpoint version {checkpoint_version}")

        else:
            raise NotImplementedError(f"Parameters {param} should be a NanotronParameter")

    return param_shard_metadata


def get_checkpoint_paths_list(
    model: nn.Module,
    parallel_context: ParallelContext,
    root_folder: Path,
    only_list_folders: bool = False,
    only_list_current_process: bool = True,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    """Return the list of all the files or folders created/accessed by the current process in a checkpoint

    Args:
        model: model to load weights into
        parallel_context: distributed process groups
        root_folder: root folder of the checkpoint
        filtered_state_dict: state dict to load from (overrides model.state_dict()). if None, load from model.state_dict()
    """
    param_root_folder = root_folder / "model"

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    paths = []

    filtered_state_dict = filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    for name in tqdm(
        filtered_state_dict.values(),
        disable=dist.get_rank(parallel_context.world_pg) != 0,
        desc="Listing checkpoint paths",
    ):
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None

        if isinstance(param, NanotronParameter) or not only_list_current_process:
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()

                if param.is_tied:
                    # When params are tied only the first rank of tied param group stores weights (see save_weights)
                    group = parallel_context.world_ranks_to_pg[tied_info.global_ranks]
                    group_rank = 0
                else:
                    group = parallel_context.world_ranks_to_pg[sharded_info.global_ranks]
                    group_rank = dist.get_rank(group)

                tp_and_pp_rank_and_size = get_tp_and_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=group_rank), parallel_context=parallel_context
                )
            else:
                tp_and_pp_rank_and_size = None

            if only_list_folders:
                paths.append(param_root_folder.joinpath(base_name.split(".")[:-1]))
            else:
                paths.append(
                    param_root_folder.joinpath(
                        *get_path(
                            base_name,
                            type=ObjectType.MODEL,
                            tp_and_pp_rank_and_size=tp_and_pp_rank_and_size,
                        )
                    )
                )

    return paths
