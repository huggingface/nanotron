from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dacite
import torch
from packaging.version import Version
from torch import nn

from nanotron.core import distributed as dist
from nanotron.core import logging
from nanotron.core.dataclass import DistributedProcessGroups
from nanotron.core.distributed import get_global_rank
from nanotron.core.logging import log_rank
from nanotron.core.parallelism.parameters import BRRRParameter, ShardedInfo, SlicesPair
from nanotron.core.serialize.constants import CHECKPOINT_VERSION
from nanotron.core.serialize.meta import CheckpointMetadata, TensorMetadata, TensorMetadataV2, load_meta
from nanotron.core.serialize.path import (
    ObjectType,
    check_path_is_local,
    get_path,
    get_tp_and_pp_rank_and_size_from,
)
from nanotron.core.serialize.serialize import safe_open, save_file

logger = logging.get_logger(__name__)


def save_weights(model: nn.Module, dpg: DistributedProcessGroups, root_folder: Path):
    root_folder = root_folder / "model"

    # We save only `dist.get_rank(dpg.dp_pg) == 0`
    # TODO @thomasw21: Figure how this works with Zero-3
    if dist.get_rank(dpg.dp_pg) != 0:
        return

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    # We chunk everything by `tp_world_size` in order to make sure that we gather all the weights into a single device before saving it
    for name, param_or_buffer in model.state_dict().items():
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            # TODO @thomasw21: That's supposed to be slow. Can we try not calling `get_parameter()`?
            param = model.get_parameter(name)
        except AttributeError:
            # TODO @thomasw21: Handle buffers
            param = None

        if isinstance(param, BRRRParameter):
            metadata = {}
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
                group_ranks = tied_info.global_ranks
                group = dpg.world_ranks_to_pg[group_ranks]
                # Only the first rank of the group of the tied weights saves weights
                # TODO @thomasw21: We could rotate in order to balance the load.
                if dist.get_rank(group) != 0:
                    continue
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()
                group = dpg.world_ranks_to_pg[sharded_info.global_ranks]
                tp_and_pp_rank_and_size = get_tp_and_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=dist.get_rank(group)), dpg=dpg
                )
                metadata = TensorMetadataV2(
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
            if check_path_is_local(path):
                path.parent.mkdir(exist_ok=True, parents=True)
            save_file(tensors={"data": param_or_buffer}, filename=path, metadata=metadata)
        else:
            raise NotImplementedError("Parameters are required to be BRRRParameter")


class CheckpointVersionFromShardFileException(Exception):
    """Raise when loading checkpoint version from shard file fails"""


def read_checkpoint_version_from_shard_file(param_save_path: Path) -> Version:
    try:
        with safe_open(param_save_path, framework="pt", device=str("cpu")) as fi:
            param_metadata = fi.metadata()
            param_metadata = TensorMetadataV2.from_str_dict(param_metadata)
            checkpoint_version = param_metadata.version
    except (dacite.exceptions.MissingValueError, dacite.exceptions.UnexpectedDataError):
        raise CheckpointVersionFromShardFileException()
    return checkpoint_version


def read_checkpoint_version_from_meta(dpg: DistributedProcessGroups, root_folder: Path) -> Version:
    checkpoint_metadata: CheckpointMetadata = load_meta(dpg=dpg, root_folder=root_folder)
    checkpoint_version = checkpoint_metadata.version
    return checkpoint_version


def get_checkpoint_version(dpg, root_folder, param_save_path: Path) -> Version:
    try:
        checkpoint_version = read_checkpoint_version_from_shard_file(param_save_path=param_save_path)
    except CheckpointVersionFromShardFileException:
        log_rank(
            f"Failed to read checkpoint version from shard file {param_save_path}, reading from meta file.",
            logger=logger,
            level=logging.ERROR,
            rank=0,
        )
        checkpoint_version = read_checkpoint_version_from_meta(dpg=dpg, root_folder=root_folder)
    return checkpoint_version


def load_sharded_param_v1_0(param_or_buffer: torch.Tensor, sharded_info: ShardedInfo, shards_path: List[Path]):
    checkpoint_sharded_concat_dim = None
    shards = []
    for shard_path in shards_path:
        with safe_open(shard_path, framework="pt", device=str(param_or_buffer.device)) as fi:
            # TODO @thomasw21: Choose only a slice if we switch the TP topology
            shards.append(fi.get_tensor("data"))
            param_metadata = fi.metadata()
            if checkpoint_sharded_concat_dim is None:
                checkpoint_sharded_concat_dim = int(param_metadata["concat_dim"])
            else:
                assert checkpoint_sharded_concat_dim == int(param_metadata["concat_dim"])

    assert checkpoint_sharded_concat_dim is not None
    # TODO @thomasw21: Interestingly enough we don't actually need to instantiate the entire model at all.
    unsharded_tensor = torch.cat(shards, dim=checkpoint_sharded_concat_dim)

    # TODO(kunhao): check unsharded_tensor is fully filled
    for slices_pair in sharded_info.local_global_slices_pairs:
        local_slices = slices_pair.local_slices
        global_slices = slices_pair.global_slices
        param_or_buffer[local_slices] = unsharded_tensor[global_slices]


def load_sharded_param_w_metadataclass(
    meta_dataclass: Union[TensorMetadata, TensorMetadataV2],
    param_or_buffer: torch.Tensor,
    sharded_info: ShardedInfo,
    shards_path: List[Path],
):
    checkpoint_unsharded_shape = None
    shards_and_slices_maps: List[Tuple[torch.Tensor, Tuple[SlicesPair, ...]]] = []
    for shard_path in shards_path:
        with safe_open(shard_path, framework="pt", device=str(param_or_buffer.device)) as fi:
            # TODO @thomasw21: Choose only a slice if we switch the TP topology
            param_metadata = fi.metadata()
            param_metadata = meta_dataclass.from_str_dict(param_metadata)
            shards_and_slices_maps.append((fi.get_tensor("data"), param_metadata.local_global_slices_pairs))
            if checkpoint_unsharded_shape is None:
                checkpoint_unsharded_shape = param_metadata.unsharded_shape
            else:
                assert checkpoint_unsharded_shape == param_metadata.unsharded_shape

    assert checkpoint_unsharded_shape is not None
    # TODO @thomasw21: Interestingly enough we don't actually need to instantiate the entire model at all.
    unsharded_tensor = torch.empty(checkpoint_unsharded_shape, device=param_or_buffer.device)
    for shard, slices_pairs in shards_and_slices_maps:
        for slices_pair in slices_pairs:
            local_slices = slices_pair.local_slices
            global_slices = slices_pair.global_slices
            unsharded_tensor[global_slices] = shard[local_slices]

    # TODO(kunhao): check unsharded_tensor is fully filled
    for slices_pair in sharded_info.local_global_slices_pairs:
        local_slices = slices_pair.local_slices
        global_slices = slices_pair.global_slices
        param_or_buffer[local_slices] = unsharded_tensor[global_slices]


def load_sharded_param_v1_1(param_or_buffer: torch.Tensor, sharded_info: ShardedInfo, shards_path: List[Path]):
    load_sharded_param_w_metadataclass(
        meta_dataclass=TensorMetadata,
        param_or_buffer=param_or_buffer,
        sharded_info=sharded_info,
        shards_path=shards_path,
    )


def load_sharded_param_latest(param_or_buffer: torch.Tensor, sharded_info: ShardedInfo, shards_path: List[Path]):
    load_sharded_param_w_metadataclass(
        meta_dataclass=TensorMetadataV2,
        param_or_buffer=param_or_buffer,
        sharded_info=sharded_info,
        shards_path=shards_path,
    )


def load_weights(
    model: nn.Module,
    dpg: DistributedProcessGroups,
    root_folder: Path,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    """Load weights from a checkpoint

    Args:
        model: model to load weights into
        dpg: distributed process groups
        root_folder: root folder of the checkpoint
        filtered_state_dict: state dict to load from (overrides model.state_dict()). if None, load from model.state_dict()
    """
    param_root_folder = root_folder / "model"

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    checkpoint_version: Optional[Version] = None

    filtered_state_dict = filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    for name, param_or_buffer in filtered_state_dict.items():
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None

        if isinstance(param, BRRRParameter):
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()

                if param.is_tied:
                    # When params are tied only the first rank of tied param group stores weights (see save_weights)
                    group = dpg.world_ranks_to_pg[tied_info.global_ranks]
                    group_rank = 0
                else:
                    group = dpg.world_ranks_to_pg[sharded_info.global_ranks]
                    group_rank = dist.get_rank(group)

                tp_and_pp_rank_and_size = get_tp_and_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=group_rank), dpg=dpg
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
            else:
                # Let's assume that the topology changed and the param is sharded.
                # We search for all the files from the shards, concatenate the "unsharded" tensor and load the specific shard we're interested in.
                assert (
                    param.is_sharded
                ), f"`{name}` is not a sharded parameter. It's possible you were expecting {path} to exist."
                # TODO @thomasw21: Make so that we don't need to code this logic somewhere else than in `get_path`
                sharded_info = param.get_sharded_info()
                suffix = base_name.rsplit(".", 1)[-1]
                shards_path = list(path.parent.glob(f"{ObjectType.MODEL.value}_{suffix}*.safetensors"))
                assert len(shards_path) > 0, f"Could not find any shards in {path.parent}"

                if checkpoint_version is None:
                    checkpoint_version = get_checkpoint_version(dpg, root_folder, param_save_path=shards_path[0])
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

                if checkpoint_version <= Version("1.0"):
                    load_sharded_param_v1_0(
                        param_or_buffer=param_or_buffer, sharded_info=sharded_info, shards_path=shards_path
                    )
                elif checkpoint_version <= Version("1.1"):
                    load_sharded_param_v1_1(
                        param_or_buffer=param_or_buffer, sharded_info=sharded_info, shards_path=shards_path
                    )
                elif checkpoint_version <= CHECKPOINT_VERSION:
                    load_sharded_param_latest(
                        param_or_buffer=param_or_buffer, sharded_info=sharded_info, shards_path=shards_path
                    )
                else:
                    raise ValueError(f"Unsupported checkpoint version {checkpoint_version}")

        else:
            raise NotImplementedError(f"Parameters {param} should be a BRRRParameter")
