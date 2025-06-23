import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union

import dacite
import torch
from dacite import from_dict
from packaging.version import Version

from nanotron import distributed as dist
from nanotron.constants import CHECKPOINT_FILE_NAME, CHECKPOINT_VERSION
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import SlicesPair
from collections import defaultdict

@dataclasses.dataclass
class DataStageMetadata:
    """
    consumed_train_samples: The number of samples consumed by the model in the this stage (resets at each stage).
    consumed_tokens_per_dataset_folder: The number of tokens consumed by the model in the this stage for each dataset folder. (resets at each stage)
    """

    name: str
    start_training_step: int
    consumed_train_samples: int # We use this for sampler, and it's reset at each stage
    sequence_length: Optional[int] = None # TODO: put back as non-optional
    consumed_tokens_per_dataset_folder: Dict[str, int] = dataclasses.field(default_factory=dict) # this gets reset at each stage

    def __post_init__(self):
        if self.sequence_length is None:
            self.sequence_length = 4096 # TODO: temp

    def sanity_consumed_train_samples(self):
        assert self.consumed_train_samples*self.sequence_length == sum(self.consumed_tokens_per_dataset_folder.values()), f"Mismatch between the total consumed samples and the sum of consumed samples across dataset folders! consumed_train_samples={self.consumed_train_samples}, sequence_length={self.sequence_length}, consumed_tokens_per_dataset_folder={self.consumed_tokens_per_dataset_folder}"

    @property
    def consumed_tokens_all_datasets(self):
        return sum(self.consumed_tokens_per_dataset_folder.values())

@dataclasses.dataclass
class TrainingMetadata:
    """
    consumed_train_samples: The number of samples consumed globally, across all stages.
    last_train_step: The last training step across all stages.
    last_stage_idx: The index of the last stage that was trained.
    data_stages: The metadata for each stage.
    """

    consumed_train_samples: int # TODO: Legacy. This assumed same sequence length across all stages. Not used anymore
    last_train_step: int
    consumed_tokens_total: Optional[int] = None # TODO: put back as non-optional

    # TODO(xrsrke): make this not optional, once we entirely remove
    # the old checkpoint version
    last_stage_idx: Optional[int] = None
    data_stages: Optional[List[DataStageMetadata]] = None

    def __post_init__(self):
        # NOTE: this is a sanity check after loading a trained checkpoint
        assert (
            self.consumed_train_samples == sum(stage.consumed_train_samples for stage in self.data_stages)
        ), "Mismatch between the total consumed samples and the sum of consumed samples across stages! Something went wrong in the training."

        if self.consumed_tokens_total is not None:
            assert self.consumed_tokens_total == sum(stage.consumed_tokens_all_datasets for stage in self.data_stages), "Mismatch between the total consumed tokens and the sum of consumed tokens across stages! Something went wrong in the training."
        else:
            self.consumed_tokens_total = sum(stage.consumed_tokens_all_datasets for stage in self.data_stages)

        # TODO(xrsrke): remove this once we entirely remove non-data-stage training
        if self.last_stage_idx is not None:
            assert self.data_stages is not None, "data_stages should not be None if last_stage_idx is not None"

    @property
    def consumed_tokens_per_dataset_folder_total(self):
        consumed = defaultdict(int)
        for stage in self.data_stages:
            for dataset_folder, tokens in stage.consumed_tokens_per_dataset_folder.items():
                consumed[dataset_folder] += tokens
        return consumed
    
    @property
    def current_stage(self) -> DataStageMetadata:
        return self.data_stages[self.last_stage_idx]


@dataclasses.dataclass
class CheckpointMetadata:
    version: Version
    tp: int
    dp: int
    metas: TrainingMetadata
    cp: int = 1
    custom_metas: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class TensorMetadata:
    # Mandatory for checkpoint version higher than 1.2
    version: Version
    # Anything users want to store
    # Info of to what slice of the unsharded tensor (global_slices) the current sharded tensor corresponds (local_slices)
    local_global_slices_pairs: Tuple[SlicesPair, ...]
    # The shape of the unsharded tensor
    unsharded_shape: Tuple[int, ...]

    _metadata_config: ClassVar[dacite.Config] = dacite.Config(
        cast=[Version],
        type_hooks={
            Tuple[SlicesPair, ...]: SlicesPair.tuple_from_str,
            Tuple[int, ...]: lambda x: torch.Size(
                int(re.search(r"\((\d+)\)", size).group(1)) if "np.int" in size else int(size)
                for size in x.strip("()").split(",")
                if size
            ),
        },
        strict=True,
    )

    def to_str_dict(self) -> Dict[str, str]:
        return {
            "version": str(self.version),
            "local_global_slices_pairs": SlicesPair.tuple_to_str(self.local_global_slices_pairs),
            "unsharded_shape": str(tuple(self.unsharded_shape)),
        }

    @classmethod
    def from_str_dict(cls, dictionary: Dict[str, str]) -> "TensorMetadata":
        tensor_metadata: TensorMetadata = dacite.from_dict(
            data_class=TensorMetadata,
            data=dictionary,
            config=cls._metadata_config,
        )
        return tensor_metadata


def process_type(elt: Any, type_hooks: Dict[Type, Callable[[Any], Any]]):
    if isinstance(elt, dict):
        return to_dict(elt, type_hooks=type_hooks)
    elif elt.__class__ in type_hooks:
        return type_hooks[elt.__class__](elt)
    elif isinstance(elt, (list, tuple)):
        return to_list(elt, type_hooks=type_hooks)
    else:
        return elt


def to_dict(dict_: Dict, type_hooks: Dict[Type, Callable[[Any], Any]]):
    result = {}
    for key, value in dict_.items():
        result[key] = process_type(value, type_hooks=type_hooks)
    return result


def to_list(list_: Union[List, Tuple], type_hooks: Dict[Type, Callable[[Any], Any]]):
    return list_.__class__((process_type(elt, type_hooks=type_hooks) for elt in list_))


def save_meta(parallel_context: ParallelContext, root_folder: Path, training_metadata: TrainingMetadata):
    assert isinstance(training_metadata, TrainingMetadata)

    if dist.get_rank(parallel_context.world_pg) != 0:
        return

    root_folder.mkdir(exist_ok=True, parents=True)
    checkpoint_metadata = CheckpointMetadata(
        version=CHECKPOINT_VERSION,
        tp=parallel_context.tp_pg.size(),
        dp=parallel_context.dp_pg.size(),
        cp=parallel_context.cp_pg.size(),
        metas=training_metadata,
    )

    # There are some types that require manual casting in order to work correctly.
    processed_metadata = process_type(dataclasses.asdict(checkpoint_metadata), type_hooks={Version: lambda x: str(x)})

    with open(root_folder / CHECKPOINT_FILE_NAME, mode="w") as fo:
        json.dump(processed_metadata, fo, indent=2, sort_keys=True)


def load_meta(parallel_context: ParallelContext, root_folder: Path) -> CheckpointMetadata:
    with open(root_folder / CHECKPOINT_FILE_NAME, mode="r") as fi:
        checkpoint_metadata = json.load(fi)
        checkpoint_metadata = from_dict(
            data_class=CheckpointMetadata,
            data=checkpoint_metadata,
            config=dacite.Config(
                cast=[Version],
            ),
        )
        # Assume that we're always backward compatible, we only increment CHECKPOINT_VERSION when there's a breaking change.
        assert (
            checkpoint_metadata.version <= CHECKPOINT_VERSION
        ), f"Checkpoint is of version {checkpoint_metadata.version}, Current `nanotron` checkpoint version is {CHECKPOINT_VERSION}"
    return checkpoint_metadata
