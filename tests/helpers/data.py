import hashlib
import importlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

package = importlib.import_module("nanotron")
package_path = Path(package.__file__).parent.parent.parent
sys.path.append(str(package_path))

from argparse import Namespace

import nanotron.distributed as dist
import numpy as np
import torch
from nanotron.data.blended_nanoset import BlendedNanoset
from nanotron.data.nanoset import Nanoset
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.sanity_checks import assert_tensor_synced_across_pg

from tools.preprocess_data import main


def create_dataset_paths(tmp_dir: str, quantity: int):
    json_dataset_path = [os.path.join(tmp_dir, f"pytest_{i}") for i in range(quantity)]
    bin_dataset_path = [f"{path}_text" for path in json_dataset_path]

    return json_dataset_path, bin_dataset_path


def create_dummy_json_dataset(path_to_json: str, dummy_text: str, n_samples: int = 50000):

    with open(path_to_json + ".json", "a") as json_file:
        for sample in range(n_samples):
            sample_dict = {"text": f"[{sample}] Hello! Im sample {sample}! And this is my dummy text: {dummy_text}"}
            json_file.write(json.dumps(sample_dict))
            json_file.write("\n")


def preprocess_dummy_dataset(path_to_json: str):
    # Create args for preprocessing
    args = Namespace(
        input=path_to_json + ".json",
        json_key="text",
        output_prefix=path_to_json,
        pretrained_model_name_or_path="openai-community/gpt2",
        workers=int(min(os.cpu_count(), 8)),
        partitions=int((min(os.cpu_count(), 8) / 2)),
        append_eos=True,
        log_interval=int(1000),
    )

    # tools/preprocess_data.py main
    main(args)


def assert_batch_dataloader(
    batch: dict, parallel_context: ParallelContext, micro_batch_size: int, sequence_length: int
):
    """
    batch (dict): Batch produced from the Dataloader, with keys input_ids, input_mask, label_ids, label_mask

    """
    for element in batch:
        tensor = batch[element]

        # Assert that inputs are only present in input_pp_rank and outputs in output_pp_rank
        input_pp_rank, output_pp_rank = 0, int(parallel_context.pp_pg.size() - 1)
        if dist.get_rank(parallel_context.pp_pg) == input_pp_rank and element.startswith("input_"):
            assert isinstance(tensor, torch.Tensor)
        elif dist.get_rank(parallel_context.pp_pg) == output_pp_rank and element.startswith("label_"):
            assert isinstance(tensor, torch.Tensor)
        else:
            assert isinstance(tensor, TensorPointer)

        data_class = (
            0  # 0 if tensor is from the ids, 1 if TensorPointer and 2 if mask. Used in the data parallel group check
        )

        # Check shape of mask and ids tensors
        if isinstance(tensor, torch.Tensor):
            assert tensor.shape == (micro_batch_size, sequence_length)

        # TensorPointer case: Check that all TensorPointers from the same tp_pg point to the same group_rank. Create torch.tensor with group_rank
        if isinstance(tensor, TensorPointer):
            tensor = torch.tensor(tensor.group_rank)
            data_class = 1

        # Attention Masks case: dtype is torch.bool --> Transform to int64
        if tensor.dtype == torch.bool:
            tensor = tensor.long()
            data_class = 2

        # Assert that we have the SAME element in all the processes belonging to the same tensor parallel group
        assert_tensor_synced_across_pg(
            tensor=tensor.flatten().cuda(),
            pg=parallel_context.tp_pg,
            msg=lambda err: f"{element} is not synchronized across TP {err}",
        )

        # Assert that we have the SAME class of data in all processes belonging to the same data parallel group
        assert_tensor_synced_across_pg(
            tensor=torch.tensor(data_class, device="cuda"),
            pg=parallel_context.dp_pg,
            msg=lambda err: f"{element} is not synchronized across DP {err}",
        )


def get_max_value_by_group(dataset_idx: np.array, dataset_sample_idx: np.array, group_idx: int):
    mask = dataset_idx == group_idx
    filtered_values = dataset_sample_idx[mask]
    max_val = np.amax(filtered_values)
    return max_val


def compute_hash(identifier: OrderedDict, n_digit: int = 8) -> int:
    """
    Creates a sha256 hash from the elements of a OrderedDict
    """
    unique_description = json.dumps(identifier, indent=4)
    # Create n_digit description hash
    unique_description_hash = int(hashlib.sha256(unique_description.encode("utf-8")).hexdigest(), 16) % 10**n_digit
    return unique_description_hash


def assert_nanoset(nanoset: Nanoset, parallel_context: ParallelContext):
    """
    Checks that the same Nanoset is created in all processes
    """
    # Extract a sample from the Nanoset
    IDX_SAMPLE = 23

    nanoset_identifiers = OrderedDict()
    nanoset_identifiers["path_prefix"] = nanoset.indexed_dataset.path_prefix
    nanoset_identifiers["split"] = nanoset.config.split
    nanoset_identifiers["random_seed"] = nanoset.config.random_seed
    nanoset_identifiers["sequence_length"] = nanoset.config.sequence_length
    nanoset_identifiers["length"] = len(nanoset)
    nanoset_identifiers["input_ids"] = nanoset[IDX_SAMPLE]["input_ids"].tolist()
    nanoset_identifiers["indices"] = nanoset.indexed_indices.tolist()

    unique_description_hash = compute_hash(nanoset_identifiers)
    assert_tensor_synced_across_pg(
        tensor=torch.tensor(unique_description_hash, device="cuda"),
        pg=parallel_context.world_pg,
        msg=lambda err: f"Nanoset is not synchronized across all processes {err}",
    )


def assert_blendednanoset(blendednanoset: BlendedNanoset, parallel_context: ParallelContext):
    """
    Checks that the same BlendedNanoset is created in all processes
    """
    # Extract a sample from the BlendedNanoset
    IDX_SAMPLE = 23

    blendednanoset_identifiers = OrderedDict()
    blendednanoset_identifiers["datasets"] = [dataset.unique_identifiers for dataset in blendednanoset.datasets]
    blendednanoset_identifiers["weights"] = blendednanoset.weights
    blendednanoset_identifiers["length"] = len(blendednanoset)
    blendednanoset_identifiers["dataset_sizes"] = blendednanoset.dataset_sizes
    blendednanoset_identifiers["input_ids"] = blendednanoset[IDX_SAMPLE]["input_ids"].tolist()
    blendednanoset_identifiers["dataset_index"] = blendednanoset.dataset_index.tolist()
    blendednanoset_identifiers["dataset_sample_index"] = blendednanoset.dataset_sample_index.tolist()

    unique_description_hash = compute_hash(blendednanoset_identifiers)
    assert_tensor_synced_across_pg(
        tensor=torch.tensor(unique_description_hash, device="cuda"),
        pg=parallel_context.world_pg,
        msg=lambda err: f"BlendedNanoset is not synchronized across all processes {err}",
    )


def compute_batch_hash(batch: dict) -> int:
    """
    Checks that the Nanoset/BlendedNanoset is in the same state after recovering from a crash

    batch (dict): Batch produced from the Dataloader, with keys input_ids, input_mask, label_ids, label_mask

    """
    batch_identifiers = OrderedDict()

    for element in batch:
        tensor = batch[element]

        # TensorPointer
        if isinstance(tensor, TensorPointer):
            identifier = tensor.group_rank

        # Attention Masks case: dtype is torch.bool --> Transform to int64
        elif tensor.dtype == torch.bool:
            identifier = tensor.long().tolist()

        # Input IDs tensor
        else:
            identifier = tensor.tolist()

        batch_identifiers[element] = identifier

    unique_description_hash = compute_hash(batch_identifiers)

    return unique_description_hash
