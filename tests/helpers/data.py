import hashlib
import importlib
import json
import os
import sys
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

package = importlib.import_module("nanotron")
package_path = Path(package.__file__).parent.parent.parent
sys.path.append(str(package_path))

import nanotron.distributed as dist
import torch
from nanotron.data.nanoset import Nanoset
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.sanity_checks import assert_tensor_synced_across_pg

from tools.preprocess_data import main


def create_dataset_paths(tmp_dir: str, quantity: int):
    json_dataset_path = [os.path.join(tmp_dir, f"pytest_{i}.json") for i in range(quantity)]
    datatrove_tokenized_dataset_paths = [os.path.join(tmp_dir, f"tokenized_documents_{i}") for i in range(quantity)]

    return json_dataset_path, datatrove_tokenized_dataset_paths


def create_dummy_json_dataset(path_to_json: str, dummy_text: str, n_samples: int = 50000):

    with open(path_to_json, "a") as json_file:
        for sample in range(n_samples):
            sample_dict = {"text": f"[{sample}] Hello! Im sample {sample}! And this is my dummy text: {dummy_text}"}
            json_file.write(json.dumps(sample_dict))
            json_file.write("\n")


def preprocess_dummy_dataset(json_dataset_path: str, datatrove_tokenized_dataset_path: str, tokenizer: str):
    # Create args for preprocessing
    args = Namespace(
        readers="jsonl",
        dataset=json_dataset_path,
        column="text",
        glob_pattern=None,
        output_folder=datatrove_tokenized_dataset_path,
        tokenizer_name_or_path=tokenizer,
        eos_token=None,
        n_tasks=1,
        logging_dir=None,
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


def compute_hash(identifier: OrderedDict, n_digit: int = 8) -> int:
    """
    Creates a sha256 hash from the elements of a OrderedDict
    """
    unique_description = json.dumps(identifier, indent=4)
    # Create n_digit description hash
    unique_description_hash = int(hashlib.sha256(unique_description.encode("utf-8")).hexdigest(), 16) % 10**n_digit
    return unique_description_hash


def assert_nanoset_sync_across_all_ranks(nanoset: Nanoset, parallel_context: ParallelContext):
    """
    Checks that the same Nanoset is created in all processes
    """
    # Extract a sample from the Nanoset
    IDX_SAMPLE = 23

    nanoset_identifiers = OrderedDict()
    nanoset_identifiers["dataset_folders"] = nanoset.dataset_folders
    nanoset_identifiers["dataset_weights"] = nanoset.dataset_weights.tolist()
    nanoset_identifiers["sequence_length"] = nanoset.sequence_length
    nanoset_identifiers["train_split_num_samples"] = nanoset.train_split_num_samples
    nanoset_identifiers["random_seed"] = nanoset.random_seed
    nanoset_identifiers["length"] = len(nanoset)
    nanoset_identifiers["input_ids"] = nanoset[IDX_SAMPLE]["input_ids"].tolist()
    nanoset_identifiers["dataset_index"] = nanoset.dataset_index.tolist()
    nanoset_identifiers["dataset_sample_index"] = nanoset.dataset_sample_index.tolist()
    nanoset_identifiers["token_size"] = nanoset.token_size

    unique_description_hash = compute_hash(nanoset_identifiers)
    assert_tensor_synced_across_pg(
        tensor=torch.tensor(unique_description_hash, device="cuda"),
        pg=parallel_context.world_pg,
        msg=lambda err: f"Nanoset is not synchronized across all processes {err}",
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
