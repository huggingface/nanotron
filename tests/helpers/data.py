import importlib
import json
import os
import sys
from pathlib import Path

package = importlib.import_module("nanotron")
package_path = Path(package.__file__).parent.parent.parent
sys.path.append(str(package_path))

from argparse import Namespace

import numpy as np
import torch
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


def assert_batch_dataloader(batch: dict, parallel_context, micro_batch_size: int, sequence_length: int):
    """
    batch (dict): Batch produced from the Dataloader, with keys input_ids, input_mask, label_ids, label_mask

    """
    for element in batch:
        tensor = batch[element]
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


def get_max_value_by_group(dataset_idx, dataset_sample_idx, group_idx):
    mask = dataset_idx == group_idx
    filtered_values = dataset_sample_idx[mask]
    max_val = np.amax(filtered_values)
    return max_val
