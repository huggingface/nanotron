import os
import sys

# Hack to import preprocess_data.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)))
from argparse import Namespace

import torch
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.sanity_checks import assert_tensor_synced_across_pg

import datasets
from tools.preprocess_data import main


def create_dummy_json_dataset(path_to_json: str, dummy_text: str, n_samples: int = 500):

    ds_dict = {"indices": list(range(n_samples))}
    ds = datasets.Dataset.from_dict(ds_dict)

    def create_text(sample):
        return {
            "text": f"[{sample['indices']}] Hello! I'm sample {sample['indices']}! And this is my dummy text: {dummy_text}"
        }

    new_ds = ds.map(create_text, batched=False).remove_columns(["indices"])
    new_ds.to_json(path_or_buf=path_to_json + ".json")


def preprocess_dummy_dataset(path_to_json: str):
    # Create args for preprocessing
    args = Namespace(
        input=path_to_json + ".json",
        json_keys=["text"],
        output_prefix=path_to_json,
        pretrained_model_name_or_path="openai-community/gpt2",
        workers=int(min(os.cpu_count(), 8)),
        partitions=int((min(os.cpu_count(), 8) / 2)),
        append_eod=True,
        log_interval=int(1000),
    )

    # tools/preprocess_data.py main
    main(args)


def assert_batch_dataloader(batch: dict, parallel_context):
    """
    batch (dict): Batch produced from the Dataloader, with keys input_ids, input_mask, label_ids, label_mask

    """
    for element in batch:
        tensor = batch[element]
        data_class = (
            0  # 0 if tensor is from the ids, 1 if TensorPointer and 2 if mask. Used in the data parallel group check
        )

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
            tensor=tensor.flatten(),
            pg=parallel_context.tp_pg,
            msg=lambda err: f"{element} is not synchronized across TP {err}",
        )

        # Assert that we have the SAME class of data in all processes belonging to the same data parallel group
        assert_tensor_synced_across_pg(
            tensor=torch.tensor(data_class),
            pg=parallel_context.dp_pg,
            msg=lambda err: f"{element} is not synchronized across DP {err}",
        )
