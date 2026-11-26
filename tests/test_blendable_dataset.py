from types import SimpleNamespace

import numpy as np
import pytest
from nanotron.data.nemo_dataset.blendable_dataset import BlendableDataset


def _blendable_dataset(folder_paths, dataset_index):
    # Set only what the consumption tracking reads, so no data files or C++ helpers are needed
    dataset = BlendableDataset.__new__(BlendableDataset)
    dataset.datasets = [SimpleNamespace(folder_path=folder_path) for folder_path in folder_paths]
    dataset.dataset_index = np.array(dataset_index, dtype=np.int16)
    dataset.consumed_tokens = {idx: 0 for idx in range(len(folder_paths))}
    dataset.sequence_length = None
    return dataset


@pytest.mark.parametrize(
    "folder_paths",
    [
        ["datasets/c4-es/tokenized"],
        ["/data/SlimPajama-6B/tokenized", "datasets/c4-es/tokenized"],
        ["s3://bucket/SlimPajama-6B/tokenized", "/data/c4-es/tokenized"],
    ],
)
def test_consumption_stats_support_local_and_s3_folders(folder_paths):
    dataset_index = [i % len(folder_paths) for i in range(12)]
    dataset = _blendable_dataset(folder_paths, dataset_index)

    # Two steps with a global batch size of 4, as the trainer does after each step
    dataset.update_consumption_metrics(start_idx=0, end_idx=4, sequence_length=8)
    dataset.update_consumption_metrics(start_idx=4, end_idx=8, sequence_length=8)

    stats = dataset.get_consumption_stats()

    expected_samples = np.bincount(dataset_index[:8], minlength=len(folder_paths))
    assert stats == {
        folder_path: {"tokens": int(num_samples) * 8}
        for folder_path, num_samples in zip(folder_paths, expected_samples)
    }
