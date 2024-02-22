import pytest
from utils import create_dummy_dataset, set_system_path

set_system_path()
from examples.doremi.doremi.dataloader import CombinedDataset


@pytest.fixture
def dataset1():
    return create_dummy_dataset(4000)


@pytest.fixture
def dataset2():
    return create_dummy_dataset(6000)


def test_combined_dataset_length(dataset1, dataset2):
    combined_dataset = CombinedDataset([dataset1, dataset2])
    assert len(combined_dataset) == len(dataset1) + len(dataset2)


@pytest.mark.parametrize("idx_type", ["idxs", "batch_of_idxs"])
def test_get_item_from_combined_dataset(dataset1, dataset2, idx_type):
    def count_elements(lst):
        return sum(count_elements(i) if isinstance(i, list) else 1 for i in lst)

    if idx_type == "batch_of_idxs":
        total_samples = len(dataset1) + len(dataset2)
        idxs = [[0, 1], [total_samples - 2, total_samples - 1]]
    else:
        idxs = [0, 1]

    combined_dataset = CombinedDataset([dataset1, dataset2])
    outputs = combined_dataset[idxs]
    # NOTE: obtain the first key in a dict
    first_key = next(iter(outputs))

    assert isinstance(outputs, dict)
    assert outputs.keys() == dataset1[0].keys()
    assert len(outputs[first_key]) == count_elements(idxs)

    assert outputs[first_key][0] == dataset1[0][first_key]
    assert outputs[first_key][1] == dataset1[1][first_key]
    if idx_type == "batch_of_idxs":
        assert outputs[first_key][2] == dataset2[len(dataset2) - 2][first_key]
        assert outputs[first_key][3] == dataset2[len(dataset2) - 1][first_key]
