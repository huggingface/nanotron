import pytest
from datasets import load_dataset
from nanotron.doremi.dataloader import CombinedDataset


@pytest.fixture
def dataset1():
    return load_dataset("stas/c4-en-10k", split="train")


@pytest.fixture
def dataset2():
    return load_dataset("tiny_shakespeare", split="train")


def test_combined_dataset_length(dataset1, dataset2):
    combined_dataset = CombinedDataset([dataset1, dataset2])
    assert len(combined_dataset) == len(dataset1) + len(dataset2)


@pytest.mark.parametrize("idxs", [[0, 1], [[0, 1], [2, 3]]])
def test_get_item_from_combined_dataset(dataset1, dataset2, idxs):
    def count_elements(lst):
        return sum(count_elements(i) if isinstance(i, list) else 1 for i in lst)

    combined_dataset = CombinedDataset([dataset1, dataset2])
    outputs = combined_dataset[idxs]
    total_elements = count_elements(idxs)
    first_key = next(iter(outputs))  # NOTE: obtain the first key in adict

    assert isinstance(outputs, dict)
    assert outputs.keys() == dataset1[0].keys()
    assert len(outputs[first_key]) == total_elements

    assert outputs[first_key][0] == dataset1[0][first_key]
    assert outputs[first_key][1] == dataset1[1][first_key]
    # TODO(xrsrke): add test get items from other datasets
