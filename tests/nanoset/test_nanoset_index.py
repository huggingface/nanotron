import numpy as np
import pytest
from nanotron.data.nanoset import Nanoset, build_nanoset_index_helper
from nanotron.data.utils import normalize


def _build_index(dataset_lengths, train_split_num_samples, random_seed=1234):
    # Set only what build_nanoset_index reads, so no tokenized dataset files are needed
    nanoset = Nanoset.__new__(Nanoset)
    nanoset.dataset_lengths = dataset_lengths
    nanoset.dataset_weights = normalize(dataset_lengths)
    nanoset.train_split_num_samples = train_split_num_samples
    nanoset.random_seed = random_seed
    return nanoset.build_nanoset_index()


def _epochs(dataset_index, dataset_sample_index, samples_per_epoch, num_epochs):
    pairs = list(zip(dataset_index.tolist(), dataset_sample_index.tolist()))
    return [pairs[i * samples_per_epoch : (i + 1) * samples_per_epoch] for i in range(num_epochs)]


@pytest.mark.parametrize("dataset_lengths", [[20], [8, 12], [5, 30, 15]])
def test_nanoset_index_reshuffles_each_epoch(dataset_lengths):
    samples_per_epoch = sum(dataset_lengths)
    train_split_num_samples = 3 * samples_per_epoch + samples_per_epoch // 2
    dataset_index, dataset_sample_index = _build_index(dataset_lengths, train_split_num_samples)

    assert len(dataset_index) == len(dataset_sample_index) == train_split_num_samples

    one_epoch_index, one_epoch_sample_index = build_nanoset_index_helper(
        n_samples=samples_per_epoch, weights=normalize(dataset_lengths), dataset_sizes=dataset_lengths
    )
    one_epoch = sorted(zip(one_epoch_index.tolist(), one_epoch_sample_index.tolist()))

    epochs = _epochs(dataset_index, dataset_sample_index, samples_per_epoch, num_epochs=3)
    # Each full epoch reads the same samples as one epoch of the helper...
    for epoch in epochs:
        assert sorted(epoch) == one_epoch
    # ...in a new order every time
    assert epochs[0] != epochs[1]
    assert epochs[1] != epochs[2]


def test_nanoset_index_shuffles_each_epoch_with_its_own_seed():
    dataset_lengths = [8, 12]
    samples_per_epoch = sum(dataset_lengths)
    num_epochs = 3
    dataset_index, dataset_sample_index = _build_index(dataset_lengths, num_epochs * samples_per_epoch)

    # Epoch k is a single shuffle of both indexes with random_seed + k.
    # For epoch 0 this is the seed the previous single shuffle used, so the first epoch keeps its order.
    for epoch in range(num_epochs):
        expected_index, expected_sample_index = build_nanoset_index_helper(
            n_samples=samples_per_epoch, weights=normalize(dataset_lengths), dataset_sizes=dataset_lengths
        )
        np.random.RandomState(1234 + epoch).shuffle(expected_index)
        np.random.RandomState(1234 + epoch).shuffle(expected_sample_index)

        epoch_slice = slice(epoch * samples_per_epoch, (epoch + 1) * samples_per_epoch)
        np.testing.assert_array_equal(dataset_index[epoch_slice], expected_index)
        np.testing.assert_array_equal(dataset_sample_index[epoch_slice], expected_sample_index)


def test_nanoset_index_is_deterministic():
    first = _build_index([8, 12], 70, random_seed=42)
    second = _build_index([8, 12], 70, random_seed=42)
    other_seed = _build_index([8, 12], 70, random_seed=43)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other_seed[0]) or not np.array_equal(first[1], other_seed[1])
