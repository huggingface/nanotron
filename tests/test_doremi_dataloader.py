import pytest
from datasets import load_dataset
from nanotron.doremi.dataloader import CombinedDataset


@pytest.fixture
def dataset1():
    return load_dataset("stas/c4-en-10k", split="train")


@pytest.fixture
def dataset2():
    return load_dataset("stas/openwebtext-synthetic-testing", split="10.repeat")


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


# # @pytest.mark.parametrize(
# #     "tp,dp,pp",
# #     [
# #         pytest.param(*all_3d_configs)
# #         for gpus in range(1, min(available_gpus(), 4) + 1)
# #         for all_3d_configs in get_all_3d_configurations(gpus)
# #     ],
# # )
# def test_sampling_from_dist_doremi_sampler():
#     # domain_weights = torch.tensor([0.5, 0.3, 0.1, 0.1])
#     # domain_keys = ["domain 0", "domain 1", "domain 2", "domain 3"]
#     # datasets = [dataset1, dataset2]
#     # doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)
#     # batch_size = 100

#     init_distributed(tp=1, dp=2, pp=1)(_test_sampling_from_dist_doremi_sampler)()


# def _test_sampling_from_dist_doremi_sampler(parallel_context: ParallelContext):
#     dp_size = dist.get_world_size(parallel_context.dp_pg)
#     dp_rank = dist.get_rank(parallel_context.dp_pg)

#     # sampler = DistributedSamplerForDoReMi(
#     #     datasets,
#     #     batch_size=batch_size,
#     #     num_replicas=dp_size,
#     #     rank=dp_rank,
#     #     doremi_context=doremi_context,
#     #     parallel_context=parallel_context,
#     # )
#     pass
