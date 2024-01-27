# from typing import List
# from datasets.arrow_dataset import Dataset
# from datasets.dataset_dict import DatasetDict, IterableDatasetDict
# from datasets.iterable_dataset import IterableDataset

# import pytest
# import torch
# from datasets import load_dataset
# from helpers.utils import init_distributed
# from nanotron import distributed as dist
# from nanotron.doremi.dataloader import DistributedSamplerForDoReMi
# from nanotron.doremi.doremi_context import DoReMiContext
# from nanotron.parallel import ParallelContext
# from torch.utils.data import Dataset


import pytest
import torch
from datasets import load_dataset
from helpers.utils import init_distributed
from nanotron import distributed as dist
from nanotron.doremi.dataloader import DistributedSamplerForDoReMi
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.parallel import ParallelContext


@pytest.fixture
def dataset1():
    return load_dataset("stas/c4-en-10k", split="train")


@pytest.fixture
def dataset2():
    return load_dataset("stas/openwebtext-synthetic-testing", split="10.repeat")


@pytest.fixture
def datasets(dataset1, dataset2):
    return [dataset1, dataset2]


# @pytest.mark.parametrize(
#     "domain_weights",
#     [
#         torch.tensor([0.7, 0.3]),
#         # NOTE: test auto fill samples if there are rounding errors
#         torch.tensor([0.296, 0.201, 0.501]),
#         torch.tensor(
#             [
#                 0.34356916553540745,
#                 0.16838812972610234,
#                 0.24711766854236725,
#                 0.0679225638705455,
#                 0.059079828519653675,
#                 0.043720261601881555,
#                 0.01653850841342608,
#                 0.00604146633842096,
#                 0.04342813428189645,
#                 0.0041942731702987,
#             ]
#         ),
#     ],
# )
# def test_sampling_from_dist_doremi_sampler(domain_weights: torch.Tensor, dataset1: DatasetDict | Dataset | IterableDatasetDict | IterableDataset):
#     global_batch_size = 512
#     num_microbatches = 32
#     batch_size = 4
#     dp_size = global_batch_size // (batch_size * num_microbatches)

#     datasets = [dataset1 for _ in range(len(domain_weights))]
#     domain_keys = [f"domain {i}" for i in range(len(datasets))]
#     doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)

#     init_distributed(tp=1, dp=dp_size, pp=1)(_test_sampling_from_dist_doremi_sampler)(
#         batch_size=batch_size,
#         datasets=datasets,
#         doremi_context=doremi_context,
#     )


# def _test_sampling_from_dist_doremi_sampler(
#     parallel_context: ParallelContext, batch_size: int, datasets: List[Dataset], doremi_context: DoReMiContext
# ):
#     dp_size = dist.get_world_size(parallel_context.dp_pg)
#     dp_rank = dist.get_rank(parallel_context.dp_pg)

#     sampler = DistributedSamplerForDoReMi(
#         datasets,
#         batch_size=batch_size,
#         num_replicas=dp_size,
#         rank=dp_rank,
#         doremi_context=doremi_context,
#         parallel_context=parallel_context,
#     )

#     domain_weights = doremi_context.domain_weights
#     batch_size_per_domain = [round(batch_size * weight.item()) for weight in domain_weights]
#     yielded_idxs = []

#     for idxs in sampler:
#         assert batch_size == len(idxs)

#         # NOTE: make sure the indicies from a batch
#         # is proportion to the domain weights
#         start_indices = [sum([len(ds) for ds in datasets[:i]]) for i in range(len(datasets))]
#         end_indices = [sum([len(ds) for ds in datasets[: i + 1]]) for i in range(len(datasets))]
#         for domain_idx, expected_batch_size in enumerate(batch_size_per_domain):
#             num_samples_per_domain = sum(
#                 1 for idx in idxs if idx >= start_indices[domain_idx] and idx < end_indices[domain_idx]
#             )

#             # NOTE: rounding errors
#             assert abs(expected_batch_size - num_samples_per_domain) <= 1

#         yielded_idxs.extend(idxs)


@pytest.mark.parametrize(
    "domain_weights",
    [
        # torch.tensor([0.7, 0.3]),
        # NOTE: test auto fill samples if there are rounding errors
        # torch.tensor([0.296, 0.201, 0.501]),
        # NOTE: if sampling based on batch size, then
        # the last domain results in no sample (round(0.004 * 64) = 0)
        # but if do with global batch size, (round(0.004 * 512) = 2)
        torch.tensor([0.498, 0.498, 0.004]),
        torch.tensor(
            [
                0.34356916553540745,
                0.16838812972610234,
                0.24711766854236725,
                0.0679225638705455,
                0.059079828519653675,
                0.043720261601881555,
                0.01653850841342608,
                0.00604146633842096,
                0.04342813428189645,
                0.0041942731702987,
            ]
        ),
    ],
)
def test_sampling_from_dist_doremi_sampler_with_global_batch_size(domain_weights: torch.Tensor, dataset1):
    global_batch_size = 512
    num_microbatches = 32
    batch_size = 4

    dp_size = global_batch_size // (batch_size * num_microbatches)
    # dp_size = global_batch_size // batch_size

    datasets = [dataset1 for _ in range(len(domain_weights))]
    domain_keys = [f"domain {i}" for i in range(len(datasets))]
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_sampling_from_dist_doremi_sampler_with_global_batch_size)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        global_batch_size=global_batch_size,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_sampling_from_dist_doremi_sampler_with_global_batch_size(
    parallel_context: ParallelContext,
    batch_size: int,
    num_microbatches: int,
    global_batch_size: int,
    datasets,
    doremi_context: DoReMiContext,
):
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    domain_weights = doremi_context.domain_weights
    [round(global_batch_size * weight.item()) for weight in domain_weights]

    microbatch_idx = 0
    num_samples_per_domain = [0 for _ in range(len(domain_weights))]
    for idxs in sampler:
        assert batch_size == len(idxs)

        # NOTE: make sure the indicies from a batch
        # is proportion to the domain weights
        start_indices = [sum([len(ds) for ds in datasets[:i]]) for i in range(len(datasets))]
        end_indices = [sum([len(ds) for ds in datasets[: i + 1]]) for i in range(len(datasets))]
        for domain_idx in range(len(domain_weights)):
            num_samples = sum(1 for idx in idxs if idx >= start_indices[domain_idx] and idx < end_indices[domain_idx])
            # num_samples_per_domain.append(num_samples)
            num_samples_per_domain[domain_idx] += num_samples

        print(f"microbatch_idx: {microbatch_idx}")
        if microbatch_idx == num_microbatches - 1:
            assert 1 == 1

        microbatch_idx += 1

        assert 1 == 1
        # num_samples_per_domain = torch.tensor(num_samples_per_domain, dtype=torch.int, device="cuda")

        # # NOTE: the domain weights are chosen so that we expect
        # # a domain have zero samples in a batch size
        # min_samples_per_domain = num_samples_per_domain.clone()
        # dist.all_reduce(min_samples_per_domain, op=dist.ReduceOp.MIN)
        # assert (min_samples_per_domain == 0).sum().item() > 0

        # # NOTE: the domain weights are chosen so that we expect
        # # no domains have zero sample in the global batch size
        # dist.all_reduce(num_samples_per_domain, op=dist.ReduceOp.SUM)
        # assert (num_samples_per_domain == 0).sum().item() == 0

        # for expected_bs, bs in zip(global_batch_size_per_domain, num_samples_per_domain):
        #     # NOTE: take into account rounding errors
        #     # can be accumulated across dp ranks
        #     assert abs(expected_bs - bs) < dp_size


def test_dist_doremi_sampler_sync_across_tp(datasets: list):
    num_microbatches = 32
    batch_size = 16

    domain_weights = torch.tensor([0.7, 0.3])
    domain_keys = [f"domain {i}" for i in range(len(datasets))]
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)

    init_distributed(tp=2, dp=1, pp=1)(_test_dist_doremi_sampler_sync_across_tp)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_dist_doremi_sampler_sync_across_tp(
    parallel_context: ParallelContext, batch_size: int, num_microbatches: int, datasets, doremi_context: DoReMiContext
):
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    tp_size = dist.get_world_size(parallel_context.tp_pg)
    yield_idxs = torch.tensor(list(sampler), device="cuda").view(-1)
    gathered_idxs = [torch.empty_like(yield_idxs, device="cuda") for _ in range(tp_size)]
    dist.all_gather(gathered_idxs, yield_idxs)
    assert all(torch.allclose(t1, t2) for t1, t2 in zip(gathered_idxs, gathered_idxs[1:]))


def test_dist_doremi_sampler_not_overlapse_across_dp(datasets: list):
    # batch_size = 100
    num_microbatches = 32
    batch_size = 16

    domain_weights = torch.tensor([0.7, 0.3])
    domain_keys = [f"domain {i}" for i in range(len(datasets))]
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)

    init_distributed(tp=1, dp=2, pp=1)(_test_dist_doremi_sampler_not_overlapse_across_dp)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_dist_doremi_sampler_not_overlapse_across_dp(
    parallel_context: ParallelContext, batch_size: int, num_microbatches: int, datasets, doremi_context: DoReMiContext
):
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    yield_idxs = torch.tensor(list(sampler), device="cuda").view(-1)
    gathered_idxs = [torch.empty_like(yield_idxs, device="cuda") for _ in range(dp_size)]
    dist.all_gather(gathered_idxs, yield_idxs)
    assert not torch.any(torch.isin(*gathered_idxs))


def test_determistic_doremi_sampler(datasets: list):
    # batch_size = 100
    num_microbatches = 32
    batch_size = 16

    domain_weights = torch.tensor([0.7, 0.3])
    domain_keys = [f"domain {i}" for i in range(len(datasets))]
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)
    n_epochs = 3

    init_distributed(tp=1, dp=1, pp=1)(_test_determistic_doremi_sampler)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
        n_epochs=n_epochs,
    )


def _test_determistic_doremi_sampler(
    parallel_context: ParallelContext,
    batch_size: int,
    num_microbatches: int,
    n_epochs: int,
    datasets,
    doremi_context: DoReMiContext,
):
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    idxs_per_epoch = []
    for _ in range(n_epochs):
        all_idxs = []
        for idxs in sampler:
            all_idxs.append(idxs)

        idxs_per_epoch.append(all_idxs)
        sampler.reset()

    # NOTE: check if the sequence of idxs across epochs are all the same
    assert all(
        all(arr1[i] == arr2[i] for i in range(len(arr1))) for arr1, arr2 in zip(idxs_per_epoch, idxs_per_epoch[1:])
    )
