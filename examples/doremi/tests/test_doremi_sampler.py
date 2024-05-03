import pytest
import torch
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from torch.utils.data import DataLoader
from utils import create_dummy_dataset, set_system_path

set_system_path()

from examples.doremi.doremi.dataloader import (
    CombinedDataset,
    DistributedSamplerForDoReMi,
)
from examples.doremi.doremi.doremi_context import DoReMiContext
from tests.helpers.utils import init_distributed


@pytest.fixture
def dataset1():
    return create_dummy_dataset(7000)


@pytest.fixture
def dataset2():
    return create_dummy_dataset(3000)


@pytest.fixture
def datasets(dataset1, dataset2):
    return [dataset1, dataset2]


@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_dist_doremi_sampler_sync_across_tp(num_microbatches, dataset1, is_proxy):
    NUM_DOMAINS = 2
    BATCH_SIZE = 16

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=2, dp=1, pp=1)(_test_dist_doremi_sampler_sync_across_tp)(
        batch_size=BATCH_SIZE,
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

    for idxs in sampler:
        idxs = torch.tensor(idxs, device="cuda")
        assert_tensor_synced_across_pg(idxs, parallel_context.tp_pg)


@pytest.mark.parametrize("dp_size", [2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_dist_doremi_sampler_not_overlapse_across_dp_for_proxy_training(dp_size, num_microbatches, dataset1, is_proxy):
    NUM_DOMAINS = 2
    GLOBAL_BATCH_SIZE = 512
    batch_size = GLOBAL_BATCH_SIZE // (num_microbatches * dp_size)

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=2, pp=1)(_test_dist_doremi_sampler_not_overlapse_across_dp_for_proxy_training)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_dist_doremi_sampler_not_overlapse_across_dp_for_proxy_training(
    parallel_context: ParallelContext,
    batch_size: int,
    num_microbatches: int,
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

    for idxs in sampler:
        idxs = torch.tensor(idxs, device="cuda").view(-1)

        # NOTE: i tried to use assert_fail_except_rank_with, but it mark the test as failed
        # even the test raises an exception as expected
        gathered_idxs = [torch.empty_like(idxs, device="cuda") for _ in range(dp_size)]
        dist.all_gather(gathered_idxs, idxs)

        # NOTE: whether proxy or reference training
        # the idxs should not be overlapse
        assert not torch.any(torch.isin(*gathered_idxs))


@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_determistic_doremi_sampler(num_microbatches, dataset1, is_proxy):
    BATCH_SIZE = 100
    NUM_DOMAINS = 2

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)
    n_epochs = 3

    init_distributed(tp=1, dp=1, pp=1)(_test_determistic_doremi_sampler)(
        batch_size=BATCH_SIZE,
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


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_sampling_from_dist_doremi_sampler_with_global_batch_size(
    dp_size,
    num_microbatches,
    # domain_weights: torch.Tensor,
    dataset1,
    is_proxy,
):
    NUM_DOMAINS = 8
    GLOBAL_BATCH_SIZE = 512
    batch_size = GLOBAL_BATCH_SIZE // (num_microbatches * dp_size)

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_sampling_from_dist_doremi_sampler_with_global_batch_size)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        global_batch_size=GLOBAL_BATCH_SIZE,
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
    global_batch_size_per_domain = [round(global_batch_size * weight.item()) for weight in domain_weights]

    microbatch_idx = 0
    num_samples_per_domain = [0 for _ in range(len(domain_weights))]
    for idxs in sampler:
        assert batch_size == len(idxs)

        # NOTE: make sure the indices from a batch
        # is proportion to the domain weights
        start_indices = [sum([len(ds) for ds in datasets[:i]]) for i in range(len(datasets))]
        end_indices = [sum([len(ds) for ds in datasets[: i + 1]]) for i in range(len(datasets))]
        for domain_idx in range(len(domain_weights)):
            num_samples = sum(1 for idx in idxs if idx >= start_indices[domain_idx] and idx < end_indices[domain_idx])
            num_samples_per_domain[domain_idx] += num_samples

        if microbatch_idx == num_microbatches - 1:
            # NOTE: if this is the last microbatch => we iterate through all the microbatches
            # now we check if the overall number of samples in each domain is correct across
            # all the microbatches
            num_samples_per_domain = torch.tensor(num_samples_per_domain, dtype=torch.int, device="cuda")

            # NOTE: the domain weights are chosen so that we expect
            # no domains have zero sample in the global batch size
            dist.all_reduce(num_samples_per_domain, op=dist.ReduceOp.SUM)
            assert (num_samples_per_domain == 0).sum().item() == 0

            for expected_bs, bs in zip(global_batch_size_per_domain, num_samples_per_domain):
                assert bs > 0
                # NOTE: take into account rounding errors
                # across all the dp ranks
                assert abs(expected_bs - bs) <= dp_size, f"abs(expected_bs - bs): {abs(expected_bs - bs)}"

            microbatch_idx = 0
            num_samples_per_domain = [0 for _ in range(len(domain_weights))]
        else:
            microbatch_idx += 1


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_dist_doremi_sampler_not_repeating_samples(dp_size, num_microbatches, dataset1, is_proxy):
    NUM_DOMAINS = 2
    GLOBAL_BATCH_SIZE = 512
    batch_size = GLOBAL_BATCH_SIZE // (num_microbatches * dp_size)

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_dist_doremi_sampler_not_repeating_samples)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_dist_doremi_sampler_not_repeating_samples(
    parallel_context: ParallelContext,
    batch_size: int,
    num_microbatches: int,
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

    local_yieled_idxs = []
    yielded_idxs = []
    epoch = 0
    for idxs in sampler:
        # NOTE: check that the indices are not repeated
        assert not set(idxs).intersection(
            local_yieled_idxs
        ), f"set(idxs): {set(idxs)}, local_yieled_idxs: {local_yieled_idxs}"
        assert not set(idxs).intersection(
            yielded_idxs
        ), f"set(idxs): {set(idxs)}, yielded_idxs: {yielded_idxs} \
        epoch: {epoch}"

        local_yieled_idxs.extend(idxs)

        # NOTE: gather all the indices from all the dp ranks
        idxs = torch.tensor(idxs, dtype=torch.int, device="cuda")
        all_idxs = [torch.zeros_like(idxs) for _ in range(dp_size)]
        dist.all_gather(all_idxs, idxs)
        all_idxs = torch.cat(all_idxs, dim=0).view(-1).cpu().tolist()
        yielded_idxs.extend(all_idxs)
        epoch += 1

    assert len(set(yielded_idxs)) == len(yielded_idxs)


@pytest.mark.parametrize("dp_size", [2, 4, 8])
@pytest.mark.parametrize("num_microbatches", [1, 5])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_yielding(dp_size, num_microbatches, dataset1, is_proxy):
    NUM_DOMAINS = 2
    BATCH_SIZE = 100
    global_batch_size = BATCH_SIZE * num_microbatches * dp_size

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_yielding)(
        batch_size=BATCH_SIZE,
        global_batch_size=global_batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_yielding(
    parallel_context: ParallelContext,
    batch_size: int,
    global_batch_size: int,
    num_microbatches: int,
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

    step = 0
    num_yielded_microbatches = 0
    expected_domain_weights = torch.tensor([0.5, 0.5])

    for idxs in sampler:
        idxs = torch.tensor(idxs, dtype=torch.int, device="cuda")
        idxs_dp = [torch.empty_like(idxs) for _ in range(dp_size)]
        dist.all_gather(idxs_dp, idxs)
        idxs_dp = torch.cat(idxs_dp, dim=0)

        assert idxs_dp.numel() == batch_size * dp_size

        # NOTE: if it loops through all the microbatches
        # then we check if the number of samples in each domain
        if (step + 1) % num_microbatches == 0:
            num_yielded_microbatches += 1
            for i, weight in enumerate(expected_domain_weights):
                assert sampler.domain_counters[i] == int(num_yielded_microbatches * global_batch_size * weight)

        step += 1


@pytest.mark.parametrize("dp_size", [2, 4, 8])
@pytest.mark.parametrize("num_microbatches", [1, 5])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_yielding_with_dataloader(dp_size, num_microbatches, dataset1, is_proxy):
    NUM_DOMAINS = 2
    BATCH_SIZE = 100
    global_batch_size = BATCH_SIZE * num_microbatches * dp_size

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_yielding_with_dataloader)(
        batch_size=BATCH_SIZE,
        global_batch_size=global_batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_yielding_with_dataloader(
    parallel_context: ParallelContext,
    batch_size: int,
    global_batch_size: int,
    num_microbatches: int,
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
    comebined_dataset = CombinedDataset(datasets)
    dataloader = DataLoader(comebined_dataset, batch_sampler=sampler)

    step = 1
    num_yielded_microbatches = 0
    expected_domain_weights = torch.tensor([0.5, 0.5])

    for idxs in dataloader:
        num_idxs = torch.tensor(len(idxs["text"]), dtype=torch.int, device="cuda")
        assert num_idxs.item() == batch_size

        dist.all_reduce(num_idxs, op=dist.ReduceOp.SUM, group=parallel_context.dp_pg)
        assert num_idxs == batch_size * dp_size

        if step % num_microbatches == 0:
            num_yielded_microbatches += 1
            for i, weight in enumerate(expected_domain_weights):
                assert sampler.domain_counters[i] == int(num_yielded_microbatches * global_batch_size * weight)

        step += 1

    assert step > 1
