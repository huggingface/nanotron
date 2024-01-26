from typing import List

import pytest
import torch
from datasets import load_dataset
from helpers.utils import init_distributed
from nanotron import distributed as dist
from nanotron.doremi.dataloader import DistributedSamplerForDoReMi
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.parallel import ParallelContext
from torch.utils.data import Dataset


@pytest.fixture
def dataset1():
    return load_dataset("stas/c4-en-10k", split="train")


@pytest.fixture
def dataset2():
    return load_dataset("stas/openwebtext-synthetic-testing", split="10.repeat")


# @pytest.mark.parametrize(
#     "tp,dp,pp",
#     [
#         pytest.param(*all_3d_configs)
#         for gpus in range(1, min(available_gpus(), 4) + 1)
#         for all_3d_configs in get_all_3d_configurations(gpus)
#     ],
# )
@pytest.mark.parametrize(
    "domain_weights",
    [
        torch.tensor([0.7, 0.3]),
        # NOTE: test auto fill samples if there are rounding errors
        torch.tensor([0.496, 0.5]),
    ],
)
def test_sampling_from_dist_doremi_sampler(domain_weights, dataset1, dataset2):
    batch_size = 100
    datasets = [dataset1, dataset1]
    # domain_weights = torch.tensor([0.7, 0.3])
    domain_keys = [f"domain {i}" for i in range(len(datasets))]
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)

    init_distributed(tp=1, dp=1, pp=1)(_test_sampling_from_dist_doremi_sampler)(
        batch_size=batch_size,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_sampling_from_dist_doremi_sampler(
    parallel_context: ParallelContext, batch_size: int, datasets: List[Dataset], doremi_context: DoReMiContext
):
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )
    # for idxs in sampler:
    #     assert 1 == 1

    # assert abs(batch_size - len(next(iter(sampler)))) < 2

    # NOTE: make sure the indicies from a batch is proportion
    # to the domain weights
    domain_weights = doremi_context.domain_weights
    # idxs = list(iter(sampler))[0]

    # assert sum(1 for idx in idxs if idx < len(datasets[0])) ==  int((batch_size * domain_weights[i].item()))

    yielded_idxs = []
    # num_samples_per_domain = [0 for _ in range(len(datasets))]
    domain_batch_size = [round(batch_size * weight.item()) for weight in domain_weights]

    for idxs in sampler:
        assert batch_size == len(idxs)

        num_sample_domain_0 = sum(1 for idx in idxs if idx < len(datasets[0]))
        num_sample_domain_1 = sum(1 for idx in idxs if idx >= len(datasets[1]))

        assert domain_batch_size[0] == num_sample_domain_0
        assert domain_batch_size[1] == num_sample_domain_1

        yielded_idxs.extend(idxs)
