import sys

import torch

sys.path.append("/fsx/phuc/projects/nanotron")


from utils import create_dummy_dataset

from examples.doremi.doremi.utils import compute_domain_weights_based_on_token_count


def test_compute_domain_weights_based_on_token_count():
    datasets = [
        create_dummy_dataset(10),
        create_dummy_dataset(20),
        create_dummy_dataset(70),
    ]

    domain_weights = compute_domain_weights_based_on_token_count(datasets)

    assert torch.equal(domain_weights, torch.tensor([0.1, 0.2, 0.7]))
    assert torch.allclose(domain_weights.sum(), torch.tensor(1.0))
