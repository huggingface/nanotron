import torch
from datasets import load_dataset
from nanotron.doremi.utils import compute_domain_weights_based_on_token_count


def test_compute_domain_weights_based_on_token_count():
    datasets = [
        load_dataset("stas/c4-en-10k", split="train[:10]"),
        load_dataset("stas/c4-en-10k", split="train[:20]"),
        load_dataset("stas/c4-en-10k", split="train[:70]"),
    ]

    domain_weights = compute_domain_weights_based_on_token_count(datasets)

    assert torch.equal(domain_weights, torch.tensor([0.1, 0.2, 0.7]))
    assert torch.allclose(domain_weights.sum(), torch.tensor(1.0))
