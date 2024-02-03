import pytest
import torch
from nanotron.doremi.doremi_context import DoReMiContext


def test_initialization():
    domain_weights = torch.tensor([0.3, 0.7])
    domain_keys = ["domain1", "domain2"]
    step_size, smoothing_param = 0.01, 0.001
    is_proxy = False
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy, step_size, smoothing_param=smoothing_param)

    assert torch.equal(doremi_context.domain_weights, domain_weights)
    assert doremi_context.domain_keys == domain_keys
    assert doremi_context.is_proxy == is_proxy
    assert doremi_context.step_size == step_size
    assert doremi_context.smoothing_param == smoothing_param


def test_num_domains():
    domain_weights = torch.tensor([0.3, 0.7])
    domain_keys = ["domain1", "domain2"]
    context = DoReMiContext(domain_weights, domain_keys, False)
    assert context.num_domains == 2


def test_get_domain_name():
    domain_weights = torch.tensor([0.3, 0.7])
    domain_keys = ["domain1", "domain2"]
    context = DoReMiContext(domain_weights, domain_keys, False)
    assert context.get_domain_name(0) == "domain1"
    assert context.get_domain_name(1) == "domain2"


def test_domain_keys_length():
    domain_weights = torch.tensor([[0.1, 0.3, 0.6]])
    domain_keys = ["domain1"]
    with pytest.raises(AssertionError):
        DoReMiContext(domain_weights, domain_keys, False)


def test_record_domain_weights_history():
    domain_weights = [torch.tensor([0.1, 0.3, 0.6]), torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.3, 0.3, 0.4])]
    domain_keys = ["domain1", "domain2", "domain3"]

    doremi_context = DoReMiContext(domain_weights[0], domain_keys, False)

    assert torch.equal(doremi_context.domain_weights, domain_weights[0])

    doremi_context.add_weight_with_history(domain_weights[1], 1)
    assert torch.equal(doremi_context.domain_weights, domain_weights[1])
    doremi_context.add_weight_with_history(domain_weights[2], 2)
    assert torch.equal(doremi_context.domain_weights, domain_weights[2])

    for i, history in enumerate(doremi_context.domain_weight_history):
        assert history["step"] == i
        assert torch.equal(history["domain_weights"], domain_weights[i])


def test_domain_weights_sum():
    with pytest.raises(AssertionError):
        DoReMiContext(torch.tensor([0.5, 0.6]), ["a", "b"], False)


def test_update_weights():
    context = DoReMiContext(torch.tensor([0.5, 0.5]), ["a", "b"], False)
    new_weights = torch.tensor([0.4, 0.6])
    context.domain_weights = new_weights
    assert torch.equal(context.domain_weights, new_weights)
