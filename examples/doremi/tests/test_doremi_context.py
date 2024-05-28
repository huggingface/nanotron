from copy import deepcopy

import torch
from utils import set_system_path

set_system_path()

from examples.doremi.doremi.doremi_context import DoReMiContext


def test_initialization():
    domain_keys = ["domain1", "domain2"]
    step_size, smoothing_param = 0.01, 0.001
    is_proxy = False
    doremi_context = DoReMiContext(domain_keys, is_proxy, step_size, smoothing_param=smoothing_param)

    assert torch.equal(doremi_context.domain_weights, torch.tensor([0.5, 0.5]))
    assert doremi_context.domain_keys == domain_keys
    assert doremi_context.is_proxy == is_proxy
    assert doremi_context.step_size == step_size
    assert doremi_context.smoothing_param == smoothing_param


def test_num_domains():
    domain_keys = ["domain1", "domain2"]
    context = DoReMiContext(domain_keys, False)
    assert context.num_domains == 2


def test_get_domain_name():
    domain_keys = ["domain1", "domain2"]
    context = DoReMiContext(domain_keys, False)
    assert context.get_domain_name(0) == "domain1"
    assert context.get_domain_name(1) == "domain2"


def test_record_domain_weights_history():
    domain_weights = [torch.tensor([0.1, 0.3, 0.6]), torch.tensor([0.2, 0.3, 0.5])]
    domain_keys = ["domain1", "domain2", "domain3"]

    doremi_context = DoReMiContext(domain_keys, False)
    initial_domain_weights = deepcopy(doremi_context.domain_weights)

    doremi_context.add_weight_with_history(domain_weights[0], step=1)
    doremi_context.add_weight_with_history(domain_weights[1], step=2)
    assert torch.equal(initial_domain_weights, doremi_context.domain_weights)

    expected_weight_history = [initial_domain_weights, *domain_weights]

    for i, history in enumerate(doremi_context.domain_weight_history):
        assert history["step"] == i
        assert torch.equal(history["weight"], expected_weight_history[i])
