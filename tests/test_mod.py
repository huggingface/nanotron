import torch
from torch import nn
import pytest

from nanotron.mod import MixtureOfDepth, Router


@pytest.mark.parametrize("seq_len, top_k", [(1, 1), (10, 5), (10, 10)])
def test_mod(seq_len, top_k):
    BATCH_SIZE = 15
    D_MODEL = 1024
        
    linear = nn.Linear(D_MODEL, D_MODEL)
    block = MixtureOfDepth(top_k, D_MODEL, linear)
    
    inputs = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
    ref_inputs = inputs.clone()
    outputs = block(inputs)
    
    expected_num_tokens_not_changed = (seq_len - top_k) * BATCH_SIZE
    num_tokens_not_changed = torch.eq(outputs.view(-1, D_MODEL), ref_inputs.view(-1, D_MODEL)).all(dim=1).sum().item()
    
    assert outputs.shape == linear(ref_inputs).shape
    assert num_tokens_not_changed == expected_num_tokens_not_changed, f"num_tokens_not_changed: {num_tokens_not_changed}, expected: {expected_num_tokens_not_changed}"
    

@pytest.mark.parametrize("capacity, d_model", [(1, 64), (10, 64)])
def test_router(capacity, d_model):
    BATCH_SIZE, SEQ_LEN = 5, 10
    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, d_model)
    
    router = Router(capacity, d_model)
    selected_idxs = router(inputs)
    
    assert selected_idxs.shape == (BATCH_SIZE, capacity)
    assert selected_idxs.dtype == torch.int64
