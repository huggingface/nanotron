"""
# run with pytest
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 --max_restarts=0 --tee=3 \
    --module pytest tests/test_module/test_mla.py

torchrun --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 --max_restarts=0 --tee=3 \
    tests/test_module/test_mla.py
"""
import pytest
import torch
from helpers.utils import (
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.config import LlamaConfig, ModelArgs, ParallelismArgs
from nanotron.models.base import init_on_device_and_dtype
from nanotron.models.llama import MLA
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode


@pytest.mark.fa2
@pytest.mark.parametrize("tp", [1, 2, 4])
@rerun_if_address_is_in_use()
def test_mla_output_shape(tp: int):
    model_args = LlamaConfig(
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=192,
    )
    init_distributed(tp=tp, dp=1, pp=1)(_test_mla_output_shape)(model_args=model_args)


def _test_mla_output_shape(parallel_context: ParallelContext, model_args: ModelArgs):
    parallel_config = ParallelismArgs(
        dp=1, pp=1, tp=parallel_context.tensor_parallel_size, tp_mode=TensorParallelLinearMode.ALL_REDUCE
    )

    with init_on_device_and_dtype(device="cuda", dtype=torch.bfloat16):
        mla = MLA(config=model_args, parallel_config=parallel_config, tp_pg=parallel_context.tp_pg, layer_idx=0)

    bs, seq_len, dim = 4, model_args.max_position_embeddings, model_args.hidden_size
    input_tensor = torch.randn(seq_len, bs, dim, device="cuda", dtype=torch.bfloat16)
    sequence_mask = torch.ones(bs, seq_len, device=input_tensor.device)

    mla_output = mla(input_tensor, sequence_mask=sequence_mask)

    assert mla_output["hidden_states"].shape == (seq_len, bs, dim), "MLA output shape should be (seq_len, bs, dim)"

    parallel_context.destroy()
