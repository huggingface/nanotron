"""
# run with pytest
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 --max_restarts=0 --tee=3 \
    --module pytest tests/test_module/test_mla.py

torchrun --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 --max_restarts=0 --tee=3 \
    tests/test_module/test_mla.py
"""
import os
import sys
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from nanotron.config import ParallelismArgs
from nanotron.models.llama import MLA
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.utils import init_distributed  # noqa: E402


@dataclass
class ModelArgs:
    max_batch_size: int = 4
    max_position_embeddings: int = 4096
    hidden_size: int = 2048
    num_attention_heads: int = 16
    # mla
    q_lora_rank: int = 512
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    is_using_mup: bool = False


def setup_module():
    """Setup function that runs once before all tests"""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_default_device(torch.device(f"cuda:{local_rank}"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def teardown_module():
    """Teardown function that runs once after all tests"""
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.fa2
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_mla_output_shape(tp_size):
    init_distributed(tp=tp_size, dp=1, pp=1)(_test_mla_output_shape)


def _test_mla_output_shape(tp_size):
    """
    Test the output shape of the MLA module.
    """
    # Initialize
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    model_args = ModelArgs()
    parallel_config = ParallelismArgs(dp=1, pp=1, tp=tp_size, tp_mode=TensorParallelLinearMode.ALL_REDUCE)
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    mla = MLA(config=model_args, parallel_config=parallel_config, tp_pg=parallel_context.tp_pg, layer_idx=0)

    # input
    bs, seq_len, dim = model_args.max_batch_size, model_args.max_position_embeddings, model_args.hidden_size
    input_tensor = torch.randn(seq_len, bs, dim)
    sequence_mask = torch.ones(bs, seq_len, device=input_tensor.device)

    mla_output = mla(input_tensor, sequence_mask=sequence_mask)

    assert mla_output["hidden_states"].shape == (seq_len, bs, dim), "MLA output shape should be (seq_len, bs, dim)"


if __name__ == "__main__":
    # debug purpose
    world_size = int(os.environ["WORLD_SIZE"])
    test_mla_output_shape(tp_size=world_size)
    print("Test passed")
