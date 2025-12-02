"""Test script for ClimLlamaDataset.

This script tests the ClimLlamaDataset on the data/combined_interleave_256 dataset.
"""

import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock


def init_distributed():
    """Initialize distributed environment for single-process testing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29515"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def cleanup_distributed():
    """Clean up distributed environment."""
    dist.destroy_process_group()


class SimpleParallelContext:
    """Simple parallel context for single-process testing."""

    def __init__(self, world_pg):
        self.world_pg = world_pg
        self.dp_pg = world_pg
        self.tp_pg = world_pg
        self.pp_pg = world_pg
        self.expert_pg = world_pg
        self.world_rank = dist.get_rank(world_pg)
        self.dp_rank = 0
        self.tp_rank = 0
        self.pp_rank = 0


@dataclass
class MockDatasetConfig:
    """Mock dataset configuration."""

    index_mapping_dir: Optional[str] = None


def test_climllama_dataset():
    """Test ClimLlamaDataset on data/combined_interleave_256."""
    # Set up paths
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/combined_interleave_256",
    )
    data_prefix = os.path.join(data_path, "1")

    print(f"Testing ClimLlamaDataset with data_prefix: {data_prefix}")

    # Check if data exists
    if not os.path.exists(f"{data_prefix}.bin"):
        print(f"ERROR: Data file not found at {data_prefix}.bin")
        print("Please ensure the test data is available.")
        return False

    from nanotron.data.nemo_dataset import get_indexed_dataset
    from nanotron.data.climllama_dataset import ClimLlamaDataset

    # Load indexed dataset
    print("\n1. Loading indexed dataset...")
    indexed_dataset = get_indexed_dataset(data_prefix, skip_warmup=False)
    # doc_idx has length (num_documents + 1), so num_documents = len(doc_idx) - 1
    total_num_of_documents = len(indexed_dataset.doc_idx) - 1
    total_num_of_sequences = indexed_dataset.sizes.shape[0]
    print(f"   Documents: {total_num_of_documents}")
    print(f"   Sequences: {total_num_of_sequences}")
    print(f"   First 5 sequence sizes: {indexed_dataset.sizes[:5]}")

    # Create parallel context
    print("\n2. Creating ParallelContext...")
    world_pg = dist.distributed_c10d._get_default_group()
    parallel_context = SimpleParallelContext(world_pg)
    tokenizer = MagicMock()
    cfg = MockDatasetConfig()

    # Build dataset
    print("\n3. Building ClimLlamaDataset...")
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)

    dataset = ClimLlamaDataset(
        cfg=cfg,
        tokenizer=tokenizer,
        name="test",
        data_prefix=data_prefix,
        documents=documents,
        indexed_dataset=indexed_dataset,
        num_samples=100,
        seq_length=256,
        seed=42,
        parallel_context=parallel_context,
        drop_last=True,
        codebook_size=32768,
    )
    print(f"   Dataset created successfully!")
    print(f"   Dataset length: {len(dataset)}")

    # Test __getitem__
    print("\n4. Testing __getitem__ with whole documents...")
    item = dataset[0]
    print(f"   Item keys: {item.keys()}")
    print(f"   input_ids shape: {item['input_ids'].shape}")
    print(f"   var_idx shape: {item['var_idx'].shape}")
    print(f"   spatial_temporal_features shape: {item['spatial_temporal_features'].shape}")

    # Check values
    print(f"\n   var_idx unique values: {np.unique(item['var_idx'])}")
    print(f"   res_idx unique values: {np.unique(item['res_idx'])}")
    print(f"   leadtime_idx unique values: {np.unique(item['leadtime_idx'])}")
    print(f"   spatial_temporal_features stats:")
    for i, name in enumerate(
        ["x", "y", "z", "cos_hour", "sin_hour", "cos_day", "sin_day"]
    ):
        col = item["spatial_temporal_features"][:, i]
        print(f"     {name}: min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}")

    # Test multiple items
    print("\n5. Testing multiple items...")
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        doc_idx = dataset.shuffle_idx[i]
        print(
            f"   Item {i:2d} (doc {doc_idx:3d}): len={len(item['input_ids']):5d}, "
            f"var_idx unique={len(np.unique(item['var_idx']))}, "
            f"res_idx unique={np.unique(item['res_idx'])}"
        )

    # Analyze sequence structure (not documents)
    print("\n6. Analyzing sequence structure...")
    from atmtokenizer.eval.special_tokens import create_special_tokens, print_special_tokens

    st = create_special_tokens(codebook_size=32768)

    eot_seqs = 0
    eor_seqs = 0
    other_seqs = 0

    for seq_id in range(total_num_of_sequences):
        seq = indexed_dataset[seq_id]
        first_token = seq[0]
        last_token = seq[-1]

        if first_token == st.token_endoftime:
            eot_seqs += 1
        elif last_token == st.token_endofres:
            eor_seqs += 1
        else:
            other_seqs += 1

    print(f"   Resolution blocks (ending with EOR): {eor_seqs}")
    print(f"   EOT markers (starting with EOT): {eot_seqs}")
    print(f"   Other: {other_seqs}")

    print("\n✓ All tests completed!")
    return True


def main():
    """Main entry point."""
    init_distributed()
    try:
        success = test_climllama_dataset()
        sys.exit(0 if success else 1)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
