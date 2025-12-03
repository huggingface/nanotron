"""Test script for ClimLlamaDataset.

This script tests the ClimLlamaDataset on the data/combined_interleave_256 dataset.
"""

import json
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock


def dump_sample(item, filename="sample_0.json"):
    """Dump dataset sample contents to a JSON file.

    Args:
        item: A dataset sample (e.g., dataset[0])
        filename: Output JSON filename (default: sample_0.json)
    """
    # Convert numpy arrays and tensors to lists for JSON serialization
    serializable_item = {}
    for key, value in item.items():
        if isinstance(value, np.ndarray):
            serializable_item[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_item[key] = value.cpu().numpy().tolist()
        else:
            serializable_item[key] = value

    with open(filename, "w") as f:
        json.dump(serializable_item, f, indent=None)
    print(f"   Sample dumped to {filename}")


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
    """Mock dataset configuration matching ClimLlamaDatasetsArgs."""

    # Variable names for position embeddings (must match model config)
    variables: tuple = (
        "unk",
        "z",
        "t",
        "q",
        "u",
        "v",
        "w",
        "t2m",
        "msl",
        "u10",
        "v10",
        "tp",
        "tp_6h",
    )
    # Size of the VQ-VAE codebook for special token generation
    codebook_size: int = 32768


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
    )
    print(f"   Dataset created successfully!")
    print(f"   Dataset length: {len(dataset)}")

    # Test __getitem__
    print("\n4. Testing __getitem__ with whole documents...")
    item = dataset[0]
    dump_sample(item, "sample_0.json")
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


def test_blendable_climllama_dataset():
    """Test ClimLlamaDataset with multiple .bin files using BlendableDataset."""
    # Set up paths - using the same file twice to test blending logic
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/combined_interleave_256",
    )
    data_prefix_1 = os.path.join(data_path, "1")
    data_prefix_2 = os.path.join(data_path, "2")  # Using same file for testing

    print(f"\nTesting BlendableDataset with ClimLlamaDataset")
    print(f"  data_prefix_1: {data_prefix_1}")
    print(f"  data_prefix_2: {data_prefix_2}")

    # Check if data exists
    if not os.path.exists(f"{data_prefix_1}.bin"):
        print(f"ERROR: Data file not found at {data_prefix_1}.bin")
        print("Please ensure the test data is available.")
        return False

    from nanotron.data.climllama_dataset import build_climllama_dataset
    from nanotron.data.nemo_dataset.blendable_dataset import BlendableDataset

    # Create parallel context
    print("\n1. Creating ParallelContext...")
    world_pg = dist.distributed_c10d._get_default_group()
    parallel_context = SimpleParallelContext(world_pg)
    tokenizer = MagicMock()
    cfg = MockDatasetConfig()

    # Test 1: Multiple paths without weights (equal weights)
    print("\n2. Testing with list of paths (equal weights)...")
    data_prefix_list = [data_prefix_1, data_prefix_2]
    dataset = build_climllama_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        data_prefix=data_prefix_list,
        num_samples=100,
        seq_length=256,
        seed=42,
        parallel_context=parallel_context,
        name="test",
        drop_last=True,
    )
    print(f"   Dataset type: {type(dataset).__name__}")
    assert isinstance(dataset, BlendableDataset), "Expected BlendableDataset for multiple prefixes"
    print(f"   Dataset length: {len(dataset)}")
    print(f"   Number of sub-datasets: {len(dataset.datasets)}")

    # Test __getitem__
    print("\n3. Testing __getitem__ on BlendableDataset...")
    item = dataset[0]
    print(f"   Item keys: {item.keys()}")
    print(f"   input_ids shape: {item['input_ids'].shape}")
    print(f"   var_idx shape: {item['var_idx'].shape}")
    print(f"   spatial_temporal_features shape: {item['spatial_temporal_features'].shape}")

    # Test multiple items from blended dataset
    print("\n4. Testing multiple items from BlendableDataset...")
    dataset_indices_used = {0: 0, 1: 0}
    for i in range(min(20, len(dataset))):
        item = dataset[i]
        ds_idx = dataset.dataset_index[i]
        sample_idx = dataset.dataset_sample_index[i]
        dataset_indices_used[ds_idx] += 1
        if i < 10:
            print(
                f"   Item {i:2d}: dataset_idx={ds_idx}, sample_idx={sample_idx:3d}, "
                f"len={len(item['input_ids']):5d}"
            )
    print(f"\n   Dataset usage: {dict(dataset_indices_used)}")

    # Test 2: Blended format with weights [weight1, path1, weight2, path2]
    print("\n5. Testing with blended format (70/30 weights)...")
    data_prefix_weighted = [0.7, data_prefix_1, 0.3, data_prefix_2]
    dataset_weighted = build_climllama_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        data_prefix=data_prefix_weighted,
        num_samples=100,
        seq_length=256,
        seed=42,
        parallel_context=parallel_context,
        name="test",
        drop_last=True,
    )
    print(f"   Dataset type: {type(dataset_weighted).__name__}")
    assert isinstance(dataset_weighted, BlendableDataset), "Expected BlendableDataset for weighted prefixes"
    print(f"   Dataset length: {len(dataset_weighted)}")

    # Verify weights are being used
    dataset_indices_weighted = {0: 0, 1: 0}
    for i in range(len(dataset_weighted)):
        ds_idx = dataset_weighted.dataset_index[i]
        dataset_indices_weighted[ds_idx] += 1
    print(f"   Dataset usage with weights: {dict(dataset_indices_weighted)}")
    ratio = dataset_indices_weighted[0] / (dataset_indices_weighted[0] + dataset_indices_weighted[1])
    print(f"   Actual ratio for dataset 0: {ratio:.2f} (expected ~0.70)")

    # Test 3: Single prefix (should return ClimLlamaDataset, not BlendableDataset)
    print("\n6. Testing with single prefix (should NOT be BlendableDataset)...")
    from nanotron.data.climllama_dataset import ClimLlamaDataset

    dataset_single = build_climllama_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        data_prefix=data_prefix_1,
        num_samples=100,
        seq_length=256,
        seed=42,
        parallel_context=parallel_context,
        name="test",
        drop_last=True,
    )
    print(f"   Dataset type: {type(dataset_single).__name__}")
    assert isinstance(dataset_single, ClimLlamaDataset), "Expected ClimLlamaDataset for single prefix"
    print(f"   Dataset length: {len(dataset_single)}")

    # Test 4: List with single prefix (should also return ClimLlamaDataset)
    print("\n7. Testing with list containing single prefix...")
    dataset_single_list = build_climllama_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        data_prefix=[data_prefix_1],
        num_samples=100,
        seq_length=256,
        seed=42,
        parallel_context=parallel_context,
        name="test",
        drop_last=True,
    )
    print(f"   Dataset type: {type(dataset_single_list).__name__}")
    assert isinstance(dataset_single_list, ClimLlamaDataset), "Expected ClimLlamaDataset for single-item list"

    print("\n✓ BlendableDataset tests completed!")
    return True


def main():
    """Main entry point."""
    init_distributed()
    try:
        success = test_climllama_dataset()
        if success:
            success = test_blendable_climllama_dataset()
        sys.exit(0 if success else 1)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
