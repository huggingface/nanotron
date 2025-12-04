"""Parse weaved tokens from a ClimLlama dataset and dump samples for inspection."""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from unittest.mock import MagicMock

import nanotron.mock_flash_attn  # Ensure flash_attn is available


def dump_sample(item, parser, var_level_names, cfg=None, filename="sample_0.json"):
    """Dump dataset sample contents to a JSON file."""
    import json
    from lark.tree import Tree
    from lark.lexer import Token
    import pandas as pd

    def tree_to_dict(node, df):
        if isinstance(node, Tree):
            return {
                "type": node.data,
                "children": [tree_to_dict(child, df) for child in node.children],
            }
        assert isinstance(node, Token), f"Expected Token node, got {type(node)}"
        idx = node.start_pos
        assert idx is not None, "Token node missing start_pos"
        row_dict = df.iloc[idx].to_dict() if idx < len(df) else {}
        return {"pos": idx, "token": node.value, "type": node.type, **row_dict}

    data = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                data[key] = value
            elif value.ndim == 2:
                for i in range(value.shape[1]):
                    if key == "spatial_temporal_features":
                        col_names = [
                            "x",
                            "y",
                            "z",
                            "cos_hour",
                            "sin_hour",
                            "cos_day",
                            "sin_day",
                            "log10_level_hPa",
                        ]
                        col_name = f"{key}_{col_names[i]}" if i < len(col_names) else f"{key}_{i}"
                    else:
                        col_name = f"{key}_{i}"
                    data[col_name] = value[:, i]
    df = pd.DataFrame(data)

    serializable_item = {"var_level_name": var_level_names}

    assert parser is not None, "Parser is required for parsing input_ids"
    input_ids = item.get("input_ids")
    assert input_ids is not None, "input_ids not found in item for parsing"
    tree = parser.parse(input_ids)
    serializable_item["parsed_tree"] = tree_to_dict(tree, df)

    if cfg is not None:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(cfg):
            serializable_item["cfg"] = asdict(cfg) # type: ignore
        elif hasattr(cfg, "__dict__"):
            serializable_item["cfg"] = cfg.__dict__
        else:
            serializable_item["cfg"] = str(cfg)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(serializable_item, f, indent=None)
    print(f"Sample dumped to {filename}")


def init_distributed():
    """Initialize a single-process distributed group if needed."""
    if dist.is_initialized():
        return dist.distributed_c10d._get_default_group()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29515")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return dist.distributed_c10d._get_default_group()


def cleanup_distributed():
    """Tear down the distributed group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


class SimpleParallelContext:
    """Simple parallel context for single-process runs."""

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
    """Minimal dataset configuration matching ClimLlamaDatasetsArgs."""

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
    leadtimes: tuple[int, ...] = (0, 1, 3, 6, 12, 24, 48, 72, 120, 168, 336, 720)
    codebook_size: int = 32768


def _default_data_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "combined_interleave_256")


def parse_args():
    parser = argparse.ArgumentParser(description="Parse weaved tokens and dump ClimLlama samples to JSON.")
    parser.add_argument(
        "--dataset-path",
        default=_default_data_path(),
        help="Directory containing the dataset shards (e.g., data/combined_interleave_256).",
    )
    parser.add_argument(
        "--prefix",
        default="1",
        help="Prefix of the dataset shard to read (e.g., '1' for files like 1.bin/1.idx).",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=1,
        help="Number of documents to parse and dump.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Sequence length passed to the dataset (kept for parity with training setup).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document shuffling.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "visualization"),
        help="Directory to place dumped JSON files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_prefix = os.path.join(args.dataset_path, args.prefix)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data_file = f"{data_prefix}.bin"
    if not os.path.exists(data_file):
        print(f"Data file not found at {data_file}")
        sys.exit(1)

    from nanotron.data.nemo_dataset import get_indexed_dataset
    from nanotron.data.climllama_dataset import ClimLlamaDataset

    world_pg = init_distributed()

    try:
        indexed_dataset = get_indexed_dataset(data_prefix, skip_warmup=False)
        total_num_of_documents = len(indexed_dataset.doc_idx) - 1

        print(f"Loaded dataset at prefix {data_prefix}")
        print(f"Documents: {total_num_of_documents}")

        num_to_dump = min(args.num_documents, total_num_of_documents)
        if num_to_dump < args.num_documents:
            print(f"Requested {args.num_documents} documents but only {total_num_of_documents} available, using {num_to_dump}.")

        documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)

        dataset = ClimLlamaDataset(
            cfg=MockDatasetConfig(),  # type: ignore
            tokenizer=MagicMock(),
            name="parse",
            data_prefix=data_prefix,
            documents=documents,
            indexed_dataset=indexed_dataset,
            num_samples=num_to_dump,
            seed=args.seed,
            parallel_context=SimpleParallelContext(world_pg),  # type: ignore
            seq_length=args.seq_length,
            drop_last=True,
        )
        print(f"Dataset ready with {len(dataset)} samples to parse.")

        for i in range(num_to_dump):
            item = dataset[i]
            doc_idx = dataset.shuffle_idx[i]
            filename = os.path.join(output_dir, f"{args.prefix}_doc{doc_idx}_sample{i}.json")
            dump_sample(
                item,
                parser=dataset.parser,
                var_level_names=dataset.var_level_names,
                cfg=dataset.cfg,
                filename=filename,
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
