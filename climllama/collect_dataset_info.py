"""Collect aggregate dataset statistics for Megatron/NeMo indexed datasets.

This script mirrors the data_prefix parsing behavior of
climllama.prepare_training_config: it accepts single paths, wildcard patterns,
or weighted, comma-separated blends. It loads the companion ``.json`` metadata
for each dataset prefix, reads ``statistics.num_documents``/``statistics.num_samples``
and ``statistics.total_tokens``, and prints per-dataset and total counts.
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from nanotron.logging import human_format


def expand_wildcard_paths(path_pattern: str) -> List[str]:
    """Expand wildcard pattern to list of matching paths without extensions."""
    # If a directory is provided, treat it as "dir/*"
    if Path(path_pattern).is_dir():
        path_pattern = os.path.join(path_pattern, "*")

    # If the pattern already targets .bin/.idx files, match and strip extension
    if path_pattern.endswith((".bin", ".idx")):
        matched = glob.glob(path_pattern)
        return sorted({os.path.splitext(p)[0] for p in matched})

    # Directory-style wildcard: "/path/*" -> look for "/path/*.bin"
    if path_pattern.endswith("*"):
        matched = glob.glob(os.path.join(path_pattern.rstrip("*"), "*.bin"))
        if matched:
            return sorted({os.path.splitext(p)[0] for p in matched})

    # Generic wildcard without extension: append .bin
    if "*" in path_pattern:
        matched = glob.glob(f"{path_pattern}.bin")
        if matched:
            return sorted({os.path.splitext(p)[0] for p in matched})

    # No wildcard: return as-is
    return [path_pattern]


def parse_data_prefix(data_prefix_str: str) -> List:
    """Simplified parser mirroring prepare_training_config behavior."""
    if "," in data_prefix_str:
        parts = [p.strip() for p in data_prefix_str.split(",")]
        parsed: List = []
        for part in parts:
            try:
                parsed.append(float(part))  # weight
            except ValueError:
                parsed.extend(expand_wildcard_paths(part))
        return parsed

    return expand_wildcard_paths(data_prefix_str)


def extract_dataset_paths(parsed_prefixes: List) -> List[str]:
    """Filter parsed data_prefix entries down to unique dataset paths."""
    seen = set()
    paths: List[str] = []
    for item in parsed_prefixes:
        if isinstance(item, str) and item not in seen:
            paths.append(item)
            seen.add(item)
    return paths


def read_stats(json_path: Path) -> Tuple[int, int]:
    with json_path.open("r") as f:
        payload = json.load(f)
    stats: Dict = payload.get("statistics") or {}
    num_documents = stats.get("num_documents", stats.get("num_samples"))
    total_tokens = stats.get("total_tokens")
    if num_documents is None or total_tokens is None:
        raise ValueError(f"Missing statistics fields in {json_path}")
    return int(num_documents), int(total_tokens)


def collect_dataset_info(data_prefix: str) -> None:
    parsed_prefixes = parse_data_prefix(data_prefix)
    dataset_paths = extract_dataset_paths(parsed_prefixes)

    if not dataset_paths:
        raise ValueError(f"No dataset paths found for data_prefix='{data_prefix}'")

    total_documents = 0
    total_tokens = 0
    missing_json = []
    per_dataset: List[Tuple[str, int, int]] = []

    for prefix in dataset_paths:
        json_path = Path(f"{prefix}.json")
        if not json_path.exists():
            missing_json.append(str(json_path))
            continue

        try:
            num_documents, tokens = read_stats(json_path)
        except Exception as exc:  # noqa: BLE001
            missing_json.append(f"{json_path} ({exc})")
            continue

        per_dataset.append((str(json_path), num_documents, tokens))
        total_documents += num_documents
        total_tokens += tokens

    print(f"Parsed data_prefix -> {len(dataset_paths)} dataset(s) (ignoring weights).")
    for path, documents, tokens in per_dataset:
        print(
            f"- {path}: documents={documents:,}, tokens={tokens:,} "
            f"({human_format(tokens)})"
        )

    print("\nTotals:")
    print(f"- Documents: {total_documents:,}")
    print(f"- Tokens:  {total_tokens:,} ({human_format(total_tokens)})")

    if missing_json:
        print("\nSkipped entries without usable metadata:")
        for msg in missing_json:
            print(f"- {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect statistics from dataset metadata JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        required=True,
        help=(
            "Path prefix to Megatron indexed dataset (.bin/.idx). Supports wildcards "
            "(e.g., '/path/data_*' or '/path/*') and weighted blends "
            "(e.g., '0.6,/path1,0.4,/path2')."
        ),
    )
    args = parser.parse_args()
    collect_dataset_info(args.data_prefix)


if __name__ == "__main__":
    main()
