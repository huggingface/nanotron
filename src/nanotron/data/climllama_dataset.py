"""ClimLlama dataset with on-the-fly position generation.

This module implements the ClimLlamaDataset which extends GPTDataset to generate
positional metadata on-the-fly during training using the WeavedTokensPositionVisitor
from atmtokenizer.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext

from .nemo_dataset import GPTDataset
from .nemo_dataset.indexed_dataset import MMapIndexedDataset

logger = logging.get_logger(__name__)


class ClimLlamaDataset(GPTDataset):
    """Dataset for ClimLlama that generates position arrays on-the-fly.

    This dataset extends GPTDataset with climate-specific positional metadata:
    - var_idx: Variable index (z, t, q, u, v, etc.)
    - res_idx: Resolution level index
    - leadtime_idx: Lead time index in hours
    - spatial_temporal_features: 7D features [x, y, z, cos_hour, sin_hour, cos_day, sin_day]

    The positional metadata is generated on-the-fly using the WeavedTokensPositionVisitor
    from atmtokenizer, which parses the token sequence using a Lark grammar and extracts
    positional information from the parse tree.
    """

    def __init__(
        self,
        cfg: Any,
        tokenizer: PreTrainedTokenizerBase,
        name: str,
        data_prefix: str,
        documents: np.ndarray,
        indexed_dataset: MMapIndexedDataset,
        num_samples: int,
        seq_length: int,
        seed: int,
        parallel_context: ParallelContext,
        drop_last: bool = True,
        codebook_size: int = 32768,
    ):
        super().__init__(
            cfg,
            tokenizer,
            name,
            data_prefix,
            documents,
            indexed_dataset,
            num_samples,
            seq_length,
            seed,
            parallel_context,
            drop_last,
        )

        self.codebook_size = codebook_size

        # Lazy import atmtokenizer modules
        self._parser = None
        self._special_tokens = None

        # Load resolution shapes from metadata
        self.resolution_shapes = self._load_resolution_shapes_from_metadata(data_prefix)

        # Load global timestamp array
        self.timestamps = self._load_timestamps(data_prefix)

    def _load_resolution_shapes_from_metadata(self, data_prefix: str) -> Dict[int, Tuple[int, int]]:
        """Load resolution shapes from dataset metadata.json.

        The dataset folder contains metadata.json with structure:
        {
            "model_config": {
                "resolutions": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32],
                ...
            }
        }

        Returns:
            Dict mapping resolution ID to (height, width) tuple
            E.g., {0: (1, 2), 1: (2, 4), 2: (3, 6), ...}
        """
        data_path = Path(data_prefix)
        if data_path.is_file():
            data_path = data_path.parent

        # Try different locations for metadata.json
        metadata_paths = [
            data_path / "metadata.json",
            data_path.parent / "metadata.json",
        ]

        metadata = None
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                log_rank(
                    f"Loaded metadata from {metadata_path}",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
                break

        if metadata is None:
            log_rank(
                "Warning: metadata.json not found, using default resolutions",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            default_res = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
            return {i: (n, 2 * n) for i, n in enumerate(default_res)}

        resolution_values = metadata.get("model_config", {}).get("resolutions", [])
        if not resolution_values:
            log_rank(
                "Warning: No resolutions found in metadata, using defaults",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            default_res = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
            return {i: (n, 2 * n) for i, n in enumerate(default_res)}

        # Map resolution ID to (height, width) tuple
        # Grid dimensions: height = n, width = 2n (for equirectangular projection)
        return {i: (n, 2 * n) for i, n in enumerate(resolution_values)}

    def _load_timestamps(self, data_prefix: str) -> Optional[np.ndarray]:
        """Load global timestamp array from file.

        The timestamp file is expected at {data_prefix}_timestamps.npy and contains
        one timestamp (datetime64[s]) per document in the indexed dataset.

        Args:
            data_prefix: Path prefix for the dataset files

        Returns:
            numpy array of timestamps, or None if file not found
        """
        timestamp_path = f"{data_prefix}_timestamps.npy"
        if os.path.exists(timestamp_path):
            timestamps = np.load(timestamp_path)
            log_rank(
                f"Loaded {len(timestamps)} timestamps from {timestamp_path}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            return timestamps
        else:
            log_rank(
                f"Warning: No timestamp file found at {timestamp_path}. "
                "Temporal features will use fallback epoch time.",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            return None

    @property
    def parser(self):
        """Lazy initialization of Lark parser."""
        if self._parser is None:
            import atmtokenizer
            from lark import Lark

            # Load grammar from atmtokenizer's grammar file
            with open(atmtokenizer.GRAMMAR_PATH, "r") as f:
                grammar = f.read()
            self._parser = Lark(grammar, parser="lalr")
        return self._parser

    @property
    def special_tokens(self):
        """Lazy initialization of special tokens."""
        if self._special_tokens is None:
            from atmtokenizer.eval.special_tokens import create_special_tokens

            self._special_tokens = create_special_tokens(codebook_size=self.codebook_size)
        return self._special_tokens

    def _get_document_timestamp(self, doc_idx: int) -> datetime:
        """Get timestamp for a document.

        Args:
            doc_idx: Document index in the indexed dataset

        Returns:
            datetime object for the document's initial timestamp
        """
        if self.timestamps is not None and doc_idx < len(self.timestamps):
            ts = self.timestamps[doc_idx]
            if isinstance(ts, np.datetime64):
                # Convert numpy datetime64 to Python datetime
                # datetime64[s] -> Unix timestamp -> datetime
                unix_ts = ts.astype("datetime64[s]").astype("int64")
                return datetime.utcfromtimestamp(unix_ts)
            elif isinstance(ts, (int, float)):
                # Unix timestamp
                return datetime.utcfromtimestamp(ts)
            else:
                return ts
        else:
            # Fallback: return epoch
            return datetime(1970, 1, 1, 0, 0, 0)

    def _generate_positions(
        self, input_ids: np.ndarray, timestamp_0: datetime
    ) -> Dict[str, np.ndarray]:
        """Generate position arrays on-the-fly using WeavedTokensPositionVisitor.

        Args:
            input_ids: Token sequence from indexed dataset
            timestamp_0: Initial timestamp for the document

        Returns:
            Dict with var_idx, res_idx, leadtime_idx, spatial_temporal_features
        """
        from atmtokenizer.eval.weaved_tokens_position import get_leaf_positions

        n_tokens = len(input_ids)

        try:
            # Convert token IDs to string for parsing
            # The parser expects a string representation of the token sequence
            token_str = " ".join(str(t) for t in input_ids)
            tree = self.parser.parse(token_str)

            # Get positions for all tokens
            leaves = get_leaf_positions(
                tree,
                self.resolution_shapes,
                self.special_tokens,
                timestamp_0,
            )

            # Initialize arrays
            var_idx = np.zeros(n_tokens, dtype=np.int64)
            res_idx = np.zeros(n_tokens, dtype=np.int64)
            leadtime_idx = np.zeros(n_tokens, dtype=np.int64)
            spatial_temporal_features = np.zeros((n_tokens, 7), dtype=np.float32)

            # Fill arrays from parsed positions
            for i, (token, pos) in enumerate(leaves):
                if i >= n_tokens:
                    break

                var_idx[i] = pos.get("variable", 0) or 0
                res_idx[i] = pos.get("resolution", 0) or 0
                leadtime_idx[i] = pos.get("leadtime", 0) or 0

                # Spatial features (x, y, z) - unit sphere coordinates
                spatial_temporal_features[i, 0] = pos.get("grid_x", 0.0) or 0.0
                spatial_temporal_features[i, 1] = pos.get("grid_y", 0.0) or 0.0
                spatial_temporal_features[i, 2] = pos.get("grid_z", 0.0) or 0.0

                # Temporal features (cos_hour, sin_hour, cos_day, sin_day)
                spatial_temporal_features[i, 3] = pos.get("cos_hour_of_day", 0.0)
                spatial_temporal_features[i, 4] = pos.get("sin_hour_of_day", 0.0)
                spatial_temporal_features[i, 5] = pos.get("cos_day_of_year", 0.0)
                spatial_temporal_features[i, 6] = pos.get("sin_day_of_year", 0.0)

        except Exception as e:
            # If parsing fails, return default zero arrays
            log_rank(
                f"Warning: Failed to parse token sequence for positions: {e}",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            var_idx = np.zeros(n_tokens, dtype=np.int64)
            res_idx = np.zeros(n_tokens, dtype=np.int64)
            leadtime_idx = np.zeros(n_tokens, dtype=np.int64)
            spatial_temporal_features = np.zeros((n_tokens, 7), dtype=np.float32)

        return {
            "var_idx": var_idx,
            "res_idx": res_idx,
            "leadtime_idx": leadtime_idx,
            "spatial_temporal_features": spatial_temporal_features,
        }

    def _get_document_index_for_sample(self, sample_idx: int) -> int:
        """Get the starting document index for a given sample.

        This method determines which document a sample starts in by looking
        at the sample_idx mapping created during dataset initialization.

        Args:
            sample_idx: The sample index (0 to len(dataset)-1)

        Returns:
            Document index in the original dataset
        """
        # Get the shuffled index
        shuffled_idx = self.shuffle_idx[sample_idx]
        # Get the document index from sample_idx mapping
        doc_index_f, _ = self.sample_idx[shuffled_idx]
        # Map through doc_idx to get the actual document index
        return self.doc_idx[doc_index_f]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return item with position information generated on-the-fly.

        Args:
            idx: Sample index

        Returns:
            Dict with input_ids, positions, var_idx, res_idx, leadtime_idx,
            and spatial_temporal_features
        """
        # Get base item from GPTDataset (contains input_ids)
        base_item = super().__getitem__(idx)
        input_ids = base_item["input_ids"]

        # Get document index and timestamp
        doc_idx = self._get_document_index_for_sample(idx)
        timestamp_0 = self._get_document_timestamp(doc_idx)

        # Generate positions on-the-fly
        positions = self._generate_positions(input_ids, timestamp_0)

        # Create position_ids for document masking (sequential within each document)
        # For simplicity, we use sequential positions here
        # More sophisticated document boundary tracking would require
        # tracking document boundaries within the sample
        position_ids = np.arange(len(input_ids), dtype=np.int64)

        return {
            "input_ids": input_ids,
            "positions": position_ids,
            "var_idx": positions["var_idx"],
            "res_idx": positions["res_idx"],
            "leadtime_idx": positions["leadtime_idx"],
            "spatial_temporal_features": positions["spatial_temporal_features"],
        }


def build_climllama_dataset(
    cfg: Any,
    tokenizer: PreTrainedTokenizerBase,
    data_prefix: str,
    num_samples: int,
    seq_length: int,
    seed: int,
    parallel_context: ParallelContext,
    name: str = "train",
    drop_last: bool = True,
    codebook_size: int = 32768,
) -> ClimLlamaDataset:
    """Build a ClimLlamaDataset from the given parameters.

    This is a convenience function for creating ClimLlamaDataset instances.

    Args:
        cfg: Dataset configuration
        tokenizer: Tokenizer (for FIM support)
        data_prefix: Path prefix for indexed dataset files
        num_samples: Number of samples to generate
        seq_length: Sequence length
        seed: Random seed
        parallel_context: Parallel context for distributed training
        name: Dataset name ("train", "valid", "test")
        drop_last: Whether to drop the last incomplete batch
        codebook_size: Size of the codebook for special token generation

    Returns:
        ClimLlamaDataset instance
    """
    from .nemo_dataset import get_indexed_dataset

    # Get indexed dataset
    indexed_dataset = get_indexed_dataset(data_prefix, skip_warmup=False)
    total_num_of_documents = indexed_dataset.sizes.shape[0]

    log_rank(
        f"Building ClimLlamaDataset '{name}' with {total_num_of_documents} documents",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    # Use all documents
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)

    dataset = ClimLlamaDataset(
        cfg=cfg,
        tokenizer=tokenizer,
        name=name,
        data_prefix=data_prefix,
        documents=documents,
        indexed_dataset=indexed_dataset,
        num_samples=num_samples,
        seq_length=seq_length,
        seed=seed,
        parallel_context=parallel_context,
        drop_last=drop_last,
        codebook_size=codebook_size,
    )

    return dataset
