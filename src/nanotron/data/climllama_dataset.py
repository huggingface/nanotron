"""ClimLlama dataset with on-the-fly position generation.

This module implements the ClimLlamaDataset which returns whole documents as samples
and generates positional metadata on-the-fly during training using the
WeavedTokensPositionVisitor from atmtokenizer.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from nanotron import logging
from nanotron.config.config import ClimLlamaDatasetsArgs
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext

from nanotron.models.climllama import CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES

from .nemo_dataset.blendable_dataset import BlendableDataset
from .nemo_dataset.dataset_utils import get_datasets_weights_and_num_samples
from .nemo_dataset.indexed_dataset import MMapIndexedDataset

logger = logging.get_logger(__name__)


@dataclass
class ClimLlamaSubsetSplitLog:
    """Log information for a ClimLlamaDataset subset."""

    name: str
    data_prefix: str
    num_documents: int
    num_samples: int
    seq_length: int


class ClimLlamaDataset(Dataset):
    """Dataset for ClimLlama that returns whole documents with position arrays.

    This dataset returns entire documents as samples (not fixed-length chunks).
    Each sample contains:
    - var_idx: Variable index (z, t, q, u, v, etc.)
    - res_idx: Resolution level index
    - leadtime_idx: Lead time index in hours
    - spatial_temporal_features: CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES-D features [x, y, z, cos_hour, sin_hour, cos_day, sin_day, log10_level_hPa]

    The positional metadata is generated on-the-fly using the WeavedTokensPositionVisitor
    from atmtokenizer, which parses the token sequence using a Lark grammar and extracts
    positional information from the parse tree.
    """

    def __init__(
        self,
        cfg: ClimLlamaDatasetsArgs,
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
    ):
        super().__init__()
        self.name = name
        self.folder_path = data_prefix  # For BlendableDataset compatibility
        self.indexed_dataset = indexed_dataset
        self.seq_length = seq_length
        self.documents = documents
        self.num_samples = num_samples
        self.cfg = cfg

        # Build shuffled document indices
        np_rng = np.random.RandomState(seed=seed)
        self.shuffle_idx = self._build_document_shuffle_idx(documents, num_samples, np_rng)

        # Lazy import atmtokenizer modules
        self._parser = None
        self._special_tokens = None

        # Load metadata (resolution shapes and variable names)
        metadata = self._load_metadata(data_prefix)
        self.resolution_shapes = metadata["resolution_shapes"]
        self.var_level_names = metadata["variable_names"]

        # Build var_level to var_idx mapping
        self.var_level_to_var_idx = self._build_var_level_mapping()

        # Build leadtime (hours) to leadtime_idx mapping
        self.leadtime_to_idx = self._build_leadtime_mapping()

        # Load global timestamp array
        self.timestamps = self._load_timestamps(data_prefix)

        # Create subset log for BlendableDataset compatibility
        self.subset_log = ClimLlamaSubsetSplitLog(
            name=name,
            data_prefix=data_prefix,
            num_documents=len(documents),
            num_samples=num_samples,
            seq_length=seq_length,
        )

    def _build_document_shuffle_idx(
        self, documents: np.ndarray, num_samples: int, np_rng: np.random.RandomState
    ) -> np.ndarray:
        """Build shuffled index mapping for whole documents.

        Args:
            documents: Array of document indices to use
            num_samples: Number of samples to generate
            np_rng: Random state for shuffling

        Returns:
            Shuffled array of document indices
        """
        num_docs = len(documents)
        # Calculate how many epochs we need
        num_epochs = (num_samples + num_docs - 1) // num_docs

        # Build document index with multiple epochs if needed
        doc_indices = []
        for _ in range(num_epochs):
            epoch_docs = documents.copy()
            np_rng.shuffle(epoch_docs)
            doc_indices.append(epoch_docs)

        # Concatenate and truncate to num_samples
        all_docs = np.concatenate(doc_indices)[:num_samples]
        return all_docs

    def __len__(self) -> int:
        return len(self.shuffle_idx)

    def _load_metadata(self, data_prefix: str) -> Dict[str, Any]:
        """Load resolution shapes and variable names from dataset metadata.

        Searches for metadata in the following locations (in order):
        1. {data_path}/metadata.json
        2. {data_path.parent}/metadata.json
        3. {data_prefix}.json (e.g., data/combined_interleave_256/1.json)

        Supports two metadata formats:
        1. model_config.resolutions: list of integers [1, 2, 3, ...]
           -> converted to {0: (1, 2), 1: (2, 4), ...} assuming width = 2 * height
        2. weaver_config.resolutions: list of strings ["1×2", "2×4", ...]
           -> parsed directly to {0: (1, 2), 1: (2, 4), ...}

        Returns:
            Dict with:
            - "resolution_shapes": Dict mapping resolution ID to (height, width) tuple
              E.g., {0: (1, 2), 1: (2, 4), 2: (3, 6), ...}
            - "variable_names": List of variable names from metadata (e.g., ["z_500", "t_750", ...])
        """
        data_path = Path(data_prefix)
        if data_path.is_file():
            data_path = data_path.parent

        # Try different locations for metadata.json
        # Also check {data_prefix}.json (e.g., data/combined_interleave_256/1.json)
        metadata_paths = [
            data_path / "metadata.json",
            data_path.parent / "metadata.json",
            Path(f"{data_prefix}.json"),
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
            raise ValueError(
                f"Metadata file not found for data_prefix '{data_prefix}'. "
                f"Searched locations: {[str(p) for p in metadata_paths]}"
            )

        # Check codebook_size consistency
        metadata_codebook_size = metadata.get("codebook_size") or metadata.get("weaver_config", {}).get(
            "codebook_size"
        )
        if metadata_codebook_size is not None and metadata_codebook_size != self.cfg.codebook_size:
            raise ValueError(
                f"Codebook size mismatch for data_prefix '{data_prefix}': "
                f"metadata has codebook_size={metadata_codebook_size}, "
                f"but config has codebook_size={self.cfg.codebook_size}. "
                "Please ensure the dataset was created with the same codebook size as the model config."
            )

        # Extract variable names from metadata (required)
        var_level_names = metadata.get("variable_names", [])
        if not var_level_names:
            raise ValueError(
                f"variable_names not found in metadata for data_prefix '{data_prefix}'. "
                "Metadata must contain a 'variable_names' field with the list of variable+level names."
            )
        log_rank(
            f"Loaded {len(var_level_names)} variable names from metadata",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # Try model_config.resolutions (list of integers)
        resolution_values = metadata.get("model_config", {}).get("resolutions", [])
        if resolution_values:
            # Map resolution ID to (height, width) tuple
            # Grid dimensions: height = n, width = 2n (for equirectangular projection)
            return {
                "resolution_shapes": {i: (n, 2 * n) for i, n in enumerate(resolution_values)},
                "variable_names": var_level_names,
            }

        # Try weaver_config.resolutions (list of strings like "1×2", "2×4")
        weaver_resolutions = metadata.get("weaver_config", {}).get("resolutions", [])
        if weaver_resolutions:
            resolution_shapes = {}
            for i, res_str in enumerate(weaver_resolutions):
                # Parse "height×width" format (× is Unicode multiplication sign)
                # Also handle "heightxwidth" with lowercase x
                if "×" in res_str:
                    parts = res_str.split("×")
                elif "x" in res_str:
                    parts = res_str.split("x")
                else:
                    continue
                if len(parts) == 2:
                    height, width = int(parts[0]), int(parts[1])
                    resolution_shapes[i] = (height, width)
            if resolution_shapes:
                return {
                    "resolution_shapes": resolution_shapes,
                    "variable_names": var_level_names,
                }

        raise ValueError(
            f"No resolutions found in metadata for data_prefix '{data_prefix}'. "
            "Metadata must contain either 'model_config.resolutions' or 'weaver_config.resolutions'."
        )

    def _build_var_level_mapping(self) -> np.ndarray:
        """Build mapping from var_level index to var index.

        The metadata contains variable names like "z_500", "t_750", "msl", etc.
        The dataset config contains base variable names like "unk", "z", "t", "q", etc.
        This method creates a mapping from the var_level index (used in the parser output)
        to the var index (used in the model's position embeddings).

        Returns:
            numpy array where arr[var_level_idx] = var_idx
        """
        # Get the variable names from dataset config (ClimLlamaDatasetsArgs)
        model_variables = getattr(self.cfg, "variables", None)
        assert model_variables is not None, (
            "cfg.variables not found. The dataset config must contain a 'variables' field "
            "with the list of base variable names (e.g., 'unk', 'z', 't', 'q', ...)."
        )

        # Build a mapping from base variable name to its index in model_variables
        var_name_to_idx = {var: idx for idx, var in enumerate(model_variables)}

        # Create the mapping array
        mapping = np.zeros(len(self.var_level_names), dtype=np.int64)

        for var_level_idx, var_level_name in enumerate(self.var_level_names):
            # Extract base variable name from var_level_name (e.g., "z_500" -> "z", "msl" -> "msl")
            # Split on underscore, but handle cases like "tp" or "tp_6h"
            parts = var_level_name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                # Pressure level variable like "z_500", "t_750"
                base_var = parts[0]
            else:
                # Surface variable like "msl", "t2m", "tp", "tp_6h"
                base_var = var_level_name

            # Look up the index in model_variables
            if base_var in var_name_to_idx:
                mapping[var_level_idx] = var_name_to_idx[base_var]
            else:
                # Unknown variable, map to index 0 ("unk")
                mapping[var_level_idx] = 0
                log_rank(
                    f"Warning: Variable '{base_var}' (from '{var_level_name}') not found in model variables, "
                    f"mapping to index 0 ('{model_variables[0]}')",
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )

        return mapping

    def _get_log10_level_hPa(self, var_level_name: str) -> float:
        """Calculate log10(10*level) for a variable name.

        For pressure level variables like "z_500", extracts level (500) and returns log10(10*500).
        For surface variables like "msl" or "t2m", returns 5.0 (representing surface).

        Args:
            var_level_name: Variable name with optional level suffix (e.g., "z_500", "msl")

        Returns:
            log10(10*level) for pressure level variables, or 5.0 for surface variables
        """
        parts = var_level_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            # Pressure level variable like "z_500", "t_750"
            level = int(parts[1])
            return np.log10(10.0 * level)
        else:
            # Surface variable like "msl", "t2m", "tp"
            return 5.0

    def _build_leadtime_mapping(self) -> Dict[int, int]:
        """Build mapping from leadtime hours to leadtime index.

        The parser returns leadtime in hours (e.g., 0, 6, 12, 24, ...).
        The model expects a leadtime index into cfg.leadtimes for position embeddings.
        This method creates a mapping from leadtime hours to the corresponding index.

        Returns:
            Dict mapping leadtime hours to leadtime index
        """
        leadtimes = getattr(self.cfg, "leadtimes", None)
        assert leadtimes is not None, (
            "cfg.leadtimes not found. The dataset config must contain a 'leadtimes' field "
            "with the list of leadtime hours (e.g., (0, 1, 3, 6, 12, 24, 48, 72, 120, 168, 336, 720))."
        )

        # Build mapping: leadtime_hours -> index
        mapping = {lt: idx for idx, lt in enumerate(leadtimes)}
        log_rank(
            f"Built leadtime mapping for {len(leadtimes)} leadtimes: {leadtimes}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        return mapping

    def _load_timestamps(self, data_prefix: str) -> Optional[np.ndarray]:
        """Load global timestamp array from file.

        The timestamp file is expected at {data_prefix}.npy (e.g., data/combined_interleave_256/1.npy)
        and contains one timestamp (datetime64[s]) per document in the indexed dataset.

        Args:
            data_prefix: Path prefix for the dataset files

        Returns:
            numpy array of timestamps, or None if file not found
        """
        timestamp_path = f"{data_prefix}.npy"
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
        """Lazy initialization of Lark parser with custom TokenIDLexer."""
        if self._parser is None:
            import atmtokenizer
            from atmtokenizer.eval.special_tokens import create_parser

            # Create parser with custom TokenIDLexer for token classification
            self._parser = create_parser(atmtokenizer.GRAMMAR_PATH, self.special_tokens)
        return self._parser

    # TODO: Create special tokens based on the metadata
    @property
    def special_tokens(self):
        """Lazy initialization of special tokens."""
        if self._special_tokens is None:
            from atmtokenizer.eval.special_tokens import create_special_tokens

            self._special_tokens = create_special_tokens(codebook_size=self.cfg.codebook_size)
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
            # Parse the token sequence using the custom lexer
            # The parser's custom lexer accepts an array of integer token IDs
            tree = self.parser.parse(input_ids)

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
            spatial_temporal_features = np.zeros((n_tokens, CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES), dtype=np.float32)

            # Fill arrays from parsed positions
            for i, (token, pos) in enumerate(leaves):
                if i >= n_tokens:
                    break

                # Get var_level index from parser and remap to var index
                var_level_idx = pos.get("variable", 0) or 0
                if var_level_idx < len(self.var_level_to_var_idx):
                    var_idx[i] = self.var_level_to_var_idx[var_level_idx]
                else:
                    var_idx[i] = 0  # Unknown variable, map to index 0 ("unk")

                res_idx[i] = pos.get("resolution", 0) or 0

                # Get leadtime in hours from parser and remap to leadtime index
                leadtime_hours = pos.get("leadtime", 0) or 0
                leadtime_idx[i] = self.leadtime_to_idx.get(leadtime_hours, 0)

                # Spatial features (x, y, z) - unit sphere coordinates
                spatial_temporal_features[i, 0] = pos.get("grid_x", 0.0) or 0.0
                spatial_temporal_features[i, 1] = pos.get("grid_y", 0.0) or 0.0
                spatial_temporal_features[i, 2] = pos.get("grid_z", 0.0) or 0.0

                # Temporal features (cos_hour, sin_hour, cos_day, sin_day)
                spatial_temporal_features[i, 3] = pos.get("cos_hour_of_day", 0.0)
                spatial_temporal_features[i, 4] = pos.get("sin_hour_of_day", 0.0)
                spatial_temporal_features[i, 5] = pos.get("cos_day_of_year", 0.0)
                spatial_temporal_features[i, 6] = pos.get("sin_day_of_year", 0.0)

                # Pressure level feature: log10(10*level) in hPa
                if var_level_idx < len(self.var_level_names):
                    var_level_name = self.var_level_names[var_level_idx]
                    spatial_temporal_features[i, 7] = self._get_log10_level_hPa(var_level_name)
                else:
                    spatial_temporal_features[i, 7] = 5.0  # Default to surface level

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
            spatial_temporal_features = np.zeros((n_tokens, CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES), dtype=np.float32)

        return {
            "var_idx": var_idx,
            "res_idx": res_idx,
            "leadtime_idx": leadtime_idx,
            "spatial_temporal_features": spatial_temporal_features,
        }

    def _get_document_tokens(self, doc_idx: int) -> np.ndarray:
        """Get all tokens for a document by concatenating its sequences.

        The indexed dataset stores multiple sequences per document. The doc_idx
        array maps document indices to sequence index ranges.

        Args:
            doc_idx: Document index

        Returns:
            Concatenated token array for the entire document
        """
        doc_indices = self.indexed_dataset.doc_idx
        doc_start = doc_indices[doc_idx]
        doc_end = doc_indices[doc_idx + 1]

        # Collect all tokens for this document
        tokens = []
        for seq_idx in range(doc_start, doc_end):
            seq = self.indexed_dataset.get(seq_idx)
            tokens.extend(seq.tolist())

        return np.array(tokens, dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return a whole document with position information generated on-the-fly.

        Args:
            idx: Sample index

        Returns:
            Dict with input_ids, positions, var_idx, res_idx, leadtime_idx,
            and spatial_temporal_features
        """
        # Get document index from shuffled mapping
        doc_idx = self.shuffle_idx[idx]

        # Get the entire document from indexed dataset
        input_ids = self._get_document_tokens(doc_idx)

        # Get timestamp for this document
        timestamp_0 = self._get_document_timestamp(doc_idx)

        # Generate positions on-the-fly
        positions = self._generate_positions(input_ids, timestamp_0)

        # Create position_ids (sequential within the document)
        position_ids = np.arange(len(input_ids), dtype=np.int64)

        return {
            "input_ids": input_ids,
            "positions": position_ids,
            "var_idx": positions["var_idx"],
            "res_idx": positions["res_idx"],
            "leadtime_idx": positions["leadtime_idx"],
            "spatial_temporal_features": positions["spatial_temporal_features"],
        }


def _build_single_climllama_dataset(
    cfg: ClimLlamaDatasetsArgs,
    tokenizer: PreTrainedTokenizerBase,
    data_prefix: str,
    num_samples: int,
    seq_length: int,
    seed: int,
    parallel_context: ParallelContext,
    name: str = "train",
    drop_last: bool = True,
) -> ClimLlamaDataset:
    """Build a single ClimLlamaDataset from the given parameters.

    This is an internal function for creating individual ClimLlamaDataset instances.

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

    Returns:
        ClimLlamaDataset instance
    """
    from .nemo_dataset import get_indexed_dataset

    # Get indexed dataset
    indexed_dataset = get_indexed_dataset(data_prefix, skip_warmup=False)
    # doc_idx has length (num_documents + 1), so num_documents = len(doc_idx) - 1
    total_num_of_documents = len(indexed_dataset.doc_idx) - 1

    log_rank(
        f"Building ClimLlamaDataset '{name}' with {total_num_of_documents} documents from {data_prefix}",
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
    )

    return dataset


def build_climllama_dataset(
    cfg: ClimLlamaDatasetsArgs,
    tokenizer: PreTrainedTokenizerBase,
    data_prefix: Union[str, List[str]],
    num_samples: int,
    seq_length: int,
    seed: int,
    parallel_context: ParallelContext,
    name: str = "train",
    drop_last: bool = True,
) -> Union[ClimLlamaDataset, BlendableDataset]:
    """Build a ClimLlamaDataset or BlendableDataset from the given parameters.

    This function supports both single and multiple data prefixes. When multiple
    prefixes are provided, it creates a BlendableDataset that blends samples from
    multiple ClimLlamaDataset instances.

    Args:
        cfg: Dataset configuration
        tokenizer: Tokenizer
        data_prefix: Path prefix for indexed dataset files. Can be:
            - A single string path (e.g., "data/train")
            - A list with a single path (e.g., ["data/train"])
            - A list of paths with equal weights (e.g., ["data/train1", "data/train2"])
            - A blended format list (e.g., [0.7, "data/train1", 0.3, "data/train2"])
        num_samples: Number of samples to generate
        seq_length: Sequence length
        seed: Random seed
        parallel_context: Parallel context for distributed training
        name: Dataset name ("train", "valid", "test")
        drop_last: Whether to drop the last incomplete batch

    Returns:
        ClimLlamaDataset instance for single prefix, or BlendableDataset for multiple prefixes
    """
    # Handle single string prefix
    if isinstance(data_prefix, str):
        return _build_single_climllama_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            data_prefix=data_prefix,
            num_samples=num_samples,
            seq_length=seq_length,
            seed=seed,
            parallel_context=parallel_context,
            name=name,
            drop_last=drop_last,
        )

    # Handle list with single prefix
    if len(data_prefix) == 1:
        return _build_single_climllama_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            data_prefix=data_prefix[0],
            num_samples=num_samples,
            seq_length=seq_length,
            seed=seed,
            parallel_context=parallel_context,
            name=name,
            drop_last=drop_last,
        )

    # Handle multiple prefixes - check if blended format [weight1, path1, weight2, path2, ...]
    is_blended = len(data_prefix) % 2 == 0
    if is_blended:
        try:
            float(data_prefix[0])  # Check if first element can be converted to float (weight)
        except (ValueError, TypeError):
            is_blended = False

    if not is_blended:
        # Not blended format - treat as multiple paths with equal weights
        # Convert to blended format: [1.0, path1, 1.0, path2, ...]
        log_rank(
            f"data_prefix has {len(data_prefix)} paths without weights. "
            f"Treating as {len(data_prefix)} datasets with equal weights.",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        blended_data_prefix = []
        for path in data_prefix:
            blended_data_prefix.extend([1.0, path])
        data_prefix = blended_data_prefix

    # Parse weights and prefixes
    output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
    prefixes, weights, datasets_num_samples = output

    log_rank(
        f"Building BlendableDataset with {len(prefixes)} ClimLlamaDatasets",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    # Build individual datasets
    datasets = []
    for i, (prefix, ds_num_samples) in enumerate(zip(prefixes, datasets_num_samples)):
        log_rank(
            f"  Dataset {i}: {prefix} with weight {weights[i]:.4f}, {ds_num_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        dataset = _build_single_climllama_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            data_prefix=prefix,
            num_samples=ds_num_samples,
            seq_length=seq_length,
            seed=seed,
            parallel_context=parallel_context,
            name=name,
            drop_last=drop_last,
        )
        datasets.append(dataset)

    # Create and return BlendableDataset
    return BlendableDataset(
        datasets=datasets,
        weights=weights,
        size=num_samples,
        parallel_context=parallel_context,
        seed=seed,
    )
