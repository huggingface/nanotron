"""
Simple config consistency checker for ClimLlama runs.

Usage:
    python climllama/check_config.py path/to/config.yaml

Checks for:
- data_stages[*].sequence_length matching tokens.sequence_length
- model.model_config.max_position_embeddings (if set) matching tokens.sequence_length
- tokens.sequence_length divisible by context_parallel_size * tp
- flash_attention_2 is not used with context_parallel_size > 1
- ring_attn_heads_k_stride is set when using llama3_ring_attention
"""

import argparse
import sys
from typing import List, Tuple

from nanotron.config.config import Config, get_config_from_file


def check_sequence_lengths(config: Config) -> List[str]:
    errors: List[str] = []

    tokens_seq_len = config.tokens.sequence_length
    cp_size = getattr(config.parallelism, "context_parallel_size", 1)
    tp_size = getattr(config.parallelism, "tp", 1)
    cp_tp = (cp_size or 1) * (tp_size or 1)

    # data_stages stage sequence_length consistency
    for stage in config.data_stages or []:
        if stage.sequence_length != tokens_seq_len:
            errors.append(
                f"Stage '{stage.name}' (start_step={stage.start_training_step}) has sequence_length={stage.sequence_length} "
                f"but tokens.sequence_length={tokens_seq_len}"
            )

    # sequence_length must be divisible by context parallel size * tensor parallel size
    if cp_tp not in (None, 0, 1) and tokens_seq_len % cp_tp != 0:
        errors.append(
            f"tokens.sequence_length={tokens_seq_len} must be divisible by context_parallel_size={cp_size} * tp={tp_size} "
            f"(product={cp_tp})"
        )

    # model max_position_embeddings consistency (optional but helpful)
    model_max_pos = getattr(config.model.model_config, "max_position_embeddings", None)
    if model_max_pos is not None and model_max_pos != tokens_seq_len:
        errors.append(
            f"Model max_position_embeddings={model_max_pos} does not match tokens.sequence_length={tokens_seq_len}"
        )

    return errors


def check_flash_attention_with_cp(config: Config) -> List[str]:
    errors: List[str] = []

    cp_size = getattr(config.parallelism, "context_parallel_size", 1)
    attn_impl = getattr(config.model.model_config, "_attn_implementation", None)

    if cp_size > 1 and attn_impl == "flash_attention_2":
        errors.append(
            f"flash_attention_2 is not supported with context_parallel_size > 1 (got cp={cp_size}). "
            f"Use _attn_implementation='llama3_ring_attention' instead."
        )

    # Check ring_attn_heads_k_stride is set for llama3_ring_attention
    if attn_impl == "llama3_ring_attention":
        heads_k_stride = getattr(config.model.model_config, "ring_attn_heads_k_stride", None)
        if heads_k_stride is None:
            errors.append(
                "ring_attn_heads_k_stride must be specified when using llama3_ring_attention. "
                "Set ring_attn_heads_k_stride to a positive integer (e.g., 1)."
            )

    return errors


def check_heads_divisible_by_tp(config: Config) -> List[str]:
    errors: List[str] = []

    num_heads = getattr(config.model.model_config, "num_attention_heads", None)
    tp_size = getattr(config.parallelism, "tp", None)

    if num_heads is not None and tp_size not in (None, 0) and num_heads % tp_size != 0:
        errors.append(f"num_attention_heads={num_heads} must be divisible by tp={tp_size}")

    return errors


def check_log_attn_probs(config: Config, allow_log_attn_probs: bool) -> List[str]:
    errors: List[str] = []

    log_attn_probs = getattr(config.model.model_config, "log_attn_probs", False)
    if log_attn_probs and not allow_log_attn_probs:
        errors.append(
            "log_attn_probs=true is not allowed by default because it materializes full attention matrices "
            "and can cause OOMs; pass --allow-log-attn-probs to acknowledge the overhead."
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ClimLlama config for sequence length consistency.")
    parser.add_argument("config_file", help="Path to YAML or python config file.")
    parser.add_argument(
        "--allow-log-attn-probs",
        action="store_true",
        help="Permit configs with log_attn_probs=true (otherwise raises to avoid memory blow-ups).",
    )
    args = parser.parse_args()

    config = get_config_from_file(args.config_file)

    errors = check_sequence_lengths(config)
    errors.extend(check_flash_attention_with_cp(config))
    errors.extend(check_heads_divisible_by_tp(config))
    errors.extend(check_log_attn_probs(config, allow_log_attn_probs=args.allow_log_attn_probs))

    if errors:
        print("Config consistency check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"Config consistency check passed: sequence_length={config.tokens.sequence_length}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
