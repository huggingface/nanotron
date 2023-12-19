"""

You can run using command:
```
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/run_evals.py --lighteval-config examples/thomwolf/config_thomwolf_llama_evals.py --s3-prefix s3://bucket/path --checkpoints-pattern.*/500.*
```
"""
import argparse

from nanotron.config import get_lighteval_config_from_file
from nanotron import logging
from nanotron.lighteval.runner import LightEvalRunner

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lighteval-config", type=str, required=True, help="Path to the YAML or python config file with new tasks"
    )
    parser.add_argument(
        "--s3-prefix", type=str, help="Optional s3 prefix to search for checkpoints in (e.g. s3://bucket/path)"
    )
    parser.add_argument(
        "--checkpoints-pattern",
        type=str,
        help="Optional checkpoints pattern to filter checkpoints inside the prefix (e.g. .*/500.*)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.lighteval_config
    config = get_lighteval_config_from_file(config_file)
    if config.slurm is None:
        raise ValueError("Slurm config is required to run on the cluster.")

    lighteval_runner = LightEvalRunner(config=config)

    # Launch evals
    lighteval_runner.eval_multiple_checkpoints_from_s3(
        args.s3_prefix,
        checkpoints_pattern=args.checkpoints_pattern,
        lighteval_override_config=config_file,
    )
