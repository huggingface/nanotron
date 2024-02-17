"""

You can run using command:
```
python examples/moe/config_llamoe.py; USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/moe/train_moe.py --config-file examples/moe/config_llamoe.yaml
```
"""
import argparse
import os
import sys

from config_llamoe import LlaMoEConfig
from llamoe import LlaMoEForTraining
from nanotron import logging
from nanotron.trainer import DistributedTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from run_train import get_dataloader  # noqa

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    parser.add_argument("--job-id", type=str, help="Optional job name")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file, model_config_class=LlaMoEConfig, model_class=LlaMoEForTraining)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
