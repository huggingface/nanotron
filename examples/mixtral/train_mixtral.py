"""

You can run using command:
```
python examples/mixtral/config_mixtral_tiny.py; CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/mixtral/train_mixtral.py --config-file examples/mixtral/config_mixtral.yaml
```
"""
import argparse
import os
import sys

from config_mixtral import MixtralConfig
from mixtral import MixtralForTraining
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
    trainer = DistributedTrainer(config_file, model_config_class=MixtralConfig, model_class=MixtralForTraining)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
