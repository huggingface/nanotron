import argparse
import os
import sys

from config import MambaModelConfig
from mamba import MambaForTraining
from trainer import MambaTrainer

from nanotron import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from run_train import get_dataloader  # noqa

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = MambaTrainer(config_file, model_config_class=MambaModelConfig, model_class=MambaForTraining)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
