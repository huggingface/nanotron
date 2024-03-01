import argparse
import os
import sys

from mamba.config_mamba import MambaConfig
from mamba.mamba import MambaForTraining

from nanotron import logging
from nanotron.trainer import DistributedTrainer

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
    trainer = DistributedTrainer(config_file, model_config_class=MambaConfig, model_class=MambaForTraining)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
