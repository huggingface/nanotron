"""
DoReMi training script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/config_tiny_llama.yaml
"""
import argparse

import torch
from nanotron import logging
from nanotron.doremi.dataloader import get_dataloader
from nanotron.doremi.trainer import DoReMiTrainer

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    from pathlib import Path

    DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_splitted/tokenized_data_with_correct_domain"
    REF_CHECKPOINT_PATH = Path("/fsx/phuc/checkpoints/doremi/reference-280m-llama/22000")
    DOMAIN_KEYS = [
        "Github",
        "FreeLaw",
        "OpenWebText2",
        "PubMed Abstracts",
        "DM Mathematics",
        "OpenSubtitles",
        "HackerNews",
        "NIH ExPorter",
        "PubMed Central",
        "Enron Emails",
    ]
    TOKENIZED_DATASETS = [f"{DATASET_PATH}/{domain_name}" for domain_name in DOMAIN_KEYS]

    # NUM_DOMAINS = len(DOMAIN_KEYS)
    # initial_domain_weights = F.softmax(torch.ones(NUM_DOMAINS, requires_grad=False), dim=-1)
    initial_domain_weights = torch.tensor(
        [
            0.34356916553540745,
            0.16838812972610234,
            0.24711766854236725,
            0.0679225638705455,
            0.059079828519653675,
            0.043720261601881555,
            0.01653850841342608,
            0.00604146633842096,
            0.04342813428189645,
            0.0041942731702987,
        ]
    )

    trainer = DoReMiTrainer(initial_domain_weights, DOMAIN_KEYS, REF_CHECKPOINT_PATH, config_file)
    dataloader = get_dataloader(trainer, dataset_paths=TOKENIZED_DATASETS)

    trainer.train(dataloader)
