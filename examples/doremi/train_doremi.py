"""
DoReMi training script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/config_tiny_llama.yaml
"""
import argparse

import torch
import torch.nn.functional as F
from dataloader import get_dataloader
from nanotron import logging
from trainer import DoReMiTrainer

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # DOMAIN_KEYS = ['en', 'en.noblocklist', 'en.noclean', 'realnewslike', 'multilingual', 'af', 'am', 'ar', 'az', 'be', 'bg', 'bg-Latn', 'bn', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'el-Latn', 'en-multi', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi', 'hi-Latn', 'hmn', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'ja-Latn', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'ru-Latn', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tr', 'uk', 'und', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zh-Latn', 'zu']
    # DOMAIN_KEYS = ['en', 'af', 'am', 'ar']
    # TODO(xrsrke): get these automatically

    # NOTE: for wikicorpus dataset
    # DOMAIN_KEYS = ["dihana", "ilisten", "loria", "maptask", "vm2"]

    # # NOTE: for wikicorpus dataset
    DOMAIN_KEYS = [
        "raw_ca",
        "raw_es",
        "raw_en",
        # 'tagged_ca', 'tagged_es', 'tagged_en' # Use a different column
    ]
    NUM_DOMAINS = len(DOMAIN_KEYS)
    initial_domain_weights = F.softmax(torch.ones(NUM_DOMAINS, requires_grad=False), dim=-1)

    trainer = DoReMiTrainer(initial_domain_weights, config_file)
    # TODO(xrsrke): check the micro batch size is larger than the number of domains
    dataloader = get_dataloader(trainer, DOMAIN_KEYS)

    trainer.train(dataloader)
