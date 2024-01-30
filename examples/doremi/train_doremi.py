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

    # DOMAIN_KEYS = ['en', 'en.noblocklist', 'en.noclean', 'realnewslike', 'multilingual', 'af', 'am', 'ar', 'az', 'be', 'bg', 'bg-Latn', 'bn', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'el-Latn', 'en-multi', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi', 'hi-Latn', 'hmn', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'ja-Latn', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'ru-Latn', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tr', 'uk', 'und', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zh-Latn', 'zu']
    # DOMAIN_KEYS = ['en', 'af', 'am', 'ar']
    # TODO(xrsrke): get these automatically

    # NOTE: for miami dataset
    # DOMAIN_KEYS = ["dihana", "ilisten", "loria", "maptask", "vm2"]

    # NOTE: for wikicorpus dataset
    # DOMAIN_KEYS = [
    #     "raw_ca",
    #     "raw_es",
    #     "raw_en",
    #     # 'tagged_ca', 'tagged_es', 'tagged_en' # Use a different column
    # ]
    # NOTE: for mc4 dataset
    # DOMAIN_KEYS = [
    #     "af",
    #     "am",
    #     "az",
    #     "be",
    #     "bg-Latn",
    #     "bn",
    #     "ca",
    #     "ceb",
    #     "co",
    #     "cy",
    #     "el-Latn",
    #     "en",
    #     "eo",
    #     "et",
    #     "eu",
    #     "fil",
    #     "fy",
    #     "ga",
    #     "gd",
    #     "gl",
    #     "gu",
    #     "ha",
    #     "haw",
    #     "hi-Latn",
    #     "hmn",
    #     "ht",
    #     "hy",
    #     "id",
    #     "ig",
    #     "is",
    #     "it",
    #     "iw",
    #     "ja",
    #     "ja-Latn",
    #     "jv",
    #     "ka",
    #     "kk",
    #     "km",
    #     "kn",
    #     "ko",
    #     "ku",
    #     "ky",
    #     "la",
    #     "lb",
    #     "lo",
    #     "lt",
    #     "lv",
    #     "mg",
    #     "mi",
    #     "mk",
    #     "ml",
    #     "mn",
    #     "mr",
    #     "ms",
    #     "mt",
    #     "my",
    #     "ne",
    #     "nl",
    #     "no",
    #     "ny",
    #     "pa",
    #     "pl",
    #     "ps",
    #     "pt",
    #     "ro",
    #     "ru",
    #     "ru-Latn",
    #     "sd",
    #     "si",
    #     "sk",
    #     "sl",
    #     "sm",
    #     "sn",
    #     "so",
    #     "sq",
    #     "sr",
    #     "st",
    #     "su",
    #     "sv",
    #     "sw",
    #     "ta",
    #     "te",
    #     "tg",
    #     "ur",
    #     "uz",
    #     "xh",
    #     "yi",
    #     "yo",
    #     "zh-Latn",
    #     "zu",
    # ]
    # NUM_DOMAINS = len(DOMAIN_KEYS)
    # initial_domain_weights = F.softmax(torch.ones(NUM_DOMAINS, requires_grad=False), dim=-1)

    from pathlib import Path

    # DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_splitted/tokenized_data"
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
    # TOKENIZED_DATASETS = {f"{domain_name}": f"{DATASET_PATH}/{domain_name}" for domain_name in DOMAIN_KEYS}
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

    # trainer = DoReMiTrainer(initial_domain_weights, DOMAIN_KEYS, ref_checkpoint_path=None, config_or_config_file=config_file)
    # dataloader = get_dataloader(trainer, domain_keys=DOMAIN_KEYS, tokenized_datasets=TOKENIZED_DATASETS)

    trainer = DoReMiTrainer(initial_domain_weights, DOMAIN_KEYS, REF_CHECKPOINT_PATH, config_file)
    dataloader = get_dataloader(trainer, domain_keys=DOMAIN_KEYS, tokenized_datasets=TOKENIZED_DATASETS)

    trainer.train(dataloader)
