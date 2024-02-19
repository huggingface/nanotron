"""
DoReMi training script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/config_280m_llama_proxy.yaml
"""
import argparse

import torch
from nanotron.config import get_config_from_file
from nanotron.doremi.config import DoReMiConfig
from nanotron.doremi.dataloader import get_dataloader, get_datasets
from nanotron.doremi.trainer import DoReMiTrainer
from nanotron.doremi.utils import compute_domain_weights_based_on_token_count


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    config: DoReMiConfig = get_config_from_file(config_file, config_class=DoReMiConfig)

    dataset_paths = [f"{config.data.dataset.hf_dataset_or_datasets}/{name}" for name in config.doremi.domain_names]
    datasets = get_datasets(dataset_paths)

    # TODO(xrsrke): add retrieving domain weights from config
    # or calculate it in the trainer
    if config.doremi.domain_weights is None:
        domain_weights = compute_domain_weights_based_on_token_count(datasets)
    else:
        domain_weights = torch.tensor(config.doremi.domain_weights)

    trainer = DoReMiTrainer(domain_weights, config_file, config_class=DoReMiConfig)
    dataloader = get_dataloader(trainer, datasets)
    trainer.train(dataloader)
