"""
DoReMi training script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/config_tiny_llama.yaml
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
    config = get_config_from_file(config_file, config_class=DoReMiConfig)
    dataset_paths = [f"{config.data.dataset.hf_dataset_or_datasets}/{name}" for name in config.doremi.domain_names]

    datasets = get_datasets(dataset_paths)
    # TODO(xrsrke): add retrieving domain weights from config
    # or calculate it in the trainer
    initial_domain_weights = compute_domain_weights_based_on_token_count(datasets)
    assert torch.allclose(initial_domain_weights.sum(), torch.tensor(1.0))

    domain_names = config.doremi.domain_names
    ref_model_resume_checkpoint_path = config.doremi.ref_model_resume_checkpoint_path

    # TODO(xrsrke): directly extract domain_names, and ref_model_resume_checkpoint_path from config
    trainer = DoReMiTrainer(
        initial_domain_weights, domain_names, ref_model_resume_checkpoint_path, config_file, config_class=DoReMiConfig
    )
    dataloader = get_dataloader(trainer, datasets)
    trainer.train(dataloader)
