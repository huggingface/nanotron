"""
DoReMi training script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/configs/config_280m_llama_proxy.yaml
"""
import argparse

# import torch
from nanotron.dataloader import get_dataloader

from doremi.config import DoReMiConfig

# from doremi.dataloader import get_dataloader, get_datasets
from doremi.dataloader import get_dataloader_for_proxy_training
from doremi.trainer import DoReMiTrainer

# from doremi.utils import compute_domain_weights_based_on_token_count

# from nanotron.config import get_config_from_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    # config: DoReMiConfig = get_config_from_file(config_file, config_class=DoReMiConfig)

    # dataset_paths = [f"{config.data.dataset.hf_dataset_or_datasets}/{name}" for name in config.doremi.domain_names]
    # datasets = get_datasets(dataset_paths)

    # # TODO(xrsrke): add retrieving domain weights from config
    # # or calculate it in the trainer
    # if config.doremi.domain_weights is None:
    #     domain_weights = compute_domain_weights_based_on_token_count(datasets)
    # else:
    #     domain_weights = torch.tensor(config.doremi.domain_weights)

    trainer = DoReMiTrainer(config_file, config_class=DoReMiConfig)
    dataloader = get_dataloader(trainer)

    # world_pg = trainer.parallel_context.world_pg
    for samples in dataloader:
        assert 1 == 1
        # _input_ids = torch.tensor(train_dataset[i]["input_ids"], device="cuda")
        # _dataset_idxs = torch.tensor(train_dataset[i]["dataset_idxs"], device="cuda")
        # assert_tensor_synced_across_pg(_input_ids, world_pg, msg=lambda err: f"input_ids {err}, i: {i}")
        # assert_tensor_synced_across_pg(_dataset_idxs, world_pg, msg=lambda err: f"input_ids {err}, i: {i}")

    dataloader = get_dataloader_for_proxy_training(dataloader, ref_model=trainer.ref_model)
    trainer.train(dataloader)
