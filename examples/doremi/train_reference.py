"""
DoReMi ttraining script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/config_tiny_llama.yaml
"""
import argparse
import datetime
from typing import Dict, Iterable, List, Optional, Union

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.doremi.dataloader import get_dataloader
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.trainer import DistributedTrainer

import wandb

logger = logging.get_logger(__name__)


class ReferenceTrainer(DistributedTrainer):
    def __init__(self, domain_weights: torch.Tensor, domain_keys: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)
        self.doremi_context.domain_weights = self.doremi_context.domain_weights.to("cuda")

        # NOTE: SANITY CHECKS: make sure all ranks have the same domain weights
        assert_tensor_synced_across_pg(
            tensor=self.doremi_context.domain_weights,
            pg=self.parallel_context.world_pg,
            msg=lambda err: f"Domain weights are not synced across ranks {err}",
        )

        log_rank(
            f"[DoReMi] Initial domain weights: {self.doremi_context.domain_weights}", logger=logger, level=logging.INFO
        )

    def post_init(self):
        def get_time_name():
            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        if dist.get_rank(self.parallel_context.world_pg) == 0:
            wandb.init(
                project="nanotron",
                name=f"{get_time_name()}_doremi_reference_training",
                config={"nanotron_config": self.config.as_dict()},
            )

    def train_step_logs(
        self,
        outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        loss_avg: Optional[torch.Tensor],
    ):
        super().train_step_logs(outputs, loss_avg)

        # NOTE: reset the counting in DistributedSamplerForDoReMi
        # trainer.sampler.reset()
        if dist.get_rank(self.parallel_context.world_pg) == 0:
            wandb.log(
                {
                    "loss_avg": loss_avg.item(),
                    "step": self.iteration_step,
                }
            )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # # NOTE: for wikicorpus dataset
    # DOMAIN_KEYS = [
    #     "raw_ca",
    #     "raw_es",
    #     "raw_en",
    # ]

    # DOMAIN_KEYS = ['af', 'am', 'az', 'be', 'bg-Latn', 'bn', 'ca', 'ceb', 'co', 'cy', 'el-Latn', 'en', 'eo', 'et', 'eu', 'fil', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi-Latn', 'hmn', 'ht', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'ja-Latn', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'ru-Latn', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'ur', 'uz', 'xh', 'yi', 'yo', 'zh-Latn', 'zu']
    # DOMAIN_KEYS = ["lt", "az", "ms", "bn", "ca", "cy", "et", "sl"]
    # DOMAIN_KEYS = ["lt", "az", "ms", "bn"]
    # DOMAIN_KEYS = ["ne", "lb", "hy", "sr", "mt"] # 3m sequences in the first shard

    # NOTE: some big domains just in case
    # DOMAIN_KEYS = ["lt", "az", "ms", "bn", "ca", "cy", "et", "sl"]

    # NOTE: the pile
    DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_splitted/tokenized_data"
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

    NUM_DOMAINS = len(DOMAIN_KEYS)
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

    assert len(initial_domain_weights) == NUM_DOMAINS
    assert torch.allclose(initial_domain_weights.sum(), torch.tensor(1.0))

    trainer = ReferenceTrainer(initial_domain_weights, DOMAIN_KEYS, config_file)
    # dist.barrier()
    # import time

    # # time.sleep(3)

    # # dist.barrier()

    dataloader = get_dataloader(trainer, domain_keys=DOMAIN_KEYS, tokenized_datasets=TOKENIZED_DATASETS)
    # trainer.sampler = dataloader.sampler
    trainer.train(dataloader)
