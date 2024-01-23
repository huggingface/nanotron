"""
DoReMi training script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/config_tiny_llama.yaml
"""
import argparse
import datetime
from typing import List

import torch
import torch.nn.functional as F
from dataloader import get_dataloader
from doremi_context import DoReMiContext
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.trainer import DistributedTrainer

logger = logging.get_logger(__name__)


class ReferenceTrainer(DistributedTrainer):
    def __init__(self, domain_weights: torch.Tensor, domain_keys: List[str], *args, **kwargs):
        # NOTE: save the initial domain_weights
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

    def pre_training(self):
        def get_time_name():
            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        # if dist.get_rank(self.parallel_context.world_pg) == 0:
        #     wandb.init(
        #         project="nanotron",
        #         name=f"{get_time_name()}_doremi_reference_training",
        #         config={"version": 1, "nanotron_config": self.config.as_dict()},
        #     )

    # def train_step_logs(
    #     self,
    #     outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
    #     loss_avg: Optional[torch.Tensor],
    # ):
    #     super().train_step_logs(outputs, loss_avg)
    #     if dist.get_rank(self.parallel_context.world_pg) == 0:
    #         wandb.log(
    #             {
    #                 "loss_avg": loss_avg.cpu().detach().numpy(),
    #                 "step": self.iteration_step,
    #             }
    #         )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # NOTE: for wikicorpus dataset
    DOMAIN_KEYS = [
        "raw_ca",
        "raw_es",
        "raw_en",
    ]
    NUM_DOMAINS = len(DOMAIN_KEYS)
    initial_domain_weights = F.softmax(torch.ones(NUM_DOMAINS, requires_grad=False), dim=-1)

    trainer = ReferenceTrainer(initial_domain_weights, DOMAIN_KEYS, config_file)
    dataloader = get_dataloader(trainer, DOMAIN_KEYS)
    trainer.train(dataloader)
