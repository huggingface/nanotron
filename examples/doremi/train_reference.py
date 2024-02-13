"""
DoReMi ttraining script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/config_tiny_llama.yaml
"""
import argparse
from pprint import pformat
from typing import Dict, Iterable, List, Optional, Union

import torch
import wandb
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    ExistingCheckpointInit,
    RandomInit,
    get_config_from_file,
)
from nanotron.doremi.config import DoReMiConfig
from nanotron.doremi.dataloader import get_dataloader, get_datasets
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.doremi.llama import LlamaReferenceForTrainingWithPerDomainLoss
from nanotron.doremi.utils import compute_domain_weights_based_on_token_count
from nanotron.helpers import _vocab_size_with_padding
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.serialize import load_weights, parse_ckpt_path
from nanotron.trainer import DistributedTrainer
from nanotron.utils import init_method_normal, scaled_init_method_normal
from torch.nn.parallel import DistributedDataParallel

logger = logging.get_logger(__name__)


class ReferenceTrainer(DistributedTrainer):
    def __init__(self, domain_weights: torch.Tensor, domain_keys: List[str], *args, **kwargs):
        self.doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)
        self.valid_dataloader = None
        super().__init__(*args, **kwargs)
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

    def init_model(self) -> Union[NanotronModel, DistributedDataParallel]:
        """Initialize the model and load weights from checkpoint if needed."""
        # TODO: add max_position_embeddings
        self.model_config.vocab_size = _vocab_size_with_padding(
            self.model_config.vocab_size,
            pg_size=self.parallel_context.tp_pg.size(),
            make_vocab_size_divisible_by=self.config.model.make_vocab_size_divisible_by,
        )

        if (
            getattr(self.model_config, "max_position_embeddings", None) is not None
            and self.model_config.max_position_embeddings != self.config.tokens.sequence_length
        ):
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                log_rank(
                    f"Finetuning a model with a sequence length {self.config.tokens.sequence_length} that is different from the checkpoint's max_position_embeddings {self.model_config.max_position_embeddings}.",  # noqa
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )
            else:
                log_rank(
                    f"Setting max_position_embeddings to {self.config.tokens.sequence_length}. Previous value was {self.model_config.max_position_embeddings}.",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
                self.model_config.max_position_embeddings = self.config.tokens.sequence_length

        log_rank(pformat(self.config), logger=logger, level=logging.INFO, rank=0)
        log_rank(pformat(self.model_config), logger=logger, level=logging.INFO, rank=0)

        model = self._init_model(
            model_builder=lambda: LlamaReferenceForTrainingWithPerDomainLoss(
                config=self.model_config,
                doremi_context=self.doremi_context,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
            ),
        )
        normalized_model = model.module if isinstance(model, DistributedDataParallel) else model

        # Load or initialize model weights
        self.init_checkpoint_path = parse_ckpt_path(config=self.config)
        reloaded_from_checkpoint = False
        if self.init_checkpoint_path is not None:
            # Reload from a training checkpoint
            log_rank(f"Loading weights from {self.init_checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
            self.param_shard_metadata = load_weights(
                model=normalized_model, parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
            )
            reloaded_from_checkpoint = True
        if not reloaded_from_checkpoint:
            log_rank("No checkpoint path provided.", logger=logger, level=logging.INFO)
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an pretrained model checkpoint
                self.param_shard_metadata = load_weights(
                    model=normalized_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.config.model.init_method.path,
                )
            elif isinstance(self.config.model.init_method, RandomInit):
                # Initialize model randomly
                normalized_model.init_model_randomly(
                    init_method=init_method_normal(self.config.model.init_method.std),
                    scaled_init_method=scaled_init_method_normal(
                        self.config.model.init_method.std, self.model_config.num_hidden_layers
                    ),
                )
                # Synchronize parameters so that the model is consistent
                # sync all params across dp
                for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=self.parallel_context.dp_pg)

                # sync tied params across tied groups
                for (_, group_ranks), param in sorted(
                    get_tied_id_to_param(
                        parameters=model.parameters(),
                        root_module=normalized_model,
                    ).items(),
                    key=lambda x: x[0],
                ):
                    group = self.parallel_context.world_ranks_to_pg[group_ranks]
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")

        return model

    def post_init(self):
        import datetime

        def get_time_name():
            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        if dist.get_rank(self.parallel_context.world_pg) == 0:
            wandb.init(
                project="nanotron",
                name=f"{get_time_name()}_{self.config.general.project}_{self.config.general.run}",
                config={
                    "nanotron_config": self.config.as_dict(),
                    "doremi": {
                        "smoothing_param": self.doremi_context.smoothing_param,
                        "step_size": self.doremi_context.step_size,
                        "domain_keys": self.doremi_context.domain_keys,
                        "initial_domain_weights": self.doremi_context.domain_weights.tolist(),
                    },
                },
            )

    def train_step_logs(
        self,
        outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        loss_avg: Optional[torch.Tensor],
    ):
        super().train_step_logs(outputs, loss_avg)

        domain_losses = outputs[0]["domain_losses"].tolist()
        samples_per_domain = outputs[0]["samples_per_domain"].tolist()

        log_rank(
            f"[DoReMi][Train] Domain loss: {str(domain_losses)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        log_rank(
            f"[DoReMi][Train] Samples per domain: {str(samples_per_domain)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        if dist.get_rank(self.parallel_context.world_pg) == 0:
            loss_logs = {
                f"loss_domain_{self.doremi_context.get_domain_name(i)}": loss for i, loss in enumerate(domain_losses)
            }

            samples_per_domain_logs = {
                f"samples_per_domain_{self.doremi_context.get_domain_name(i)}": n_samples
                for i, n_samples in enumerate(samples_per_domain)
            }

            wandb.log(
                {
                    **loss_logs,
                    **samples_per_domain_logs,
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
    config = get_config_from_file(config_file, config_class=DoReMiConfig)

    dataset_paths = [f"{config.data.dataset.hf_dataset_or_datasets}/{name}" for name in config.doremi.domain_names]
    datasets = get_datasets(dataset_paths)

    # TODO(xrsrke): add retrieving domain weights from config
    # or calculate it in the trainer
    if config.doremi.domain_weights is None:
        initial_domain_weights = compute_domain_weights_based_on_token_count(datasets)
    else:
        initial_domain_weights = torch.tensor(config.doremi.domain_weights)

    assert torch.allclose(initial_domain_weights.sum(), torch.tensor(1.0), rtol=1e-3)

    domain_names = config.doremi.domain_names
    trainer = ReferenceTrainer(initial_domain_weights, domain_names, config_file, config_class=DoReMiConfig)
    dataloader = get_dataloader(trainer, datasets)
    trainer.train(dataloader)
