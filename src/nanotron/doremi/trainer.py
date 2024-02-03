import datetime
from pprint import pformat
from typing import Dict, Iterable, List, Optional, Union

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    ExistingCheckpointInit,
    RandomInit,
)
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.doremi.llama import LlamaForDoReMiTraining, LLaMaForInference
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

import wandb

logger = logging.get_logger(__name__)


class DoReMiTrainer(DistributedTrainer):
    def __init__(
        self, domain_weights: torch.Tensor, domain_keys: List[str], ref_checkpoint_path: str, *args, **kwargs
    ):
        # NOTE: save the initial domain_weights
        self.doremi_context = DoReMiContext(
            domain_weights,
            domain_keys,
            is_proxy=True,
            step_size=1,
            smoothing_param=1e-3,
        )
        # TODO: add randomly initialize reference model
        self.ref_checkpoint_path = ref_checkpoint_path
        super().__init__(*args, **kwargs)

    def init_model(self) -> Union[NanotronModel, DistributedDataParallel]:
        """Initialize the model and load weights from checkpoint if needed."""

        # NOTE: after initializing parallel context, now we can move domain weights to
        # the GPU corresponding to the current rank
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
            model_builder=lambda: LlamaForDoReMiTraining(
                config=self.model_config,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
                doremi_context=self.doremi_context,
            ),
        )
        normalized_model = model.module if isinstance(model, DistributedDataParallel) else model

        log_rank("[DoReMi] Initializing reference model for DoReMi training", logger=logger, level=logging.INFO)

        self.ref_model = self._init_model(
            model_builder=lambda: LLaMaForInference(
                config=self.model_config,
                parallel_config=self.config.parallelism,
                parallel_context=self.parallel_context,
            ),
        )
        self.ref_model.eval()
        for _, param in self.ref_model.named_parameters():
            param.requires_grad_(False)

        # Load or initialize model weights
        self.init_checkpoint_path = parse_ckpt_path(config=self.config)
        reloaded_from_checkpoint = False
        if self.init_checkpoint_path is not None:
            # Reload from a training checkpoint
            log_rank(f"Loading weights from {self.init_checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
            self.param_shard_metadata = load_weights(
                model=normalized_model, parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
            )
            # load_weights(
            #     model=self.ref_model, parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
            # )
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

                # load_weights(
                #     model=self.ref_model,
                #     parallel_context=self.parallel_context,
                #     root_folder=self.config.model.init_method.path,
                # )
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
                for _, param in sorted(model.named_parameters(), key=lambda x: x[0]):
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

        if self.ref_checkpoint_path is not None:
            normalized_ref_model = (
                self.ref_model.module
                if isinstance(self.ref_model.module, DistributedDataParallel)
                else self.ref_model.module
            )

            log_rank(
                f"Loading weights from {self.ref_checkpoint_path} for reference model",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            load_weights(
                model=normalized_ref_model,
                parallel_context=self.parallel_context,
                root_folder=self.ref_checkpoint_path,
            )
            # reloaded_from_checkpoint = True

        return model

    # def pre_init(self):
    #     # NOTE: after initializing parallel context, now we can move domain weights to
    #     # the GPU corresponding to the current rank
    #     self.doremi_context.domain_weights = self.doremi_context.domain_weights.to("cuda")

    #     # NOTE: SANITY CHECKS: make sure all ranks have the same domain weights
    #     assert_tensor_synced_across_pg(
    #         tensor=self.doremi_context.domain_weights,
    #         pg=self.parallel_context.world_pg,
    #         msg=lambda err: f"Domain weights are not synced across ranks {err}",
    #     )

    #     log_rank(
    #         f"[DoReMi] Initial domain weights: {self.doremi_context.domain_weights}", logger=logger, level=logging.INFO
    #     )

    # def post_init(self):
    #     """Initialize the model and load weights from checkpoint if needed."""
    #     log_rank("[DoReMi] Initializing reference model for DoReMi training", logger=logger, level=logging.INFO)

    #     self.ref_model = self._init_model(
    #         model_builder=lambda: LLaMaForInference(
    #             config=self.model_config,
    #             parallel_config=self.config.parallelism,
    #             parallel_context=self.parallel_context,
    #         ),
    #     )
    #     self.ref_model.eval()
    #     for _, param in self.ref_model.named_parameters():
    #         param.requires_grad_(False)

    #     reloaded_from_checkpoint = False
    #     if self.init_checkpoint_path is not None:
    #         # Reload from a training checkpoint
    #         log_rank(f"Loading weights from {self.init_checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
    #         load_weights(
    #             model=self.ref_model, parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
    #         )
    #         reloaded_from_checkpoint = True

    #     if not reloaded_from_checkpoint:
    #         log_rank("No checkpoint path provided.", logger=logger, level=logging.INFO)
    #         if isinstance(self.config.model.init_method, ExistingCheckpointInit):
    #             load_weights(
    #                 model=self.ref_model,
    #                 parallel_context=self.parallel_context,
    #                 root_folder=self.config.model.init_method.path,
    #             )
    #         elif isinstance(self.config.model.init_method, RandomInit):
    #             # # Initialize model randomly
    #             # normalized_model.init_model_randomly(
    #             #     init_method=init_method_normal(self.config.model.init_method.std),
    #             #     scaled_init_method=scaled_init_method_normal(
    #             #         self.config.model.init_method.std, self.model_config.num_hidden_layers
    #             #     ),
    #             # )
    #             # # Synchronize parameters so that the model is consistent
    #             # # sync all params across dp
    #             # for _, param in sorted(model.named_parameters(), key=lambda x: x[0]):
    #             #     dist.all_reduce(param, op=dist.ReduceOp.AVG, group=self.parallel_context.dp_pg)

    #             # # sync tied params across tied groups
    #             # for (_, group_ranks), param in sorted(
    #             #     get_tied_id_to_param(
    #             #         parameters=model.parameters(),
    #             #         root_module=normalized_model,
    #             #     ).items(),
    #             #     key=lambda x: x[0],
    #             # ):
    #             #     group = self.parallel_context.world_ranks_to_pg[group_ranks]
    #             #     dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
    #             pass
    #         else:
    #             raise ValueError(f"Unsupported {self.config.model.init_method}")

    def pre_training(self):
        def get_time_name():
            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        if dist.get_rank(self.parallel_context.world_pg) == 0:
            wandb.init(
                project="nanotron",
                name=f"{get_time_name()}_{self.config.general.project}_{self.config.general.run}",
                config={
                    "version": 1,
                    "nanotron_config": self.config.as_dict(),
                    "doremi": {
                        "smoothing_param": self.doremi_context.smoothing_param,
                        "step_size": self.doremi_context.step_size,
                        "domain_keys": self.doremi_context.domain_keys,
                        "initial_domain_weights": self.doremi_context.domain_weights.cpu().detach().numpy(),
                    },
                },
            )

    def train_step_logs(
        self,
        outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        loss_avg: Optional[torch.Tensor],
    ):
        domain_weights = outputs[0]["domain_weights"]
        domain_losses = outputs[0]["domain_losses"]
        samples_per_domain = outputs[0]["samples_per_domain"].tolist()

        handle_weight = dist.all_reduce(
            domain_weights, group=self.parallel_context.dp_pg, async_op=True, op=dist.ReduceOp.AVG
        )
        handle_loss = dist.all_reduce(
            domain_losses, group=self.parallel_context.dp_pg, async_op=True, op=dist.ReduceOp.AVG
        )

        super().train_step_logs(outputs, loss_avg)

        handle_weight.wait()
        handle_loss.wait()

        self.doremi_context.add_weight_with_history(domain_weights, self.iteration_step)

        domain_weights = domain_weights.cpu().detach().numpy()
        domain_losses = domain_losses.cpu().detach().numpy()

        log_rank(
            f"[DoReMi] Domain weights: {str(domain_weights)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
            group=self.parallel_context.dp_pg,
        )

        log_rank(
            f"[DoReMi] Domain loss: {str(domain_losses)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
            group=self.parallel_context.dp_pg,
        )

        if dist.get_rank(self.parallel_context.world_pg) == 0:
            if self.iteration_step % self.config.checkpoints.checkpoint_interval == 0:
                checkpoints_path = self.config.checkpoints.checkpoints_path
                checkpoint_path = checkpoints_path / f"doremi_domain_weights_{self.iteration_step}.pt"
                torch.save(self.doremi_context.domain_weight_history, checkpoint_path)

            weight_logs = {
                f"weight_domain_{self.doremi_context.get_domain_name(i)}": weight
                for i, weight in enumerate(domain_weights)
            }
            loss_logs = {
                f"loss_domain_{self.doremi_context.get_domain_name(i)}": loss for i, loss in enumerate(domain_losses)
            }
            samples_per_domain_logs = {
                f"samples_per_domain_{self.doremi_context.get_domain_name(i)}": samples
                for i, samples in enumerate(samples_per_domain)
            }

            wandb.log(
                {
                    **weight_logs,
                    **loss_logs,
                    **samples_per_domain_logs,
                    "loss_avg": loss_avg.cpu().detach().numpy(),
                    # "lm_loss": outputs[0]["lm_loss"].cpu().detach().numpy(),
                    "step": self.iteration_step,
                }
            )
