"""
DoReMi ttraining script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/config_tiny_llama.yaml
"""
import argparse
import datetime
from pprint import pformat
from typing import Dict, Iterable, Iterator, List, Union

import torch
import wandb
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    Config,
    ExistingCheckpointInit,
    RandomInit,
    get_config_from_file,
)
from nanotron.doremi.config import DoReMiConfig
from nanotron.doremi.dataloader import get_dataloader, get_datasets
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.doremi.llama import LlamaReferenceForTrainingWithPerDomainLoss
from nanotron.helpers import _vocab_size_with_padding, init_random_states
from nanotron.logging import log_rank, set_logger_verbosity_format
from nanotron.models import NanotronModel
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.random import set_random_seed
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.serialize import load_weights, parse_ckpt_path
from nanotron.trainer import mark_tied_parameters
from nanotron.utils import init_method_normal, scaled_init_method_normal
from torch.nn.parallel import DistributedDataParallel

logger = logging.get_logger(__name__)


# class EvalRunner(DistributedTrainer):
class EvalRunner:
    def __init__(
        self, domain_weights: torch.Tensor, domain_keys: List[str], config_or_config_file, config_class=Config
    ):
        self.config = get_config_from_file(config_or_config_file, config_class=config_class)
        self.model_config = self.config.model.model_config

        ########################################
        ## We start with setting up loggers and process groups
        ########################################

        # Initialise all process groups
        self.parallel_context = ParallelContext(
            tensor_parallel_size=self.config.parallelism.tp,
            pipeline_parallel_size=self.config.parallelism.pp,
            data_parallel_size=self.config.parallelism.dp,
        )

        self.doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)
        self.doremi_context.domain_weights = self.doremi_context.domain_weights.to("cuda")

        assert_tensor_synced_across_pg(
            tensor=self.doremi_context.domain_weights,
            pg=self.parallel_context.world_pg,
            msg=lambda err: f"Domain weights are not synced across ranks {err}",
        )

        log_rank(
            f"[DoReMi] Initial domain weights: {self.doremi_context.domain_weights}", logger=logger, level=logging.INFO
        )

        # Set log levels
        if dist.get_rank(self.parallel_context.world_pg) == 0:
            if self.config.logging.log_level is not None:
                set_logger_verbosity_format(self.config.logging.log_level, parallel_context=self.parallel_context)
        else:
            if self.config.logging.log_level_replica is not None:
                set_logger_verbosity_format(
                    self.config.logging.log_level_replica, parallel_context=self.parallel_context
                )

        # # Log benchmark info
        # if os.environ.get("NANOTRON_BENCHMARK", "0") == "1":
        #     log_throughput(self.config, self.parallel_context)

        ########################################
        ## Setting up our model, optimizers, schedulers, etc.
        ########################################

        # Set random states
        set_random_seed(self.config.general.seed)

        # Init model and build on pp ranks
        self.random_states = init_random_states(
            parallel_config=self.config.parallelism, tp_pg=self.parallel_context.tp_pg
        )
        self.model = self.init_model()  # Defines self.model
        self.normalized_model: NanotronModel = (
            self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        )

        # Init optimizer
        # self.optimizer, self.grad_accumulator = init_optimizer_and_grad_accumulator(
        #     model=self.model, optimizer_args=self.config.optimizer, parallel_context=self.parallel_context
        # )
        # if self.init_checkpoint_path is not None:
        #     load_optimizer(
        #         optimizer=self.optimizer,
        #         parallel_context=self.parallel_context,
        #         root_folder=self.init_checkpoint_path,
        #         param_shard_metadata=self.param_shard_metadata,
        #         model=self.model,
        #     )

        # Define iteration start state
        self.start_iteration_step: int
        self.consumed_train_samples: int
        # if self.init_checkpoint_path is not None:
        #     checkpoint_metadata = load_meta(
        #         parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
        #     )
        #     log_rank(str(checkpoint_metadata), logger=logger, level=logging.INFO, rank=0)
        #     self.start_iteration_step = checkpoint_metadata.metas["last_train_step"]
        #     self.consumed_train_samples = checkpoint_metadata.metas["consumed_train_samples"]
        #     assert (
        #         self.config.tokens.train_steps > self.start_iteration_step
        #     ), f"Loaded checkpoint has already trained {self.start_iteration_step} batches, you need to specify a higher `config.tokens.train_steps`"
        # else:
        #     self.start_iteration_step = 0
        #     self.consumed_train_samples = 0

        self.start_iteration_step = 0
        self.consumed_train_samples = 0

        # Setup tensorboard write and log writers on output rank
        self.logger_ranks = self.parallel_context.world_rank_matrix[
            self.normalized_model.output_pp_rank, 0, 0
        ].flatten()
        # self.loggerwriter = self.setup_log_writers()

        # Log where each module is instantiated
        self.normalized_model.log_modules(level=logging.DEBUG, group=self.parallel_context.world_pg, rank=0)

        self.micro_batch_size = self.config.tokens.micro_batch_size
        self.n_micro_batches_per_batch = self.config.tokens.batch_accumulation_per_replica
        self.global_batch_size = (
            self.micro_batch_size * self.n_micro_batches_per_batch * self.parallel_context.dp_pg.size()
        )
        self.sequence_length = self.config.tokens.sequence_length
        # self.iteration_step = self.start_iteration_step
        self.limit_val_batches = self.config.tokens.limit_val_batches

        self.post_init()

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

        # log_rank(pformat(self.config), logger=logger, level=logging.INFO, rank=0)
        log_rank(pformat(self.config), logger=logger, level=logging.INFO, rank=0)
        log_rank(pformat(self.model_config), logger=logger, level=logging.INFO, rank=0)

        # model_config_cls = self.model_config.__class__.__name__
        # assert (
        #     model_config_cls in CONFIG_TO_MODEL_CLASS
        # ), f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"

        # TODO(xrsrke): split loading weights
        # from model initialization in base trainer => less code duplication
        # model = self._init_model(
        #     model_builder=lambda: LlamaReferenceForTrainingWithPerDomainLoss(
        #         config=self.model_config,
        #         doremi_context=self.doremi_context,
        #         parallel_context=self.parallel_context,
        #         parallel_config=self.config.parallelism,
        #         # random_states=self.random_states,
        #     ),
        # )

        from nanotron.models import build_model

        model = build_model(
            parallel_context=self.parallel_context,
            dtype=self.config.model.dtype,
            target_pp_ranks=None,
            model_builder=lambda: LlamaReferenceForTrainingWithPerDomainLoss(
                config=self.model_config,
                doremi_context=self.doremi_context,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
                # random_states=self.random_states,
            ),
        )

        mark_tied_parameters(
            model=model, parallel_context=self.parallel_context, parallel_config=self.config.parallelism
        )

        # Check that the model has at least one grad. Necessary for DDP
        # check_model_has_grad(model=model, parallel_context=parallel_context)
        # TODO @thomasw21: DDP doesn't support broadcasting complex buffers (and we don't really need that broadcasting anyway)
        model = DistributedDataParallel(
            model,
            process_group=self.parallel_context.dp_pg,
            broadcast_buffers=False,
            bucket_cap_mb=self.config.model.ddp_bucket_cap_mb,
        )

        # Sanity check the model, all parameters must be NanotronParameter (either tied or sharded)
        sanity_check(root_module=model)

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
        def get_time_name():
            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        if dist.get_rank(self.parallel_context.world_pg) == 0:
            wandb.init(
                project="nanotron",
                name=f"{get_time_name()}_eval_{self.config.general.project}_{self.config.general.run}",
                config={
                    "nanotron_config": self.config.as_dict(),
                    "doremi": {
                        # TODO(xrsrke): support not hardcoding these
                        # "resume_from_step": 2000,
                        "smoothing_param": self.doremi_context.smoothing_param,
                        "step_size": self.doremi_context.step_size,
                        "domain_keys": self.doremi_context.domain_keys,
                        "initial_domain_weights": self.doremi_context.domain_weights.cpu().detach().numpy(),
                    },
                },
            )

    def eval(self, dataloader):
        from nanotron.dataloader import sanity_check_dataloader

        dataloader = iter(dataloader)
        dataloader = sanity_check_dataloader(
            dataloader=dataloader, parallel_context=self.parallel_context, config=self.config
        )
        from nanotron.parallel.pipeline_parallel.engine import PipelineEngine

        self.pipeline_engine: PipelineEngine = self.config.parallelism.pp_engine
        self.pipeline_engine.nb_microbatches = self.n_micro_batches_per_batch

        for step in range(1000):
            valid_outputs = self.validation_step(dataloader=dataloader)

            loss_avg = torch.stack([output["loss"] for output in valid_outputs]).sum()
            dist.all_reduce(loss_avg, group=self.parallel_context.dp_pg, op=dist.ReduceOp.AVG)

            loss_avg = loss_avg.cpu().detach().numpy()
            valid_domain_losses = valid_outputs[0]["domain_losses"].cpu().detach().numpy()
            valid_samples_per_domain = valid_outputs[0]["samples_per_domain"].cpu().detach().numpy()

            log_rank(
                f"[DoReMi][Validation] Step: {step} | Loss: {str(loss_avg)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
                group=self.parallel_context.tp_pg,
            )

            log_rank(
                f"[DoReMi][Validation] Step: {step} | Domain loss: {str(valid_domain_losses)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
                group=self.parallel_context.tp_pg,
            )

            log_rank(
                f"[DoReMi][Validation] Step: {step} | Samples per domain: {str(valid_samples_per_domain)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
                group=self.parallel_context.tp_pg,
            )

            if dist.get_rank(self.parallel_context.world_pg) == 0:
                valid_loss_logs = {
                    f"valid_loss_domain_{self.doremi_context.get_domain_name(i)}": loss
                    for i, loss in enumerate(valid_domain_losses)
                }

                valid_samples_per_domain_logs = {
                    f"valid_samples_per_domain_{self.doremi_context.get_domain_name(i)}": n_samples
                    for i, n_samples in enumerate(valid_samples_per_domain)
                }

                wandb.log(
                    {
                        **valid_loss_logs,
                        **valid_samples_per_domain_logs,
                        "loss_avg": loss_avg,
                        "step": step,
                    }
                )

    def validation_step(self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]) -> Iterable[Dict]:
        outputs = self.pipeline_engine.validate_batch_iter(
            model=self.model,
            batch=(next(dataloader) for _ in range(self.limit_val_batches)),
            nb_microbatches=self.limit_val_batches,
        )
        return outputs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    config = get_config_from_file(config_file, config_class=DoReMiConfig)

    domain_names = config.doremi.domain_names
    NUM_DOMAINS = len(domain_names)
    VALID_DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/tokenized_data/test"
    # DOMAIN_KEYS = [
    #     "Github",
    #     "FreeLaw",
    #     "OpenWebText2",
    #     "PubMed Abstracts",
    #     "DM Mathematics",
    #     "OpenSubtitles",
    #     "HackerNews",
    #     "NIH ExPorter",
    #     "PubMed Central",
    #     "Enron Emails",
    # ]
    TOKENIZED_VALID_DATASET_PATHS = [f"{VALID_DATASET_PATH}/{domain_name}" for domain_name in domain_names]
    datasets = get_datasets(TOKENIZED_VALID_DATASET_PATHS)

    import torch.nn.functional as F

    initial_domain_weights = F.softmax(torch.ones(NUM_DOMAINS, requires_grad=False), dim=-1)

    # initial_domain_weights = torch.tensor(
    #     [0.06299, 0.177, 0.528, 0.1025, 0.0034, 0.02008, 0.01621, 0.009924, 0.07446, 0.005524]
    # )
    # initial_domain_weights = torch.tensor(
    #     [
    #         0.34356916553540745,
    #         0.16838812972610234,
    #         0.24711766854236725,
    #         0.0679225638705455,
    #         0.059079828519653675,
    #         0.043720261601881555,
    #         0.01653850841342608,
    #         0.00604146633842096,
    #         0.04342813428189645,
    #         0.0041942731702987,
    #     ]
    # )
    # initial_domain_weights = compute_domain_weights_based_on_token_count(datasets)

    assert len(initial_domain_weights) == NUM_DOMAINS
    # assert torch.allclose(initial_domain_weights.sum(), torch.tensor(1.0))

    trainer = EvalRunner(initial_domain_weights, domain_names, config_file, config_class=DoReMiConfig)
    dataloader = get_dataloader(trainer, datasets=datasets)
    trainer.eval(dataloader)
