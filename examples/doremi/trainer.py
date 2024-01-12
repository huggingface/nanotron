from pprint import pformat
from typing import Optional, Tuple

# from nanotron.models.fast.llama import LlamaModel
from llama import LlamaForDoReMiTraining, LlamaModelWithoutPP
from nanotron.config import (
    ExistingCheckpointInit,
    RandomInit,
)

# from nanotron.config import DPOPretrainDatasetsArgs as PretrainDatasetsArgs
from nanotron.core import distributed as dist
from nanotron.core import logging
from nanotron.core.logging import (
    log_rank,
)

# from examples.llama.modeling_dpo import DPOForTraining, LlamaModel
from nanotron.core.parallel.tied_parameters import (
    get_tied_id_to_param,
)
from nanotron.core.tensor_init import init_method_normal, scaled_init_method_normal

# from nanotron.core.serialize import (
#     load_weights,
# )
from nanotron.helpers import (
    _vocab_size_with_padding,
)
from nanotron.models import NanotronModel
from nanotron.serialize import (
    load_weights,
    parse_ckpt_path,
)

# from nanotron.logger import LogItem
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, DistributedTrainer
from torch.nn.parallel import DistributedDataParallel

# from .dataloaders.dpo import dpo_data_generator

logger = logging.get_logger(__name__)


class DoReMiTrainer(DistributedTrainer):
    def init_model(self) -> Tuple[NanotronModel, Optional[str]]:
        """Initialize the model and load weights from checkpoint if needed."""
        # TODO: add max_position_embeddings
        self.model_config.vocab_size = _vocab_size_with_padding(
            self.model_config.vocab_size,
            pg_size=self.dpg.tp_pg.size(),
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

        model_config_cls = self.model_config.__class__.__name__
        assert (
            model_config_cls in CONFIG_TO_MODEL_CLASS
        ), f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"

        model = self._init_model(
            model_builder=lambda: LlamaForDoReMiTraining(
                config=self.model_config,
                dpg=self.dpg,
                parallel_config=self.config.parallelism,
                random_states=self.random_states,
            ),
        )
        normalized_model = model.module if isinstance(model, DistributedDataParallel) else model

        # ref_model = self._init_model(
        #     model_builder=lambda: LLaMaForInference(
        #         config=self.model_config,
        #         dpg=self.dpg,
        #         parallel_config=self.config.parallelism,
        #         # random_states=self.random_states,
        #         # random_states=self.random_states,
        #     ),
        # )
        ref_model = LlamaModelWithoutPP(
            config=self.model_config,
            dpg=self.dpg,
            parallel_config=self.config.parallelism,
        )
        ref_model.eval()
        for _, param in ref_model.named_parameters():
            param.requires_grad_(False)

        self.ref_model = model

        # Load or initialize model weights
        checkpoint_path = parse_ckpt_path(config=self.config)
        reloaded_from_checkpoint = False
        if checkpoint_path is not None:
            # Reload from a training checkpoint
            log_rank(f"Loading weights from {checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
            load_weights(model=normalized_model, dpg=self.dpg, root_folder=checkpoint_path)
            reloaded_from_checkpoint = True
        if not reloaded_from_checkpoint:
            log_rank("No checkpoint path provided.", logger=logger, level=logging.INFO)
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an pretrained model checkpoint
                load_weights(model=normalized_model, dpg=self.dpg, root_folder=self.config.model.init_method.path)
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
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=self.dpg.dp_pg)

                # sync tied params across tied groups
                for (_, group_ranks), param in sorted(
                    get_tied_id_to_param(
                        parameters=model.parameters(),
                        root_module=normalized_model,
                    ).items(),
                    key=lambda x: x[0],
                ):
                    group = self.dpg.world_ranks_to_pg[group_ranks]
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")

        return model, checkpoint_path
