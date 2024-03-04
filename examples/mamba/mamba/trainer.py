from typing import Optional, Type, Union

from torch.nn.parallel import DistributedDataParallel

from nanotron import logging
from nanotron.trainer import DistributedTrainer

from .config import MambaConfig

logger = logging.get_logger(__name__)

from nanotron import distributed as dist
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.serialize import load_weights, parse_ckpt_path

from .config import ExistingCheckpointInit, MambaInit


class MambaTrainer(DistributedTrainer):
    def __init__(
        self,
        config_or_config_file: Union[MambaConfig, str],
        config_class: Type[MambaConfig] = MambaConfig,
        model_config_class: Optional[Type] = None,
        model_class: Type[NanotronModel] = None,
    ):
        assert config_class == MambaConfig
        super().__init__(config_or_config_file, config_class, model_config_class, model_class)

    def _load_model_checkpoint(self, model: NanotronModel) -> NanotronModel:
        unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model

        # Load or initialize model weights
        self.init_checkpoint_path = parse_ckpt_path(config=self.config)
        reloaded_from_checkpoint = False
        if self.init_checkpoint_path is not None:
            # Reload from a training checkpoint
            log_rank(f"Loading weights from {self.init_checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
            self.param_shard_metadata = load_weights(
                model=unwrapped_model, parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
            )
            reloaded_from_checkpoint = True
        if not reloaded_from_checkpoint:
            log_rank("No checkpoint path provided.", logger=logger, level=logging.INFO)
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an pretrained model checkpoint
                self.param_shard_metadata = load_weights(
                    model=unwrapped_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.config.model.init_method.path,
                )
            elif isinstance(self.config.model.init_method, MambaInit):

                unwrapped_model.init_model_randomly(config=self.config)
                # Synchronize parameters so that the model is consistent
                # sync all params across dp
                for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=self.parallel_context.dp_pg)

                # sync tied params across tied groups
                for (_, group_ranks), param in sorted(
                    get_tied_id_to_param(
                        parameters=model.parameters(),
                        root_module=unwrapped_model,
                    ).items(),
                    key=lambda x: x[0],
                ):
                    group = self.parallel_context.world_ranks_to_pg[group_ranks]
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")

        return model
