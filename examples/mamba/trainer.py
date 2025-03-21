from typing import Optional, Type, Union

from config import ExistingCheckpointInit, MambaConfig, MambaInit
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.utils import get_pp_rank_of
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.parallel.tied_parameters import (
    create_pg_for_tied_weights,
    get_tied_id_to_param,
    tie_parameters,
)
from nanotron.serialize import load_weights, parse_ckpt_path
from nanotron.trainer import DistributedTrainer
from torch.nn.parallel import DistributedDataParallel

logger = logging.get_logger(__name__)


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

    def _mark_tied_parameters(
        self,
        model: NanotronModel,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs] = None,
    ):
        # Tie embeddings
        embeddings_lm_head_tied_names = model.get_embeddings_lm_head_tied_names()
        if len(embeddings_lm_head_tied_names) > 0:
            shared_embeddings = [
                (
                    target,
                    (
                        parallel_context.world_rank_matrix[
                            dist.get_rank(parallel_context.ep_pg),
                            get_pp_rank_of(target, module=model),
                            dist.get_rank(parallel_context.dp_pg),
                            dist.get_rank(parallel_context.tp_pg),
                        ],
                    ),
                )
                for target in embeddings_lm_head_tied_names
            ]
            tie_parameters(
                root_module=model,
                ties=shared_embeddings,
                parallel_context=parallel_context,
                reduce_op=dist.ReduceOp.SUM,
            )

        # Tie custom params
        model.tie_custom_params()

        # Sync all parameters that have the same name and that are not sharded
        assert not isinstance(model, DistributedDataParallel), "model shouldn't be DDP at this point"
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                name = f"{module_name}.{param_name}"

                if isinstance(param, NanotronParameter) and (param.is_sharded or param.is_tied):
                    continue

                if isinstance(module, TensorParallelRowLinear) and "bias" == param_name:
                    # bias for TensorParallelRowLinear only exists on TP=0 so we don't need to tie it
                    continue

                shared_weights = [
                    (
                        name,
                        # sync across TP group
                        tuple(sorted(dist.get_process_group_ranks(parallel_context.tp_pg))),
                    )
                ]

                if (
                    parallel_config is None
                    or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE
                    or hasattr(model.config.model.model_config, "is_mamba_config")
                ):
                    # We add `reduce_op=None` in order to signal that the weight are synced by design without needing to reduce
                    # when TP=2 we have LN that is duplicated across TP, so by design it's tied
                    reduce_op = None
                else:
                    reduce_op = dist.ReduceOp.SUM

                tie_parameters(
                    root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=reduce_op
                )

        create_pg_for_tied_weights(root_module=model, parallel_context=parallel_context)

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
