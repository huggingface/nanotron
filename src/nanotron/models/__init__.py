from abc import ABCMeta, abstractmethod
from typing import Optional

from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel import ParallelContext
from torch import nn
from nanotron.config import NanotronConfigs

logger = logging.get_logger(__name__)


class NanotronModel(nn.Module, metaclass=ABCMeta):
    """Abstract class for Nanotron models
    We make the following assumptions:
    - When building PP blocks, we assume that the modules order are in the same order as the forward pass."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parallel_context: ParallelContext
        self.config: NanotronConfigs
        self.module_id_to_prefix: dict[int, str]

        # Attributes defined when building the model
        self.input_pp_rank: int
        self.output_pp_rank: int

    @abstractmethod
    def init_model_randomly(self, init_method, scaled_init_method):
        ...

    @staticmethod
    def get_embeddings_lm_head_tied_names() -> list[str]:
        """Returns the names of the embeddings and lm_head weights that are tied together. Returns empty list if not tied.

        Example for GPT2 model: ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        """
        return []

    def before_tbi_sanity_checks(self) -> None:
        pass

    def after_tbi_sanity_checks(self) -> None:
        pass

    def before_optim_step_sanity_checks(self) -> None:
        pass

    def after_optim_step_sanity_checks(self) -> None:
        pass

    def log_modules(self, level: int = logging.DEBUG, group: Optional[ProcessGroup] = None, rank: int = 0):
        assert hasattr(self, "parallel_context"), "`NanotronModel` needs to have a `parallel_context` attribute"

        for name, module in self.named_modules():
            if not isinstance(module, PipelineBlock):
                continue
            log_rank(
                f"module_name: {name} | PP: {module.rank}/{self.parallel_context.pp_pg.size()}",
                logger=logger,
                level=level,
                group=group,
                rank=rank,
            )
