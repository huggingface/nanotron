from abc import ABCMeta, abstractmethod
from typing import Iterable, Iterator, Optional, Tuple

from torch import nn
from transformers import AutoConfig

from nanotron.core import logging
from nanotron.core.distributed import ProcessGroup
from nanotron.core.logging import log_rank
from nanotron.core.parallelism.parameters import NanotronParameter
from nanotron.core.parallelism.pipeline_parallelism.block import PipelineBlock
from nanotron.core.process_groups_initializer import DistributedProcessGroups

logger = logging.get_logger(__name__)


class NanotronModel(nn.Module, metaclass=ABCMeta):
    """Abstract class for Nanotron models
    We make the following assumptions:
    - When building PP blocks, we assume that the modules order are in the same order as the forward pass."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dpg: DistributedProcessGroups
        self.config: AutoConfig

        # Attributes defined when building the model
        self.input_pp_rank: int
        self.output_pp_rank: int

        # Useful mapping to get param names
        self.module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in self.named_modules()}
        self.module_id_to_prefix[id(self)] = ""

    @abstractmethod
    def init_model_randomly(self, init_method, scaled_init_method):
        ...

    def log_modules(self, level: int = logging.DEBUG, group: Optional[ProcessGroup] = None, rank: int = 0):
        assert hasattr(self, "dpg"), "`NanotronModel` needs to have a `dpg` attribute"

        for name, module in self.named_modules():
            if not isinstance(module, PipelineBlock):
                continue
            log_rank(
                f"module_name: {name} | PP: {module.rank}/{self.dpg.pp_pg.size()}",
                logger=logger,
                level=level,
                group=group,
                rank=rank,
            )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, NanotronParameter]]:
        return super().named_parameters(prefix, recurse, remove_duplicate)

    def get_named_params_with_tied(self) -> Iterable[Tuple[str, NanotronParameter]]:
        named_parameters = [
            (
                param.get_tied_info().get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=self.module_id_to_prefix
                )
                if param.is_tied
                else name,
                param,
            )
            for name, param in self.named_parameters()
        ]
        return named_parameters
