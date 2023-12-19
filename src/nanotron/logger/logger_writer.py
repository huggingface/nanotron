from dataclasses import dataclass
from typing import List, Union, Optional

from nanotron import logging
from nanotron.logging import log_rank, human_format
from nanotron.logger.interface import BaseLogger

logger = logging.get_logger(__name__)

@dataclass
class LogItem:
    tag: str
    scalar_value: Union[float, int]
    log_format: Optional[str] = None


@dataclass
class LoggerWriter(BaseLogger):
    global_step: int

    def add_scalar(self, tag: str, scalar_value: Union[float, int], log_format=None) -> str:
        if log_format == "human_format":
            log_str = f"{tag}: {human_format(scalar_value)}"
        else:
            log_str = f"{tag}: {scalar_value:{log_format}}" if log_format is not None else f"{tag}: {scalar_value}"
        return log_str

    def add_scalars_from_list(self, log_entries: List[LogItem], iteration_step: int):
        log_strs = [f"iteration: {iteration_step} / {self.global_step}"]
        log_strs += [
            self.add_scalar(log_item.tag, log_item.scalar_value, log_item.log_format) for log_item in log_entries
        ]
        log_str = " | ".join(log_strs)
        log_rank(log_str, logger=logger, level=logging.INFO)
