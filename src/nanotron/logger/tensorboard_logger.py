from typing import List

from tensorboardX import SummaryWriter

from brrr.logger import LogItem
from brrr.logger.interface import BaseLogger


class BatchSummaryWriter(BaseLogger, SummaryWriter):
    def add_scalars_from_list(self, log_entries: List[LogItem], iteration_step: int):
        for log_item in log_entries:
            super().add_scalar(log_item.tag, log_item.scalar_value, iteration_step)
