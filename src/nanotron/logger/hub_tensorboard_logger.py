from typing import List

from huggingface_hub import HFSummaryWriter
from huggingface_hub.utils import disable_progress_bars

from nanotron.logger import LogItem
from nanotron.logger.interface import BaseLogger

disable_progress_bars()


class HubSummaryWriter(BaseLogger, HFSummaryWriter):
    def add_scalars_from_list(self, log_entries: List[LogItem], iteration_step: int):
        for log_item in log_entries:
            super().add_scalar(log_item.tag, log_item.scalar_value, iteration_step)
