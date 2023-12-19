from nanotron.logger.logger_writer import LoggerWriter, LogItem
from nanotron.constants import TENSORBOARDX_AVAILABLE, HUGGINGFACE_HUB_AVAILABLE, HF_TENSORBOARD_LOGGER_AVAILABLE

__all__ = ["LogItem", "LoggerWriter"]

if TENSORBOARDX_AVAILABLE:
    from tensorboardX import SummaryWriter

    from nanotron.logger.tensorboard_logger import BatchSummaryWriter

    __all__ = __all__ + ["BatchSummaryWriter", "SummaryWriter"]

if HUGGINGFACE_HUB_AVAILABLE and HF_TENSORBOARD_LOGGER_AVAILABLE:
    from nanotron.logger.hub_tensorboard_logger import HubSummaryWriter

    __all__ = __all__ + ["HubSummaryWriter"]
