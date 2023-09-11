from brrr.core.import_utils import _can_import_from_module, _is_package_available
from brrr.logger.dataclass import LogItem
from brrr.logger.logger_writer import LoggerWriter

__all__ = ["LogItem", "LoggerWriter"]

tensorboardx_available = _is_package_available("tensorboardX")

if tensorboardx_available:
    from tensorboardX import SummaryWriter

    from brrr.logger.tensorboard_logger import BatchSummaryWriter

    __all__ = __all__ + ["BatchSummaryWriter", "SummaryWriter"]

huggingface_hub_available = _is_package_available("huggingface_hub")
hf_tensorboard_logger_available = _can_import_from_module("huggingface_hub", "HFSummaryWriter")

if huggingface_hub_available and hf_tensorboard_logger_available:
    from brrr.logger.hub_tensorboard_logger import HubSummaryWriter

    __all__ = __all__ + ["HubSummaryWriter"]
