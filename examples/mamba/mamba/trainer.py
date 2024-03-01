from typing import Type, Union

from nanotron import logging
from nanotron.config import Config, get_config_from_file
from nanotron.trainer import DistributedTrainer

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.get_logger(__name__)


class MambaTrainer(DistributedTrainer):
    def __init__(
        self,
        config_or_config_file: Union[Config, str],
        config_class: Type[Config] = Config,
    ):
        get_config_from_file(config_or_config_file, config_class=config_class)
        super().__init__(config_or_config_file, config_class)
