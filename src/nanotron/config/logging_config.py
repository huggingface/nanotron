from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

@dataclass
class HubLoggerConfig:
    """Arguments related to the HF Tensorboard logger"""

    tensorboard_dir: Path
    repo_id: str
    push_to_hub_interval: int
    repo_public: bool = False

    def __post_init__(self):
        if isinstance(self.tensorboard_dir, str):
            self.tensorboard_dir = Path(self.tensorboard_dir)


@dataclass
class TensorboardLoggerConfig:
    """Arguments related to the local Tensorboard logger"""

    tensorboard_dir: Path
    flush_secs: int = 30

    def __post_init__(self):
        if isinstance(self.tensorboard_dir, str):
            self.tensorboard_dir = Path(self.tensorboard_dir)


@dataclass
class WandbLoggerConfig(TensorboardLoggerConfig):
    """Arguments related to the local Wandb logger"""

    wandb_project: str = ""
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.wandb_project != "", "Please specify a wandb_project"


@dataclass
class LoggingArgs:
    """Arguments related to logging"""

    log_level: str
    log_level_replica: str
    iteration_step_info_interval: int
    tensorboard_logger: Optional[Union[HubLoggerConfig, TensorboardLoggerConfig, WandbLoggerConfig]]

    def __post_init__(self):
        if self.log_level not in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "passive",
        ]:
            raise ValueError(
                f"log_level should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level}"
            )
        if self.log_level_replica not in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "passive",
        ]:
            raise ValueError(
                f"log_level_replica should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level_replica}"
            )


