from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from nanotron.config import (
    CheckpointsArgs,
    DataArgs,
    GeneralArgs,
    LoggingArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    ProfilerArgs,
    TokenizerArgs,
    TokensArgs,
    get_config_from_file,
)
from nanotron.config.utils_config import serialize


@dataclass
class DoReMiArgs:
    domain_weights: Optional[str] = None
    domain_names: Optional[str] = None

    # NOTE: the path where you wan to save the reference model checkpoint
    ref_model_checkpoint_path: Optional[Path] = None

    # NOTE: the path where you want to load the
    # reference model checkpoint for proxy training
    ref_model_resume_checkpoint_path: Optional[Path] = None

    def __post_init__(self):
        self.domain_names = [str(name.strip()) for name in self.domain_names.split(",")]

        if self.domain_weights is not None:
            domain_weights = [weight.strip() for weight in self.domain_weights.split(",")]
            assert sum(domain_weights) == 1.0, "Domain weights must sum to 1.0."
            self.domain_weights = torch.tensor(domain_weights)

        if self.ref_model_checkpoint_path is not None:
            self.ref_model_checkpoint_path = Path(self.ref_model_checkpoint_path)

        if self.ref_model_resume_checkpoint_path is not None:
            self.ref_model_resume_checkpoint_path = Path(self.ref_model_resume_checkpoint_path)


@dataclass
class DoReMiConfig:
    """Main configuration class"""

    general: GeneralArgs
    checkpoints: CheckpointsArgs
    parallelism: ParallelismArgs
    model: ModelArgs
    tokenizer: TokenizerArgs
    logging: LoggingArgs
    tokens: TokensArgs
    optimizer: OptimizerArgs
    data: DataArgs
    # TODO(xrsrke): remove unsupported options
    profiler: Optional[ProfilerArgs]
    doremi: DoReMiArgs

    def __post_init__(self):
        if self.profiler is not None and self.profiler.profiler_export_path is not None:
            assert self.tokens.train_steps < 10

        if self.optimizer.learning_rate_scheduler.lr_decay_steps is None:
            self.optimizer.learning_rate_scheduler.lr_decay_steps = (
                self.tokens.train_steps - self.optimizer.learning_rate_scheduler.lr_warmup_steps
            )

    @property
    def global_batch_size(self):
        return self.tokens.micro_batch_size * self.tokens.batch_accumulation_per_replica * self.parallelism.dp

    def save_as_yaml(self, file_path: str):
        config_dict = serialize(self)
        file_path = str(file_path)
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f)

        # Sanity test config can be reloaded
        _ = get_config_from_file(file_path, config_class=self.__class__)

    def as_dict(self) -> dict:
        return serialize(self)
