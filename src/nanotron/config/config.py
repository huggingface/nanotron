import datetime
import glob
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Type, Union

import dacite
import torch
import yaml
from dacite import from_dict
from datasets.download.streaming_download_manager import xPath
from transformers import AutoTokenizer
from yaml.loader import SafeLoader

from nanotron.config.lighteval_config import LightEvalConfig
from nanotron.config.models_config import ExistingCheckpointInit, NanotronConfigs, RandomInit, SpectralMupInit
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.config.utils_config import (
    InitScalingMethod,
    RecomputeGranularity,
    cast_str_to_pipeline_engine,
    cast_str_to_torch_dtype,
    serialize,
)
from nanotron.generation.sampler import SamplerType
from nanotron.logging import get_logger, human_format
from nanotron.parallel.pipeline_parallel.engine import PipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.config.models_config import Qwen2Config

logger = get_logger(__name__)

DEFAULT_SEED = 42


@dataclass
class BenchArgs:
    model_name: str
    sequence_length: int
    micro_batch_size: int
    batch_accumulation_per_replica: int
    benchmark_csv_path: str


@dataclass
class LoggingArgs:
    """Arguments related to logging"""

    log_level: Optional[str] = None
    log_level_replica: Optional[str] = None
    iteration_step_info_interval: Optional[int] = 1

    def __post_init__(self):
        if self.log_level is None:
            self.log_level = "info"
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
        if self.log_level_replica is None:
            self.log_level_replica = "info"
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


@dataclass
class MetricsLoggingArgs:
    """Arguments related to metrics logging and tracking"""

    log_level: int = 0
    log_detail_interval: int = 10

    def __post_init__(self):
        if self.log_level not in [0, 1]:
            raise ValueError(f"metrics_level should be either 0 (basic) or 1 (full) and not {self.level}")
        if self.log_detail_interval <= 0:
            raise ValueError(f"metrics_interval should be a positive integer and not {self.interval}")


@dataclass
class PretrainDatasetsArgs:
    hf_dataset_or_datasets: Union[str, list, dict]
    hf_dataset_splits: Optional[Union[str, list]] = None
    hf_dataset_config_name: Optional[str] = None
    dataset_processing_num_proc_per_process: Optional[int] = 1
    dataset_overwrite_cache: Optional[bool] = False
    text_column_name: Optional[str] = None

    def __post_init__(self):
        if self.text_column_name is None:
            self.text_column_name = "text"
        if self.hf_dataset_splits is None:
            self.hf_dataset_splits = "train"


@dataclass
class SFTDatasetsArgs:
    # TODO @nouamane: which config do we want for SFT?
    hf_dataset_or_datasets: Union[str, list, dict]
    hf_dataset_splits: Optional[Union[str, list]] = None
    hf_dataset_config_name: Optional[str] = None
    dataset_processing_num_proc_per_process: Optional[int] = 1
    dataset_overwrite_cache: Optional[bool] = False
    sft_dataloader: Optional[bool] = True
    debug_max_samples: Optional[int] = None

    def __post_init__(self):
        if self.hf_dataset_splits is None:
            self.hf_dataset_splits = "train"


@dataclass
class S3UploadArgs:
    """Arguments related to uploading checkpoints on s3"""

    upload_s3_path: xPath
    remove_after_upload: bool
    s5cmd_numworkers: Optional[int]
    s5cmd_concurrency: Optional[int]
    s5cmd_path: Optional[xPath]

    def __post_init__(self):
        if isinstance(self.upload_s3_path, str):
            self.upload_s3_path = xPath(self.upload_s3_path)
        if isinstance(self.s5cmd_path, str):
            self.s5cmd_path = xPath(self.s5cmd_path)


@dataclass
class NanosetDatasetsArgs:
    dataset_folder: Union[str, List[str]]
    dataset_weights: Optional[List[float]] = None
    dataset_read_path: Optional[
        Union[str, List[str]]
    ] = None  # Path to local file/copy to read from. If it exists, we read from this folder instead of from dataset_folder. Useful when we offload some data to remote and only keep the needed files on disk.
    # Tokenizer config, assuming all datasets use the same tokenizer
    tokenizer_name: Optional[str] = None
    vocab_size: Optional[int] = None
    token_size_in_bytes: Optional[int] = None
    return_positions: Optional[
        bool
    ] = True  # read positions stored in disk by datatrove if eos_token_id is None, else computed on the fly

    # Tokenized bytes dataset config
    skip_in_stream: Optional[bool] = False
    pad_samples_to_global_batch_size: Optional[bool] = False
    dataset_max_tokens: Optional[List[int]] = None
    shuffle_files: Optional[bool] = False
    use_old_brrr_dataloader: Optional[bool] = False

    def __post_init__(self):
        if isinstance(self.dataset_folder, str):  # Case 1: 1 Dataset folder
            self.dataset_folder = [self.dataset_folder]
            self.dataset_weights = [1]

        # Check if dataset_weights is provided and matches the number of dataset folders
        if self.dataset_weights is not None and len(self.dataset_weights) != len(self.dataset_folder):
            raise ValueError(
                f"Number of dataset weights ({len(self.dataset_weights)}) does not match number of dataset folders ({len(self.dataset_folder)})"
            )

        # Read the first metadata file in the dataset folder to extract tokenizer name and token size.
        for folder in self.dataset_folder:
            # Find all metadata files in the folder
            metadata_files = glob.glob(os.path.join(folder, "*.metadata"))
            if metadata_files:
                # Read the first line of the first metadata file
                with open(metadata_files[0], "r") as f:
                    first_line = f.readline().strip()
                    if "|" in first_line:
                        tokenizer_name, token_size_in_bytes = first_line.split("|")
                        if self.tokenizer_name is None:
                            self.tokenizer_name = tokenizer_name
                            self.token_size_in_bytes = int(token_size_in_bytes)
                            self.vocab_size = len(AutoTokenizer.from_pretrained(tokenizer_name).get_vocab())
                        else:
                            assert (
                                self.tokenizer_name == tokenizer_name
                            ), f"Tokenizer name mismatch while reading datasets metadata file, found both {self.tokenizer_name} and {tokenizer_name}"
                            assert self.token_size_in_bytes == int(
                                token_size_in_bytes
                            ), f"Token size mismatch while reading datasets metadata file, found both {self.token_size_in_bytes} and {token_size_in_bytes}"

        # Check if dataset_read_path is provided and matches the number of dataset folders
        if self.dataset_read_path is not None and len(self.dataset_read_path) != len(self.dataset_folder):
            raise ValueError(
                f"Number of dataset read paths ({len(self.dataset_read_path)}) does not match number of dataset folders ({len(self.dataset_folder)})"
            )


@dataclass
class DataArgs:
    """Arguments related to the data and data files processing"""

    dataset: Optional[
        Union[PretrainDatasetsArgs, NanosetDatasetsArgs, SFTDatasetsArgs]
    ]  # If None we use dummy_infinite_data_generator
    seed: Optional[int]
    num_loading_workers: Optional[int] = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = DEFAULT_SEED


@dataclass
class DatasetStageArgs:
    """Arguments for loading dataset in different stages of the training process"""

    name: str
    start_training_step: int
    data: DataArgs
    sequence_length: Optional[int] = None # if None, we use the sequence length from the config

    def __post_init__(self):
        if self.start_training_step < 0:
            raise ValueError(f"training_steps should be a positive integer and not {self.start_training_step}")


@dataclass
class CheckpointsArgs:
    """Arguments related to checkpoints:
    checkpoints_path: where to save the checkpoints
    checkpoint_interval: how often to save the checkpoints
    resume_checkpoint_path: if you want to load from a specific checkpoint path
    """

    checkpoints_path: Path
    checkpoint_interval: int
    save_initial_state: Optional[bool] = False
    save_final_state: Optional[bool] = True
    resume_checkpoint_path: Optional[xPath] = None
    load_lr_scheduler: Optional[bool] = True
    load_optimizer: Optional[bool] = True
    checkpoints_path_is_shared_file_system: Optional[bool] = False

    def __post_init__(self):
        if isinstance(self.checkpoints_path, str):
            self.checkpoints_path = xPath(self.checkpoints_path)
        if isinstance(self.resume_checkpoint_path, str):
            self.resume_checkpoint_path = xPath(self.resume_checkpoint_path)


@dataclass
class GeneralArgs:
    """General training experiment arguments

    Args:
        project: Name of the project (a project gather several runs in common tensorboard/hub-folders)
        run: Name of the run
        step: Global step (updated when we save the checkpoint)
        consumed_train_samples: Number of samples consumed during training (should be actually just step*batch_size)
        ignore_sanity_checks: Whether to ignore sanity checks
    """

    project: str
    run: Optional[str] = None
    seed: Optional[int] = None
    step: Optional[int] = None
    consumed_train_samples: Optional[int] = None # TODO: remove this
    benchmark_csv_path: Optional[Path] = None
    ignore_sanity_checks: bool = True

    def __post_init__(self):
        if self.seed is None:
            self.seed = DEFAULT_SEED
        if self.run is None:
            self.run = "%date_%jobid"
        self.run = self.run.replace("%date", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run = self.run.replace("%jobid", os.environ.get("SLURM_JOB_ID", "local"))


@dataclass
class ProfilerArgs:
    """Arguments related to profiling"""

    profiler_export_path: Optional[Path]  # e.g. ./tb_logs
    wait: int = 1
    warmup: int = 1
    active: int = 1
    repeat: int = 1
    skip_first: int = 3
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = True
    export_chrome_trace: bool = False

@dataclass
class ModelArgs:
    """Arguments related to model architecture"""

    model_config: NanotronConfigs
    init_method: Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]
    dtype: Optional[torch.dtype] = None
    make_vocab_size_divisible_by: int = 1
    ddp_bucket_cap_mb: int = 25

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16
        if isinstance(self.dtype, str):
            self.dtype = cast_str_to_torch_dtype(self.dtype)

        if isinstance(self.model_config, dict):
            self.model_config = Qwen2Config(**self.model_config)

        self.model_config._is_using_mup = isinstance(self.init_method, SpectralMupInit)

        # if self.model_config.max_position_embeddings is None:
        #     self.model_config.max_position_embeddings = 0


@dataclass
class TokenizerArgs:
    """Arguments related to the tokenizer"""

    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tokenizer_max_length: Optional[int] = None


@dataclass
class TokensArgs:
    """Arguments related to the tokens, sequence, batch and steps of the training"""

    sequence_length: int
    train_steps: int
    micro_batch_size: int
    batch_accumulation_per_replica: int

    val_check_interval: Optional[int] = -1
    limit_val_batches: Optional[int] = 0
    limit_test_batches: Optional[int] = 0


@dataclass
class LRSchedulerArgs:
    """Arguments related to the learning rate scheduler

    lr_warmup_steps: number of steps to warmup the learning rate
    lr_warmup_style: linear or constant
    lr_decay_style: linear, cosine or 1-sqrt
    min_decay_lr: minimum learning rate after decay
    lr_decay_steps: optional number of steps to decay the learning rate otherwise will default to train_steps - lr_warmup_steps
    lr_decay_starting_step: optional number of steps to decay the learning rate otherwise will default to lr_warmup_steps
    """

    learning_rate: float
    lr_warmup_steps: int = 0
    lr_warmup_style: str = None
    lr_decay_style: str = None
    lr_decay_steps: Optional[int] = None
    lr_decay_starting_step: Optional[int] = None
    min_decay_lr: float = None

    def __post_init__(self):
        if self.lr_warmup_style not in ["linear", "constant"]:
            raise ValueError(
                f"lr_warmup_style should be a string selected in ['linear', 'constant'] and not {self.lr_warmup_style}"
            )
        if self.lr_warmup_style is None:
            self.lr_warmup_style = "linear"
        if self.lr_decay_style is None:
            self.lr_decay_style = "linear"
        if self.lr_decay_style not in ["linear", "cosine", "1-sqrt"]:
            raise ValueError(
                f"lr_decay_style should be a string selected in ['linear', 'cosine', '1-sqrt'] and not {self.lr_decay_style}"
            )
        if self.min_decay_lr is None:
            self.min_decay_lr = self.learning_rate


@dataclass
class SGDOptimizerArgs:
    name: str = "sgd"


@dataclass
class AdamWOptimizerArgs:
    adam_eps: float
    adam_beta1: float
    adam_beta2: float
    torch_adam_is_fused: bool
    name: str = "adamW"


@dataclass
class OptimizerArgs:
    """Arguments related to the optimizer and learning rate"""

    optimizer_factory: Union[SGDOptimizerArgs, AdamWOptimizerArgs]
    zero_stage: int
    weight_decay: float
    clip_grad: Optional[float]
    accumulate_grad_in_fp32: bool
    learning_rate_scheduler: LRSchedulerArgs
    weight_decay_exclude_named_params: Optional[
        List[str]
    ] = None  # List of regex patterns to exclude parameters from weight decay

    def __post_init__(self):
        if self.weight_decay_exclude_named_params is None:
            self.weight_decay_exclude_named_params: List[str] = []


@dataclass
class GenerationArgs:
    sampler: Optional[Union[str, SamplerType]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n_samples: Optional[int] = None
    eos: Optional[str] = None
    seed: Optional[int] = None
    use_cache: Optional[bool] = False

    def __post_init__(self):
        if isinstance(self.sampler, str):
            self.sampler = SamplerType[self.sampler.upper()]
        if self.seed is None:
            self.seed = DEFAULT_SEED


@dataclass
class Config:
    """Main configuration class"""

    general: GeneralArgs
    parallelism: ParallelismArgs
    model: ModelArgs
    tokenizer: Optional[TokenizerArgs] = None
    checkpoints: Optional[CheckpointsArgs] = None
    logging: Optional[LoggingArgs] = None
    metrics_logging: Optional[MetricsLoggingArgs] = None
    tokens: Optional[TokensArgs] = None
    optimizer: Optional[OptimizerArgs] = None
    data_stages: Optional[List[DatasetStageArgs]] = None
    profiler: Optional[ProfilerArgs] = None
    lighteval: Optional[LightEvalConfig] = None
    s3_upload: Optional[S3UploadArgs] = None

    @classmethod
    def create_empty(cls):
        cls_fields = fields(cls)
        return cls(**{f.name: None for f in cls_fields})

    def __post_init__(self):

        if self.s3_upload is not None:
            self.s3_upload.__post_init__()
            if self.lighteval is not None:
                if self.lighteval.eval_interval is None:
                    self.lighteval.eval_interval = self.checkpoints.checkpoint_interval
                else:
                    assert (
                        self.lighteval.eval_interval % self.checkpoints.checkpoint_interval == 0
                    ), f"eval_interval={self.lighteval.eval_interval} must be a multiple of checkpoint_interval={self.checkpoints.checkpoint_interval}"

        # Some final sanity checks across separate arguments sections:
        if self.profiler is not None and self.profiler.profiler_export_path is not None:
            total_profiling_steps = self.profiler.skip_first + self.profiler.repeat * (
                self.profiler.wait + self.profiler.warmup + self.profiler.active
            )
            assert (
                self.tokens.train_steps >= total_profiling_steps
            ), f"Profiling steps ({total_profiling_steps}) must be less than or equal to train steps ({self.tokens.train_steps})"

        if self.optimizer is not None and self.optimizer.learning_rate_scheduler.lr_decay_steps is None:
            self.optimizer.learning_rate_scheduler.lr_decay_steps = (
                self.tokens.train_steps - self.optimizer.learning_rate_scheduler.lr_warmup_steps
            )

        if self.data_stages is not None:
            self.data_stages = sorted(self.data_stages, key=lambda stage: stage.start_training_step)
            names = [stage.name for stage in self.data_stages]
            training_steps = [stage.start_training_step for stage in self.data_stages]
            assert any(
                stage.start_training_step == 1 for stage in self.data_stages
            ), "You must have a training stage starting at 1 in the config's data_stages"

            for stage in self.data_stages:
                if names.count(stage.name) > 1:
                    raise ValueError(f"Each stage should have unique names and not {names}")

                if training_steps.count(stage.start_training_step) > 1:
                    raise ValueError(
                        f"Each stage should have unique starting training step, please change the starting training step for stage {stage.name}"
                    )

                if isinstance(stage.data.dataset, NanosetDatasetsArgs):
                    if self.model.model_config.vocab_size == -1:
                        self.model.model_config.vocab_size = stage.data.dataset.vocab_size
                        logger.warning(
                            f"Setting model's vocab_size to {self.model.model_config.vocab_size} from dataset's vocab_size ({stage.data.dataset.vocab_size})"
                        )
                    assert (
                        self.model.model_config.vocab_size == stage.data.dataset.vocab_size
                    ), f"Model's vocab_size ({self.model.model_config.vocab_size}) does not match dataset's ({stage.data.dataset.dataset_folder}) vocab_size ({stage.data.dataset.vocab_size})"
                    if self.tokenizer is None:
                        self.tokenizer = TokenizerArgs(tokenizer_name_or_path=stage.data.dataset.tokenizer_name)
                        logger.warning(
                            f"Setting tokenizer to {self.tokenizer.tokenizer_name_or_path} from dataset's tokenizer ({stage.data.dataset.tokenizer_name})"
                        )
                    assert (
                        self.tokenizer.tokenizer_name_or_path == stage.data.dataset.tokenizer_name
                    ), f"Tokenizer passed in config ({self.tokenizer.tokenizer_name_or_path}) does not match dataset's ({stage.data.dataset.dataset_folder}) tokenizer ({stage.data.dataset.tokenizer_name})"

            # NOTE: must order the stages by start_training_step from lowest to highest
            assert all(
                self.data_stages[i].start_training_step < self.data_stages[i + 1].start_training_step
                for i in range(len(self.data_stages) - 1)
            ), "The stages are not sorted by start_training_step in increasing order"

        # # if lighteval, we need tokenizer to be defined
        # if self.checkpoints.lighteval is not None:
        #     assert self.tokenizer.tokenizer_name_or_path is not None

        # Model verifications
        assert (
            self.model.model_config.num_attention_heads % self.parallelism.tp == 0
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be divisible by tp ({self.parallelism.tp})"
        assert (
            self.model.model_config.num_attention_heads >= self.model.model_config.num_key_value_heads
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be >= num_key_value_heads ({self.model.model_config.num_key_value_heads})"
        assert (
            self.model.model_config.num_key_value_heads >= self.parallelism.tp
        ), f"num_key_value_heads ({self.model.model_config.num_key_value_heads}) must be >= tp ({self.parallelism.tp})"  # TODO: remove this once we ensure KV heads get duplicated correctly
        assert (
            self.model.model_config.num_attention_heads % self.model.model_config.num_key_value_heads == 0
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be divisible by num_key_value_heads ({self.model.model_config.num_key_value_heads})"

        # data_stages
        if self.data_stages is not None:
            for stage in self.data_stages:
                if stage.sequence_length is None:
                    stage.sequence_length = self.tokens.sequence_length

    @property
    def global_batch_size(self):
        return self.tokens.micro_batch_size * self.tokens.batch_accumulation_per_replica * self.parallelism.dp

    @property
    def global_batch_size_in_tokens(self):
        return self.global_batch_size * self.tokens.sequence_length

    def save_as_yaml(self, file_path: str, sanity_checks: bool = True):
        config_dict = serialize(self)
        file_path = str(file_path)
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f)

        # Sanity test config can be reloaded
        if sanity_checks:
            _ = get_config_from_file(file_path, config_class=self.__class__)

    def get_yaml(self):
        config_dict = serialize(self)
        return yaml.dump(config_dict)

    @classmethod
    def load_from_yaml(cls, file_path: str):
        config_dict = yaml.load(open(file_path), Loader=SafeLoader)
        return get_config_from_dict(config_dict, config_class=cls)

    def as_dict(self) -> dict:
        return serialize(self)

    def print_config_details(self):
        print("\n=== Model Architecture ===")
        print(f"hidden_size: {self.model.model_config.hidden_size}")
        print(f"num_layers: {self.model.model_config.num_hidden_layers}")
        print(f"intermediate_size: {self.model.model_config.intermediate_size}")
        print(f"num_attention_heads: {self.model.model_config.num_attention_heads}")
        print(f"num_key_value_heads: {self.model.model_config.num_key_value_heads}")
        print(f"tie_word_embeddings: {self.model.model_config.tie_word_embeddings}")
        print(f"vocab_size: {self.model.model_config.vocab_size}")
        print(f"num_params: {_calculate_model_params(self)}")

        print("\n=== Training Configuration ===")
        print(
            f"seq_len: {self.model.model_config.max_position_embeddings} | mbs: {self.tokens.micro_batch_size} | batch_accum: {self.tokens.batch_accumulation_per_replica} | gbs: {human_format(self.global_batch_size_in_tokens)} | train_steps: {self.tokens.train_steps} | total_tokens: {human_format(self.tokens.train_steps * self.global_batch_size_in_tokens)}"
        )

        print("\n=== Parallelism ===")
        print(
            f"tp: {self.parallelism.tp} | pp: {self.parallelism.pp} | dp: {self.parallelism.dp} | cp: {self.parallelism.context_parallel_size} | ep: {self.parallelism.expert_parallel_size}"
        )
        print(f"zero_stage: {self.optimizer.zero_stage} | full_checkpointing: {self.parallelism.recompute_layer}")
        print("=" * 20 + "\n")


def get_config_from_dict(
    config_dict: dict, config_class: Type = Config, skip_unused_config_keys: bool = False, skip_null_keys: bool = False
):
    """Get a config object from a dictionary

    Args:
        args: dictionary of arguments
        config_class: type of the config object to get as a ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
        skip_unused_config_keys: whether to skip unused first-nesting-level keys in the config file (for config with additional sections)
        skip_null_keys: whether to skip keys with value None at first and second nesting level
    """
    if skip_unused_config_keys:
        logger.warning("skip_unused_config_keys set")
        config_dict = {
            field.name: config_dict[field.name] for field in fields(config_class) if field.name in config_dict
        }
    if skip_null_keys:
        logger.warning("Skip_null_keys set")
        config_dict = {
            k: {kk: vv for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else v
            for k, v in config_dict.items()
            if v is not None
        }
    return from_dict(
        data_class=config_class,
        data=config_dict,
        config=dacite.Config(
            cast=[Path],
            type_hooks={
                torch.dtype: cast_str_to_torch_dtype,
                PipelineEngine: cast_str_to_pipeline_engine,
                TensorParallelLinearMode: lambda x: TensorParallelLinearMode[x.upper()],
                RecomputeGranularity: lambda x: RecomputeGranularity[x.upper()],
                InitScalingMethod: lambda x: InitScalingMethod[x.upper()],
                SamplerType: lambda x: SamplerType[x.upper()],
            },
            # strict_unions_match=True,
            strict=True,
        ),
    )


def get_config_from_file(
    config_path: str,
    config_class: Type = Config,
    model_config_class: Optional[Type] = None,
    skip_unused_config_keys: bool = False,
    skip_null_keys: bool = False,
) -> Config:
    """Get a config object from a file (python or YAML)

    Args:
        config_path: path to the config file
        config_type: if the file is a python file, type of the config object to get as a
            ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
            if None, will default to Config
        skip_unused_config_keys: whether to skip unused first-nesting-level keys in the config file (for config with additional sections)
        skip_null_keys: whether to skip keys with value None at first and second nesting level
    """
    # Open the file and load the file
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    config = get_config_from_dict(
        config_dict,
        config_class=config_class,
        skip_unused_config_keys=skip_unused_config_keys,
        skip_null_keys=skip_null_keys,
    )
    if model_config_class is not None:
        if not isinstance(config.model.model_config, (dict, model_config_class)):
            raise ValueError(
                f"model_config should be a dictionary or a {model_config_class} and not {config.model.model_config}"
            )
        config.model.model_config = model_config_class(**config.model.model_config)
    return config


def _calculate_model_params(config: Config):
    """Calculate and format the number of parameters in the model.
    N = vocab * h * 2 + num_layers * (3 * h * inter + 4 * h * h)
    """
    num_params = human_format(
        config.model.model_config.vocab_size
        * config.model.model_config.hidden_size
        * (2 if config.model.model_config.tie_word_embeddings else 1)
        + config.model.model_config.num_hidden_layers
        * (
            3 * config.model.model_config.hidden_size * config.model.model_config.intermediate_size
            + 4 * config.model.model_config.hidden_size * config.model.model_config.hidden_size
        )
    )
    return num_params
