import dataclasses
import datetime
import importlib
import os
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TypeVar, Union

import dacite
import torch
import yaml
from dacite import from_dict
from datasets.download.streaming_download_manager import xPath
from datasets.load import DownloadMode, dataset_module_factory
from transformers import AutoConfig
from yaml.loader import SafeLoader

from nanotron.core.logging import get_logger
from nanotron.core.parallelism.pipeline_parallelism.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.core.parallelism.tensor_parallelism.nn import TensorParallelLinearMode
from nanotron.generate.sampler import SamplerType

logger = get_logger(__name__)


class RecomputeGranularity(Enum):
    SELECTIVE = auto()
    FULL = auto()


@dataclass
class LlamaConfig:
    """Configuration for a LLAMA model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_llama_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000

    def __post_init__(self):
        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class FalconConfig:
    """Configuration for a Falcon model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    alibi: bool = False
    attention_dropout: float = 0.0
    bias: bool = False
    bos_token_id: int = 11
    eos_token_id: int = 11
    hidden_dropout: float = 0.0
    hidden_size: int = 4544
    initializer_range: float = 0.02
    is_falcon_config: bool = True  # We use this help differentiate models in yaml/python conversion
    layer_norm_epsilon: float = 1e-5
    multi_query: bool = True
    new_decoder_architecture: bool = False
    num_attention_heads: int = 71
    num_hidden_layers: int = 32
    num_kv_heads: Optional[int] = None
    pad_token_id: int = 11
    parallel_attn: bool = True
    use_cache: bool = True
    vocab_size: int = 65024

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary(self):
        return not self.alibi


@dataclass
class GPTBigCodeConfig:
    """Configuration for a GPTBigCode model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    is_gpt_bigcode_config: bool = False  # We use this help differentiate models in yaml/python conversion
    activation_function: str = "gelu_pytorch_tanh"
    attention_softmax_in_fp32: bool = True  # TODO: not used
    attn_pdrop: float = 0.1
    bos_token_id: int = 49152  # TODO: not used
    embd_pdrop: float = 0.1
    eos_token_id: int = 49152
    initializer_range: float = 0.02  # TODO: not used
    layer_norm_epsilon: float = 1e-05
    multi_query: bool = True
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 4096
    resid_pdrop: float = 0.1
    scale_attention_softmax_in_fp32: bool = True
    scale_attn_weights: bool = True
    vocab_size: int = 49280

    @property
    def n_embed(self):
        return self.hidden_size

    @property
    def n_head(self):
        return self.num_attention_heads

    @property
    def n_layer(self):
        return self.num_hidden_layers

    @property
    def n_positions(self):
        return self.max_position_embeddings

    @property
    def n_inner(self):
        return self.intermediate_size

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size


@dataclass
class Starcoder2Config(GPTBigCodeConfig):
    """Configuration for a Starcoder2 model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    is_starcoder2_config: bool = True  # We use this help differentiate models in yaml/python conversion
    sliding_window_size: Optional[int] = None
    use_rotary_embeddings: bool = True
    rope_theta: Optional[int] = 10000
    use_position_embeddings: bool = False  # TODO @nouamane this is not used
    global_attn_layers: List[int] = dataclasses.field(default_factory=list)

    # MQA
    multi_query: bool = False
    # GQA
    grouped_query: bool = False
    num_kv_heads: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        delattr(self, "is_gpt_bigcode_config")
        if self.global_attn_layers is None:
            self.global_attn_layers = []

        if self.grouped_query:
            assert self.num_kv_heads is not None, "num_kv_heads must be specified for grouped query"
            assert self.multi_query is False, "Cannot use both multi_query and grouped_query"

        if not self.multi_query and not self.grouped_query:
            self.multi_query = True


NanotronConfigs = Union[FalconConfig, LlamaConfig, GPTBigCodeConfig, Starcoder2Config]


@dataclass
class GeneralArgs:
    """General training experiment arguments

    Args:
        project: Name of the project (a project gather several runs in common tensorboard/hub-folders)
        run: Name of the run
        step: Global step (updated when we save the checkpoint)
        consumed_train_samples: Number of samples consumed during training (should be actually just step*batch_size)
        kill_switch_path: Path to the kill switch file
        ignore_sanity_checks: Whether to ignore sanity checks
    """

    project: str
    run: Optional[str] = None
    step: Optional[int] = None
    consumed_train_samples: Optional[int] = None
    # If you want to signal the training script to stop, you just need to touch the following file
    # We force users to set one in order to programmatically be able to remove it.
    kill_switch_path: Optional[xPath] = None
    benchmark_csv_path: Optional[xPath] = None
    ignore_sanity_checks: bool = False

    def __post_init__(self):
        if isinstance(self.kill_switch_path, str):
            self.kill_switch_path = xPath(self.kill_switch_path)
        if self.benchmark_csv_path is not None:
            assert (
                os.environ.get("BRRR_BENCHMARK", None) is not None
            ), f"Please set BRRR_BENCHMARK to 1 when using benchmark_csv_path. Got {os.environ.get('BRRR_BENCHMARK', None)}"

        if self.run is None:
            self.run = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if os.environ.get("SLURM_JOB_ID", None) is not None:
                self.run += f"_{os.environ['SLURM_JOB_ID']}"
        else:
            self.run = self.run.replace("%d", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            if os.environ.get("SLURM_JOB_ID", None) is not None:
                self.run = self.run.replace("%j", os.environ["SLURM_JOB_ID"])


@dataclass
class ProfileArgs:
    """Arguments related to profiling"""

    profiler_export_path: Optional[xPath]


@dataclass
class UploadCheckpointOnS3Args:
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
class CheckpointsArgs:
    """Arguments related to checkpoints:
    checkpoints_path: where to save the checkpoints
    checkpoint_interval: how often to save the checkpoints
    resume_checkpoint_path: if you want to load from a specific checkpoint path
    s3: if you want to upload the checkpoints on s3

    """

    checkpoints_path: xPath
    checkpoint_interval: int
    resume_checkpoint_path: Optional[xPath] = None
    checkpoints_path_is_shared_file_system: Optional[bool] = True
    save_initial_state: Optional[bool] = False
    s3: Optional[UploadCheckpointOnS3Args] = None
    lighteval: Optional["LightEvalConfig"] = None

    def __post_init__(self):
        if isinstance(self.checkpoints_path, str):
            self.checkpoints_path = xPath(self.checkpoints_path)
        if isinstance(self.resume_checkpoint_path, str):
            self.resume_checkpoint_path = xPath(self.resume_checkpoint_path)


@dataclass
class _ParallelismParentArgs:
    """Hidden parent class for ParallelismArgs and PPOParallelismArgs.
    To contain common non-optional fields.
    """

    dp: int
    pp: int
    tp: int


@dataclass
class ParallelismArgs(_ParallelismParentArgs):
    """Arguments related to TP/PP/DP

    Args:
        dp: Number of DP replicas
        pp: Number of PP stages
        tp: Number of TP replicas
        pp_engine: Pipeline engine to use between "1f1b" and "afab"
        tp_mode: TP mode to use between "all_reduce" and "reduce_scatter": all_reduce is normal, reduce_scatter activate sequence parallelism
        recompute_granularity: Recompute granularity to use between "full" and "selective"
        tp_linear_async_communication: Whether to use async communication in TP linear layers
    """

    pp_engine: Optional[PipelineEngine] = None
    tp_mode: Optional[TensorParallelLinearMode] = None
    recompute_granularity: Optional[RecomputeGranularity] = None
    tp_linear_async_communication: Optional[bool] = None

    def __post_init__(self):
        # Conservative defaults
        if self.pp_engine is None:
            self.pp_engine = AllForwardAllBackwardPipelineEngine()
        if self.tp_mode is None:
            self.tp_mode = TensorParallelLinearMode.ALL_REDUCE
        if self.tp_linear_async_communication is None:
            self.tp_linear_async_communication = False

        if isinstance(self.pp_engine, str):
            self.pp_engine = cast_str_to_pipeline_engine(self.pp_engine)
        if isinstance(self.tp_mode, str):
            self.tp_mode = TensorParallelLinearMode[self.tp_mode.upper()]
        if isinstance(self.recompute_granularity, str):
            self.recompute_granularity = RecomputeGranularity[self.recompute_granularity.upper()]


@dataclass
class PPOParallelismArgs(_ParallelismParentArgs):
    """Arguments related to TP/PP/DP"""

    ref_model_pp_size: int  # For PPO, number of PP stages in the reference model
    pp_engine: Optional[PipelineEngine] = None
    tp_mode: Optional[TensorParallelLinearMode] = None
    recompute_granularity: Optional[RecomputeGranularity] = None
    tp_linear_async_communication: Optional[bool] = None

    def __post_init__(self):
        # Conservative defaults
        if self.pp_engine is None:
            self.pp_engine = AllForwardAllBackwardPipelineEngine()
        if self.tp_mode is None:
            self.tp_mode = TensorParallelLinearMode.ALL_REDUCE
        if self.tp_linear_async_communication is None:
            self.tp_linear_async_communication = False

        if isinstance(self.pp_engine, str):
            self.pp_engine = cast_str_to_pipeline_engine(self.pp_engine)
        if isinstance(self.tp_mode, str):
            self.tp_mode = TensorParallelLinearMode[self.tp_mode.upper()]
        if isinstance(self.recompute_granularity, str):
            self.recompute_granularity = RecomputeGranularity[self.recompute_granularity.upper()]


@dataclass
class RandomInit:
    std: float


@dataclass
class ExistingCheckpointInit:
    """This is used to initialize from an already existing model (without optimizer, lr_scheduler...)"""

    path: xPath

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = xPath(self.path)


@dataclass
class RemoteCodeArgs:
    """Arguments related to model architecture"""

    trust_remote_code: bool


@dataclass
class ModelArgs:
    """Arguments related to model architecture"""

    make_vocab_size_divisible_by: int
    dtype: torch.dtype
    init_method: Union[RandomInit, ExistingCheckpointInit]
    seed: Optional[int]
    remote_code: Optional[RemoteCodeArgs] = None
    hf_model_name: Optional[str] = None
    model_config: Optional[NanotronConfigs] = None
    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    ddp_bucket_cap_mb: int = 25

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = cast_str_to_torch_dtype(self.dtype)
        if self.remote_code is None:
            self.remote_code = RemoteCodeArgs(trust_remote_code=False)

        if self.model_config is not None:
            if self.model_config.max_position_embeddings is None:
                self.model_config.max_position_embeddings = 0
        else:
            assert (
                self.hf_model_name is not None
            ), "You need to specify either a model_config or a hf_model_name under model"


@dataclass
class HubLoggerConfig:
    """Arguments related to the HF Tensorboard logger"""

    tensorboard_dir: xPath
    repo_id: str
    push_to_hub_interval: int
    repo_public: bool = False

    def __post_init__(self):
        if isinstance(self.tensorboard_dir, str):
            self.tensorboard_dir = xPath(self.tensorboard_dir)


@dataclass
class TensorboardLoggerConfig:
    """Arguments related to the local Tensorboard logger"""

    tensorboard_dir: xPath
    flush_secs: int = 30

    def __post_init__(self):
        if isinstance(self.tensorboard_dir, str):
            self.tensorboard_dir = xPath(self.tensorboard_dir)


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


@dataclass
class TokensArgs:
    """Arguments related to the tokens, sequence, batch and steps of the training"""

    sequence_length: int
    train_steps: int
    micro_batch_size: int
    batch_accumulation_per_replica: int

    val_check_interval: Optional[int]
    limit_val_batches: Optional[int]
    limit_test_batches: Optional[int] = 0


@dataclass
class OptimizerArgs:
    """Arguments related to the optimizer and learning rate"""

    zero_stage: int
    weight_decay: float
    clip_grad: Optional[float]

    accumulate_grad_in_fp32: bool

    adam_eps: float
    adam_beta1: float
    adam_beta2: float
    torch_adam_is_fused: bool

    learning_rate: float


@dataclass
class LRSchedulerArgs:
    """Arguments related to the learning rate scheduler

    lr_warmup_steps: number of steps to warmup the learning rate
    lr_warmup_style: linear or constant
    lr_decay_style: linear or cosine
    min_decay_lr: minimum learning rate after decay
    lr_decay_steps: optional number of steps to decay the learning rate otherwise will default to train_steps - lr_warmup_steps
    """

    lr_warmup_steps: int
    lr_warmup_style: str
    lr_decay_style: str
    min_decay_lr: float
    lr_decay_steps: Optional[int] = None

    def __post_init__(self):
        if self.lr_warmup_style not in ["linear", "constant"]:
            raise ValueError(
                f"lr_warmup_style should be a string selected in ['linear', 'constant'] and not {self.lr_warmup_style}"
            )
        if self.lr_decay_style not in ["linear", "cosine"]:
            raise ValueError(
                f"lr_decay_style should be a string selected in ['linear', 'cosine'] and not {self.lr_decay_style}"
            )


@dataclass
class PretrainNemoArgs:
    data_prefix: Union[list, dict]
    index_mapping_dir: Optional[
        str
    ]  # path to save index mapping .npy files, by default will save in the same location as data_prefix
    splits_string: str
    skip_warmup: bool
    dataloader_type: str
    validation_drop_last: bool  # Set to false if the last partial validation samples is to be consumed
    eod_mask_loss: bool  # Mask loss for the end of document tokens
    no_seqlen_plus_one_input_tokens: bool  # Set to True to disable fetching (sequence length + 1) input tokens, instead get (sequence length) input tokens and mask the last token
    pad_samples_to_global_batch_size: bool  # Set to True if you want to pad the last partial batch with -1's to equal global batch size
    fim_rate: Optional[float] = 0.0  # Rate of fill-in-the-middle (FIM)
    fim_spm_rate: Optional[float] = 0.5  # Rate of suffix-prefix-middle (SPM) option in the fill-in-the-middle format
    fim_split_sample: Optional[
        str
    ] = None  # String around which to split the sample for FIM. If None (default), FIM is applied on the sample-level
    fragment_fim_rate: Optional[float] = 0.5  # Rate of FIM on each fragment when fim_split_sample is not None.
    no_fim_prefix: Optional[str] = None  # Do not apply FIM to fragments that start with this prefix

    def __post_init__(self):
        # TODO @thomasw21: Should probably be an enum
        if self.dataloader_type not in ["single", "cyclic"]:
            raise ValueError(
                f"dataloader_type should be a string selected in ['single', 'cyclic'] and not {self.dataloader_type}"
            )

        if self.eod_mask_loss:
            raise NotImplementedError("`eod_mask_loss` support is not implemented yet")


@dataclass
class PretrainDatasetsArgs:
    hf_dataset_mixer: Union[str, list, dict]
    hf_dataset_config_name: Optional[str]
    hf_dataset_splits: Union[str, list]
    dataset_processing_num_proc_per_process: int
    dataset_overwrite_cache: Optional[bool]
    text_column_name: Optional[str]


@dataclass
class TokenizedBytesDatasetFileArgs:
    filepath: str


@dataclass
class TokenizedBytesDatasetFolderArgs:
    folder: str
    filename_pattern: str


@dataclass
class TokenizedBytesDatasetArgs:
    datasets: List[Union[TokenizedBytesDatasetFileArgs, TokenizedBytesDatasetFolderArgs]]
    dataloader_type: str  # single or cycle
    pad_samples_to_global_batch_size: bool  # Set to True if you want to pad the last partial batch with -1's to equal global batch size
    dataset_weights: Optional[List[float]] = None
    dataset_max_tokens: Optional[
        List[float]
    ] = None  # Optional max_tokens per dataset (divide by seq len to get the number of tokens)
    skip_in_stream: Optional[bool] = True


@dataclass
class DPOPretrainDatasetsArgs:
    hf_dataset_name: str
    hf_dataset_split: str

    # TODO @nouamane: this gives unclear error: https://github.com/huggingface/brrr/issues/515
    # def __post_init__(self):
    #     if self.hf_dataset_name not in ["Anthropic/hh-rlhf"]:
    #         raise ValueError(
    #             f"hf_dataset_name={self.hf_dataset_name} is not supported yet. Only Anthropic/hh-rlhf is supported."
    #         )


@dataclass
class DataArgs:
    """Arguments related to the data and data files processing"""

    # TODO @thomasw21: Would have been great to have sealed class of something (kotlin concept)
    seed: Optional[int]
    num_loading_workers: int
    dataset: Optional[
        Union[
            PretrainNemoArgs,
            PretrainDatasetsArgs,
            DPOPretrainDatasetsArgs,
            TokenizedBytesDatasetArgs,
        ]
    ]


@dataclass
class PPOArgs:
    """Arguments related to PPO training"""

    # KL Controller
    init_kl_coef: float
    target: float
    horizon: float

    # PPO Loss
    gamma: float
    lam: float
    cliprange_value: float
    cliprange: float
    vf_coef: float

    # Other
    ppo_epochs: int
    queries_length: int
    num_layers_unfrozen: int  # Number of decoder layers to unfreeze (-1 means all layers are unfrozen)
    n_mini_batches_per_batch: int  # Number of optimization steps in 1 ppo_epoch
    generation_batch_size: Optional[int] = 32  # Micro batch size for generation


@dataclass
class DPOArgs:
    """Arguments related to DPO training"""

    # Dataloader
    max_prompt_length: int  # max length of each sample's prompt
    truncation_mode: str  # The truncation mode to use when truncating the prompt. Can be one of ['keep_end', 'keep_start']

    # DPO Loss
    beta: float  # the temperature parameter for the DPO loss, typically something in the range of `0.1` to `0.5`. We ignore the reference model as `beta` -> 0

    # Other
    num_layers_unfrozen: int  # Number of decoder layers to unfreeze (-1 means all layers are unfrozen)
    label_pad_token_id: Optional[int] = -100  # The token id to use for padding the labels


@dataclass
class RewardArgs:
    """Arguments related to Reward training"""

    # Other
    num_layers_unfrozen: int  # Number of decoder layers to unfreeze (-1 means all layers are unfrozen)


@dataclass
class LightEvalLoggingArgs:
    """Arguments related to logging for LightEval"""

    local_output_path: Optional[xPath] = None
    push_results_to_hub: Optional[bool] = None
    push_details_to_hub: Optional[bool] = None
    push_results_to_tensorboard: Optional[bool] = None
    hub_repo_results: Optional[str] = None
    hub_repo_details: Optional[str] = None
    hub_repo_tensorboard: Optional[str] = None
    tensorboard_metric_prefix: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.local_output_path, str):
            self.local_output_path = xPath(self.local_output_path)


@dataclass
class GenerationArgs:
    """ """

    sampler: Optional[Union[str, SamplerType]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n_samples: Optional[int] = None
    eos: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.sampler, str):
            self.sampler = SamplerType[self.sampler.upper()]


@dataclass
class LightEvalTasksArgs:
    """Arguments related to tasks for LightEval"""

    tasks: Optional[str] = None
    custom_tasks_file: Optional[xPath] = None
    max_samples: Optional[int] = None
    num_fewshot_seeds: Optional[int] = None

    dataset_loading_processes: Optional[int] = 8
    multichoice_continuations_start_space: Optional[bool] = None
    no_multichoice_continuations_start_space: Optional[bool] = None

    def __post_init__(self):
        if isinstance(self.custom_tasks_file, str):
            self.custom_tasks_file = xPath(self.custom_tasks_file)


@dataclass
class LightEvalSlurmArgs:
    """
    Args for SlurmExecutor.
    """

    logging_dir: str
    exclusive: bool = False
    ntasks_per_node: Optional[int] = 1
    job_name: Optional[str] = "eval"
    cpus_per_task: Optional[int] = None
    mem_per_cpu_gb: Optional[int] = None
    partition: Optional[str] = "production-cluster"
    sbatch_args: Optional[dict] = None
    time: Optional[str] = None
    nodes: Optional[int] = None
    lighteval_path: Optional[xPath] = None
    env_setup_command: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.lighteval_path, str):
            self.lighteval_path = xPath(self.lighteval_path)


@dataclass
class LightEvalConfig:
    """Arguments related to running LightEval on checkpoints.

    All is optional because you can also use this class to later supply arguments to override
    the saved config when running LightEval after training.
    """

    checkpoints_path: Optional[str] = None

    parallelism: Optional[ParallelismArgs] = None

    batch_size: Optional[int] = None
    generation: Optional[Union[GenerationArgs, Dict[str, GenerationArgs]]] = None

    tasks: Optional[LightEvalTasksArgs] = None
    logging: Optional[LightEvalLoggingArgs] = None

    slurm: Optional[LightEvalSlurmArgs] = None

    def __post_init__(self):
        # Automatically setup our node count
        if self.slurm.nodes is None:
            self.slurm.nodes = self.parallelism.dp * self.parallelism.pp * self.parallelism.tp // 8  # 8 GPUs per node


@dataclass
class Config:
    """Main configuration class"""

    general: GeneralArgs
    profile: Optional[ProfileArgs]
    checkpoints: CheckpointsArgs
    parallelism: Union[ParallelismArgs, PPOParallelismArgs]
    model: ModelArgs
    logging: LoggingArgs
    tokens: TokensArgs
    optimizer: OptimizerArgs
    learning_rate_scheduler: LRSchedulerArgs
    data: DataArgs
    ppo: Optional[PPOArgs]  # For PPO training
    dpo: Optional[DPOArgs]  # For DPO training
    reward: Optional[RewardArgs]  # For Reward training

    def __post_init__(self):
        # Some final sanity checks across separate arguments sections:
        if self.profile is not None and self.profile.profiler_export_path is not None:
            assert self.tokens.train_steps < 10

        if self.learning_rate_scheduler.lr_decay_steps is None:
            self.learning_rate_scheduler.lr_decay_steps = (
                self.tokens.train_steps - self.learning_rate_scheduler.lr_warmup_steps
            )

        # if lighteval, we need tokenizer to be defined
        if self.checkpoints.lighteval is not None:
            assert self.model.tokenizer_name_or_path is not None

    @property
    def global_batch_size(self):
        return self.tokens.micro_batch_size * self.tokens.batch_accumulation_per_replica * self.parallelism.dp

    def save_as_yaml(self, file_path: str):
        config_dict = serialize(self)
        file_path = str(file_path)
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f)

        # Sanity test config can be reloaded
        _ = get_config_from_file(file_path)

    def as_dict(self) -> dict:
        return serialize(self)


def serialize(data) -> dict:
    """Recursively serialize a nested dataclass to a dict - do some type conversions along the way"""
    if data is None:
        return None

    if not hasattr(data, "__dataclass_fields__"):
        return data

    result = {}
    for field in fields(data):
        value = getattr(data, field.name)
        if hasattr(value, "__dataclass_fields__"):
            result[field.name] = serialize(value)
        elif isinstance(value, xPath):
            result[field.name] = str(value)
        elif isinstance(value, PipelineEngine):
            result[field.name] = cast_pipeline_engine_to_str(value)
        elif isinstance(value, TensorParallelLinearMode):
            result[field.name] = value.name
        elif isinstance(value, RecomputeGranularity):
            result[field.name] = value.name
        elif isinstance(value, SamplerType):
            result[field.name] = value.name
        elif isinstance(value, torch.dtype):
            result[field.name] = dtype_to_str[value]
        elif isinstance(value, (list, tuple)):
            result[field.name] = [serialize(v) for v in value]
        elif isinstance(value, dict) and not value:
            result[field.name] = None  # So we can serialize empty dicts without issue with `datasets` in particular
        else:
            result[field.name] = value

    return result


str_to_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

dtype_to_str = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}


def cast_str_to_torch_dtype(str_dtype: str):
    if str_dtype in str_to_dtype:
        return str_to_dtype[str_dtype]
    else:
        raise ValueError(f"dtype should be a string selected in {str_to_dtype.keys()} and not {str_dtype}")


def cast_str_to_pipeline_engine(str_pp_engine: str) -> PipelineEngine:
    if str_pp_engine == "afab":
        return AllForwardAllBackwardPipelineEngine()
    elif str_pp_engine == "1f1b":
        return OneForwardOneBackwardPipelineEngine()
    else:
        raise ValueError(f"pp_engine should be a string selected in ['afab', '1f1b'] and not {str_pp_engine}")


def cast_pipeline_engine_to_str(pp_engine: PipelineEngine) -> str:
    if isinstance(pp_engine, AllForwardAllBackwardPipelineEngine):
        return "afab"
    elif isinstance(pp_engine, OneForwardOneBackwardPipelineEngine):
        return "1f1b"
    else:
        raise ValueError(
            f"pp_engine should be aan instance of AllForwardAllBackwardPipelineEngine or OneForwardOneBackwardPipelineEngine, not {type(pp_engine)}"
        )


class ConfigTypes(Enum):
    """Enum class for the different types of config files
    Name is the name of the class
    Value is the name of the object
    """

    LightEvalConfig = "lighteval"
    Config = "config"
    LightEvalSlurmArgs = "slurm"


# All the config types in ConfigTypes as a type
ConfigTypesClasses = TypeVar("ConfigTypesClasses", LightEvalConfig, Config, LightEvalSlurmArgs)


def get_config_from_file(config_path: str, config_type: Union[ConfigTypes, str, None] = None) -> ConfigTypesClasses:
    """Get a config objet from a file (python or YAML)

    Args:
        config_path: path to the config file
        config_type: if the file is a python file, type of the config object to get as a
            ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
            if None, will default to Config
    """
    if config_type is None:
        config_type = ConfigTypes.Config
    if isinstance(config_type, str):
        config_type = ConfigTypes(config_type)

    if config_path.endswith(".py"):
        # Copy the python file in a place where it's easy to import
        # dynamic_modules_path = os.path.dirname(config_path)
        # init_dynamic_modules(hf_modules_cache=dynamic_modules_path)
        try:
            factory = dataset_module_factory(
                config_path,
                # dynamic_modules_path=dynamic_modules_path,
                download_mode=DownloadMode.FORCE_REDOWNLOAD
                if os.environ.get("LOCAL_RANK", None) == 0
                else DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find a config file at {config_path}. Please check the path and try again."
            )
        module = importlib.import_module(factory.module_path)
        config: ConfigTypesClasses = getattr(module, config_type.value)
    else:
        # Open the file and load the file
        with open(config_path) as f:
            args = yaml.load(f, Loader=SafeLoader)

        # Make a nice dataclass from our yaml
        try:
            config = from_dict(
                data_class=globals()[config_type.name],
                data=args,
                config=dacite.Config(
                    cast=[xPath],
                    type_hooks={
                        torch.dtype: cast_str_to_torch_dtype,
                        PipelineEngine: cast_str_to_pipeline_engine,
                        TensorParallelLinearMode: lambda x: TensorParallelLinearMode[x.upper()],
                        RecomputeGranularity: lambda x: RecomputeGranularity[x.upper()],
                        SamplerType: lambda x: SamplerType[x.upper()],
                    },
                    strict_unions_match=True,
                    strict=True,
                ),
            )
        except Exception as e:
            raise ValueError(f"Error parsing config file {config_path}: {e}")

    return config


@dataclass
class AllTrainerConfigs:
    config: Config
    model_config: Union[NanotronConfigs, AutoConfig]


def get_all_trainer_configs(config_or_config_file: Union[Config, str]) -> AllTrainerConfigs:
    """Get the config and the config file path from either a config object or a config file path."""
    if isinstance(config_or_config_file, str):
        config: Config = get_config_from_file(config_or_config_file)
    else:
        config = config_or_config_file

    if config.model.hf_model_name:
        trust_remote_code = (
            config.model.remote_code.trust_remote_code if hasattr(config.model, "remote_code") else None
        )
        model_config = AutoConfig.from_pretrained(config.model.hf_model_name, trust_remote_code=trust_remote_code)
    else:
        assert config.model.model_config is not None, "Either model.hf_model_name or model.model_config must be set"
        model_config = config.model.model_config

    return AllTrainerConfigs(config=config, model_config=model_config)


@dataclass
class AllLightEvalConfigs:
    lighteval: LightEvalConfig
    parallelism: ParallelismArgs
    model: ModelArgs
    trainer: Config
    s3: UploadCheckpointOnS3Args
    logging: LightEvalLoggingArgs
    run: str
    step: str


def get_all_lighteval_configs(
    config_or_config_file: Union[Config, str],
    lighteval_override: Optional[Union[LightEvalConfig, str]] = "",
) -> AllLightEvalConfigs:
    """Load all the configs needed to run LightEval from either a config object or a config file path.

    Optionally override the lighteval config with a LightEvalConfig object or a config file path.
    """
    if isinstance(config_or_config_file, str):
        config: Config = get_config_from_file(config_or_config_file)
    else:
        config = config_or_config_file

    if lighteval_override and isinstance(lighteval_override, str):
        override_lighteval_config: LightEvalConfig = get_config_from_file(lighteval_override, "lighteval")
    else:
        override_lighteval_config = lighteval_override

    # update config with overriden lighteval config params
    def update_dataclass(target: Any, updates: Any):
        if not is_dataclass(target) or not is_dataclass(updates):
            raise ValueError("Both target and updates should be dataclass instances")

        for field in fields(target):
            update_value = getattr(updates, field.name)

            if update_value is not None:
                if is_dataclass(update_value):
                    if is_dataclass(getattr(target, field.name)):
                        update_dataclass(getattr(target, field.name), update_value)
                    elif getattr(target, field.name) is None:
                        setattr(target, field.name, update_value)
                    else:
                        raise ValueError(
                            f"Cannot update dataclass field {field.name} with value {update_value} because the field is neither a similar dataclass nor None"
                        )
                elif update_value is not field.default:
                    setattr(target, field.name, update_value)

    if override_lighteval_config:
        if config.checkpoints.lighteval is None:
            config.checkpoints.lighteval = override_lighteval_config
        else:
            update_dataclass(config.checkpoints.lighteval, override_lighteval_config)

    if config.model.hf_model_name:
        trust_remote_code = (
            config.model.remote_code.trust_remote_code if hasattr(config.model, "remote_code") else None
        )
        config.model.model_config = AutoConfig.from_pretrained(
            config.model.hf_model_name, trust_remote_code=trust_remote_code
        )

    if config.general.run is None:
        config.general.run = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    return AllLightEvalConfigs(
        lighteval=config.checkpoints.lighteval,
        parallelism=config.checkpoints.lighteval.parallelism,
        model=config.model,
        trainer=config,
        s3=config.checkpoints.s3,
        logging=config.checkpoints.lighteval.logging,
        run=config.general.run,
        step=config.general.step,
    )
