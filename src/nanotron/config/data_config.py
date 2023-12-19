from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List

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
    skip_tokens: Optional[int] = 0  # Optional number of tokens to skip at the beginning (We'll only train on the rest)


@dataclass
class TokenizedBytesDatasetFolderArgs:
    folder: str
    filename_pattern: str
    skip_tokens: Optional[int] = 0  # Optional number of tokens to skip at the beginning (We'll only train on the rest)


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
class DataArgs:
    """Arguments related to the data and data files processing"""

    seed: Optional[int]
    num_loading_workers: int
    dataset: Optional[
        Union[
            PretrainNemoArgs,
            PretrainDatasetsArgs,
            TokenizedBytesDatasetArgs,
        ]
    ]
