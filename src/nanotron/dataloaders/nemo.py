from dataclasses import dataclass
from typing import Union

from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from nanotron.config import PretrainNemoArgs
from nanotron.nn import distributed as dist
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.nn.process_groups import DistributedProcessGroups
from nanotron.nn.utils import human_format

from .data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from .dataloader import DataCollatorForCLM, EmptyInfiniteDataset, get_dataloader_worker_init
from .nemo_dataset import GPTDataset, SubsetSplitLog, build_train_valid_test_datasets
from .nemo_dataset.blendable_dataset import BlendedSubsetSplitLog
from .nemo_dataset.dataset_utils import compile_helper

try:
    tb_logger_available = True
except ImportError:
    tb_logger_available = False

logger = logging.get_logger(__name__)


@dataclass
class TrainDataLog:
    global_batch_size: int
    sequence_length: int
    total_training_tokens: int
    human_total_train_tokens: str
    train_num_samples: int
    eval_num_samples: int
    test_num_samples: int
    train_subset: Union[SubsetSplitLog, BlendedSubsetSplitLog]
    eval_subset: Union[SubsetSplitLog, BlendedSubsetSplitLog]
    test_subset: Union[SubsetSplitLog, BlendedSubsetSplitLog]


def get_nemo_datasets(
    config: PretrainNemoArgs,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    global_batch_size: int,
    train_steps: int,
    limit_val_batches: int,
    val_check_interval: int,
    test_iters: int,
    seed,
    dpg: DistributedProcessGroups,
):
    log_rank("Building GPT datasets.", logger=logger, level=logging.INFO, rank=0)
    if dist.get_rank(dpg.world_pg) == 0:
        try:
            compile_helper()
        except ImportError:
            raise ImportError(
                "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
            )
    dist.barrier(dpg.world_pg)

    if limit_val_batches > 1.0 and isinstance(limit_val_batches, float):
        raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

    eval_iters = (train_steps // val_check_interval + 1) * limit_val_batches

    train_valid_test_num_samples = [
        train_steps * global_batch_size,
        eval_iters * global_batch_size,
        test_iters * global_batch_size,
    ]

    if limit_val_batches <= 1.0 and isinstance(limit_val_batches, float):
        train_valid_test_num_samples[
            1
        ] = 1  # This is to make sure we only have one epoch on every validation iteration

    train_ds, validation_ds, test_ds = build_train_valid_test_datasets(
        cfg=config,
        tokenizer=tokenizer,
        data_prefix=config.data_prefix,
        splits_string=config.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=sequence_length,
        seed=seed,
        dpg=dpg,
        skip_warmup=config.skip_warmup,
    )

    total_training_tokens = train_valid_test_num_samples[0] * sequence_length
    train_data_log = TrainDataLog(
        train_num_samples=train_valid_test_num_samples[0],
        eval_num_samples=train_valid_test_num_samples[1],
        test_num_samples=train_valid_test_num_samples[2],
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        total_training_tokens=total_training_tokens,
        human_total_train_tokens=human_format(total_training_tokens),
        train_subset=train_ds.subset_log if train_ds is not None else None,
        eval_subset=validation_ds.subset_log if validation_ds is not None else None,
        test_subset=test_ds.subset_log if test_ds is not None else None,
    )

    return (train_ds, validation_ds, test_ds), train_data_log


def get_nemo_dataloader(
    dataset: Union[GPTDataset, Dataset],
    sequence_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    cfg: PretrainNemoArgs,
    num_workers: int,
    consumed_samples: int,
    dpg: DistributedProcessGroups,
    input_pp_rank: int,
    output_pp_rank: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
) -> DataLoader:
    # Only some rank require to run the dataloader.
    if dist.get_rank(dpg.pp_pg) not in [
        input_pp_rank,
        output_pp_rank,
    ]:
        dataset = EmptyInfiniteDataset(length=len(dataset))

    log_rank(
        f"Building dataloader with consumed samples: {consumed_samples}", logger=logger, level=logging.INFO, rank=0
    )
    # Megatron sampler
    if cfg.dataloader_type == "single":
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=dist.get_rank(dpg.dp_pg),
            data_parallel_size=dpg.dp_pg.size(),
            drop_last=dataloader_drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=cfg.pad_samples_to_global_batch_size,
        )
    elif cfg.dataloader_type == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=dist.get_rank(dpg.dp_pg),
            data_parallel_size=dpg.dp_pg.size(),
            global_batch_size=global_batch_size,
            drop_last=dataloader_drop_last,
            pad_samples_to_global_batch_size=cfg.pad_samples_to_global_batch_size,
        )
    else:
        raise ValueError('cfg.dataloader_type must be "single" or "cyclic"')

    # We use the data collator to put the tensors on the right pipeline parallelism rank
    data_collator = DataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        dpg=dpg,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(dpg.dp_pg)),
    )
