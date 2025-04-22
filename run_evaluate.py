"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""
import argparse
import time
from typing import Dict, cast

import nanotron.distributed as dist
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    NanosetDatasetsArgs,
    PretrainDatasetsArgs,
    Qwen2Config,
    SFTDatasetsArgs,
)
from nanotron.data.dataloader import (
    dummy_infinite_data_generator,
    get_train_dataloader,
)
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.data.processing import (
    clm_process,
    get_datasets,
)
from nanotron.data.sft_processing import prepare_sft_dataset
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first
from torch.utils.data import DataLoader

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)

# import lovely_tensors as lt

# lt.monkey_patch()


def get_valid_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    # consumed_train_samples: int, We will never use this because in each valid iteration we consume all the samples
):

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Only support Validation with Nanoset
    if isinstance(data.dataset, NanosetDatasetsArgs):
        # Create Nanoset
        from nanotron.data.nanoset import Nanoset

        with main_rank_first(trainer.parallel_context.world_pg):
            tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            eos_token_id = tokenizer.eos_token_id
            assert (
                eos_token_id is not None or data.dataset.return_positions is False
            ), "Tokenizer must have an eos token if return_positions is True"
            log_rank(
                f"[Nanoset] Creating Nanoset with {len(data.dataset.validation_folder)} dataset folders and {trainer.config.tokens.limit_val_batches * trainer.global_batch_size if trainer.config.tokens.limit_val_batches else None} validation samples",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            start_time = time.time()
            valid_dataset = Nanoset(
                dataset_folders=data.dataset.validation_folder, 
                dataset_weights=None,   # TODO(@paultltc): Should we weight the valid dataset?
                sequence_length=trainer.sequence_length,
                token_size=data.dataset.token_size_in_bytes,
                num_samples=trainer.config.tokens.limit_val_batches * trainer.global_batch_size if trainer.config.tokens.limit_val_batches else None,
                random_seed=data.seed,
                return_positions=data.dataset.return_positions,
                eos_token_id=eos_token_id,
            )
            end_time = time.time()
            log_rank(
                f"[Nanoset] Time taken to create Nanoset: {time.strftime('%M:%S', time.gmtime(end_time - start_time))} (MM:SS)",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

        # Prepare dataloader
        valid_dataloader = build_nanoset_dataloader(
            valid_dataset,
            trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
            use_position_ids=isinstance(trainer.model_config, Qwen2Config),
        )

        return valid_dataloader
    else:
        raise ValueError(
            f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}. Validation is currently just supported for MultilingualNanoset"
        )

def get_valid_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)

        if stage.data.dataset.validation_folder is not None:
            log_rank(
                f"[Validation Plan] Stage {stage.name} has {len(stage.data.dataset.validation_folder)} folders with samples for the validation set",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            dataloader = (
                get_valid_dataloader_from_data_stage(trainer, stage.data)
                if stage_idx == 0
                else lambda stage=stage: get_valid_dataloader_from_data_stage(trainer, stage.data)
            )
            # TODO(tj.solergibert) As we are creating again the valid dataloader in every validation stage, we print multiple times
            # the validation MultilingualNanoset info (Number of samples, etc.) [UPDATE: ]. In order to solve that, we could get rid of this lambda
            # funcs and directly create all dataloaders.
            #
            # This lambda functs (Used in training too) are for creating the DataLoaders lazyly FOR 1. Start training faster instead
            # of creating multiple DataLoaders 2. Consume less memory as the lambda func is lighter that the DataLoader object with
            # the Dataset, collator, etc.
            # BUT 1. The Nanoset creation process is very fast and 2. Nanosets doesn't consume any memory at all till we start sampling
            # from the Nanoset. Also they later transform the DataLoader into a Iterator object so it's impossible to retrieve
            # the DataLoader object again to delete it (More comments in trainer.py)
            dataloaders[stage.name] = dataloader
        else:
            dataloaders[stage.name] = None
    return dataloaders

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    valid_dataloader = get_valid_dataloader(trainer)

    # Train
    trainer.evaluate(valid_dataloader)