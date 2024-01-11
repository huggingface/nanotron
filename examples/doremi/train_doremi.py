"""

You can run using command:
```
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/tiny_llama.yaml
```
"""
import argparse
from typing import Optional

from dataloader import (
    clm_process,
    get_datasets,
    get_train_dataloader,
)
from huggingface_hub import __version__ as hf_hub_version
from nanotron import logging
from nanotron.config import (
    PretrainDatasetsArgs,
)
from nanotron.core import distributed as dist
from nanotron.core.utils import (
    main_rank_first,
)
from nanotron.logging import log_rank
from nanotron.trainer import DistributedTrainer
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from transformers import __version__ as tf_version

logger = logging.get_logger(__name__)


def get_dataloader(trainer: DistributedTrainer, sanity_check_dataloader_interval: Optional[int] = None):
    # Prepare dataloader
    tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
    log_rank(
        f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
        logger=logger,
        level=logging.INFO,
        rank=None,
    )

    if isinstance(trainer.model, DistributedDataParallel):
        input_pp_rank = trainer.model.module.input_pp_rank
        output_pp_rank = trainer.model.module.output_pp_rank
    else:
        input_pp_rank = trainer.model.input_pp_rank
        output_pp_rank = trainer.model.output_pp_rank

    if trainer.config.data.dataset is None:
        # dataloader = dummy_infinite_data_generator(
        #     micro_batch_size=trainer.micro_batch_size,
        #     sequence_length=trainer.sequence_length,
        #     input_pp_rank=input_pp_rank,
        #     output_pp_rank=output_pp_rank,
        #     vocab_size=trainer.model_config.vocab_size,
        #     seed=trainer.config.data.seed,
        #     dpg=trainer.dpg,
        # )()
        pass

    elif isinstance(trainer.config.data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        with main_rank_first(trainer.dpg.world_pg):
            # 1st device processes dataset and cache it, then other devices load from cache
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits

            raw_dataset = get_datasets(
                hf_dataset_or_datasets=trainer.config.data.dataset.hf_dataset_or_datasets,
                splits=trainer.config.data.dataset.hf_dataset_splits,
            )["train"]
            tokenizer = AutoTokenizer.from_pretrained(trainer.config.tokenizer.tokenizer_name_or_path)
            tokenizer.pad_token = tokenizer.eos_token

            import torch

            # def generate_toy_dataset(tokenizer, num_domains) -> Dataset:
            #     # # NOTE: later on, use ArmelR/the-pile-splitted as testing dataset
            #     class TokenizedDataset(Dataset):
            #         def __init__(self, encodings):
            #             self.encodings = encodings

            #         def __getitem__(self, idx):
            #             return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

            #         def __len__(self):
            #             return len(self.encodings["input_ids"])

            #     def generate_dataset(sentence, repetitions, tokenizer):
            #         dataset = {}
            #         for i, rep in enumerate(repetitions):
            #             # encoded_batch = tokenizer([sentence] * rep, padding=True, truncation=True, return_tensors="pt").to("cuda")
            #             # dataset[f"domain_{i+1}"] = TokenizedDataset(encoded_batch)
            #             dataset[f"domain_{i+1}"] = [sentence] * rep
            #         return dataset

            #     repetitions = [50 * 2**i for i in range(num_domains)]
            #     dataset = generate_dataset("hello world", repetitions, tokenizer)
            #     return dataset

            NUM_DOMAINS = 5
            raw_datasets = {f"domain_{i+1}": raw_dataset for i in range(NUM_DOMAINS)}

            train_datasets = {}
            # TODO(xrsrke): parallelize this
            for domain_name, raw_dataset in raw_datasets.items():
                train_dataset = clm_process(
                    raw_dataset=raw_dataset,
                    tokenizer=tokenizer,
                    text_column_name=trainer.config.data.dataset.text_column_name,
                    dataset_processing_num_proc_per_process=trainer.config.data.dataset.dataset_processing_num_proc_per_process,
                    dataset_overwrite_cache=trainer.config.data.dataset.dataset_overwrite_cache,
                    sequence_length=trainer.sequence_length,
                )
                train_datasets[domain_name] = train_dataset

            assert 1 == 1

            import torch.nn.functional as F

            domain_weights = F.softmax(torch.ones(NUM_DOMAINS, requires_grad=False) / NUM_DOMAINS)

            dataloader = get_train_dataloader(
                domain_weights=domain_weights,
                train_datasets=train_datasets,
                sequence_length=trainer.sequence_length,
                dpg=trainer.dpg,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=trainer.consumed_train_samples,
                dataloader_num_workers=trainer.config.data.num_loading_workers,
                seed_worker=trainer.config.data.seed,
                dataloader_drop_last=True,
            )

            for batch in dataloader:
                assert 1 == 1

            assert 1 == 1
            # Check if we have enough samples for train_steps
            assert (
                trainer.config.tokens.train_steps - trainer.start_iteration_step
            ) * trainer.global_batch_size // trainer.dpg.dp_pg.size() < len(dataloader), (
                f"Dataset is too small for steps ({len(dataloader)} < {(trainer.config.tokens.train_steps - trainer.start_iteration_step) * trainer.global_batch_size // trainer.dpg.dp_pg.size()}), "
                f"Try train_steps<={len(dataloader) * trainer.dpg.dp_pg.size() // trainer.global_batch_size + trainer.start_iteration_step}"
            )
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {trainer.config.data.dataset}")

    # SANITY dataloder first samples
    if sanity_check_dataloader_interval is not None:
        NUM_BATCHES = 10
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=trainer.config.model.tokenizer_revision)
        if dist.get_rank() == 0:
            check_step = -1

            with open("sanity_check.txt", "w") as f:
                f.write("")
            for i, batch in enumerate(dataloader):
                check_step += 1
                if i % sanity_check_dataloader_interval == 0:
                    with open("sanity_check.txt", "a") as f:
                        f.write("\n\n")
                        f.write("*" * 40)
                        f.write(f"Sanity check {check_step}")
                        f.write("*" * 40)
                    print(batch)

                    # joblib.dump(batch, f"sanity_check_{check_step}.pkl")
                    texts = tokenizer.batch_decode(
                        batch["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )

                    for j, text in enumerate(texts):
                        # if j != 1:
                        #     continue
                        print(f"\n\n>>Batch {i} || Sample {j}<<\n")
                        print(text[:400])
                        with open("sanity_check.txt", "a") as f:
                            f.write(f"\n\n>>Batch {i} || Sample {j}<<\n")
                            f.write(text)

                    if i // sanity_check_dataloader_interval == NUM_BATCHES - 1:
                        break
            assert False
        dist.barrier()
    return dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    parser.add_argument("--job-id", type=str, help="Optional job name")
    parser.add_argument(
        "--sanity-check-dataloader-interval",
        type=int,
        default=None,
        help="Optional interval to print dataloader samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer, args.sanity_check_dataloader_interval)

    # Train
    trainer.train(dataloader)
