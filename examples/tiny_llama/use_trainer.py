"""

You can run using command:
```
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/use_trainer.py --config-file examples/config_nouamane_test_trainer.yaml
```
"""
import argparse
from typing import Optional

from huggingface_hub import __version__ as hf_hub_version
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from transformers import __version__ as tf_version

from nanotron.core import distributed as dist
from nanotron.core import logging
from nanotron.core.logging import log_rank
from nanotron.core.utils import (
    main_rank_first,
)
from nanotron.dataloaders.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_train_dataloader,
)
from nanotron.dataloaders.nemo import get_nemo_dataloader, get_nemo_datasets
from nanotron.trainer import DistributedTrainer

logger = logging.get_logger(__name__)


def get_dataloader(trainer: DistributedTrainer, sanity_check_dataloader_interval: Optional[int] = None):
    # Prepare dataloader
    tokenizer_path = (
        trainer.config.model.hf_model_name
        if trainer.config.model.hf_model_name is not None
        else trainer.config.model.tokenizer_name_or_path
    )
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

    data_log, valid_dataloader = None, None
    if trainer.config.data.dataset is None:
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=trainer.config.data.seed,
            dpg=trainer.dpg,
        )()
    else:  # TODO: other datasets
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
    return (dataloader, valid_dataloader), data_log


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
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
    dataloader, data_log = get_dataloader(trainer, args.sanity_check_dataloader_interval)

    # Train
    trainer.train(dataloader, data_config_log=data_log)
