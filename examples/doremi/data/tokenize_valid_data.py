import os
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

# from dataloader import get_doremi_datasets
from nanotron.config import get_config_from_file
from nanotron.doremi.config import DoReMiConfig

try:
    from datasets import (
        # ClassLabel,
        Dataset,
        # DatasetDict,
        Features,
        Sequence,
        Value,
        # concatenate_datasets,
        load_dataset,
    )

    # from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    # from transformers import __version__ as tf_version
    # from transformers.trainer_pt_utils import DistributedSamplerWithLoop
except ImportError:
    warnings.warn("Datasets and/or Transformers not installed, you'll be unable to use the dataloader.")


def doremi_clm_process(
    # domain_idx: int,
    raw_dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
):
    """Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`, where chunks overlap by a single token."""
    # Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/examples/pytorch/language-modeling/run_clm.py#L391-L439

    def group_texts(examples: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        return group_texts(tokenized_batch)

    train_dataset = raw_dataset.map(
        _tokenize_and_group_texts,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features(
            {
                "input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1),
                "domain_ids": Value(dtype="int64"),
            }
        ),
        batched=True,
        # num_proc=256,
        # writer_batch_size=1,
        # TODO: remove harcode
        # load_from_cache_file=not dataset_overwrite_cache,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
        # cache_file_name="/fsx/phuc/.cache/huggingface_cache/huggingface/modules/datasets_modules/datasets/mc4"
    )

    return train_dataset


def tokenize_dataset(config, raw_dataset):
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # print(f"Downloading dataset {config.data.dataset.hf_dataset_or_datasets}")

    # raw_datasets = get_doremi_datasets(
    #     hf_dataset=config.data.dataset.hf_dataset_or_datasets,
    #     domain_name=domain_name,
    #     splits=config.data.dataset.hf_dataset_splits,
    # )["train"]

    # NOTE: only for the pile splitted

    # features = Features(
    #     {"text": Value("string"), "meta": {"pile_set_name": Value("string")}, "domain": ClassLabel(names=domain_keys)}
    # )

    # raw_dataset = load_dataset(
    #     config.data.dataset.hf_dataset_or_datasets,
    #     domain_name,
    #     split=["train"],
    #     # TODO: set this in config
    #     num_proc=24,
    #     features=features,
    # )[0]

    train_dataset = doremi_clm_process(
        # domain_idx=domain_idx,
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        # text_column_name=config.data.dataset.text_column_name,
        text_column_name="text",
        dataset_processing_num_proc_per_process=3,
        dataset_overwrite_cache=config.data.dataset.dataset_overwrite_cache,
        sequence_length=1024,
    )

    return train_dataset


def find_subfolders(path):
    subfolders = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            subfolders.append(full_path)
    return subfolders


def map_domain_ids(example):
    meta = example["meta"]
    example["domain"] = meta["pile_set_name"]
    example["domain_ids"] = DOMAIN_KEYS.index(meta["pile_set_name"])
    return example


if __name__ == "__main__":
    config_file = "/fsx/phuc/projects/nanotron/examples/doremi/config_280m_llama.yaml"
    raw_file_path = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw_test/test.jsonl"
    save_path = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/tokenized_data/test"

    DOMAIN_KEYS = [
        "Pile-CC",
        "Github",
        "OpenWebText2",
        "StackExchange",
        "Wikipedia (en)",
        "PubMed Abstracts",
        "USPTO Backgrounds",
        "FreeLaw",
        "PubMed Central",
        "Enron Emails",
        "HackerNews",
        "NIH ExPorter",
        "Books3",  # 12
        "ArXiv",  # 13 , launched
        "DM Mathematics",
        "OpenSubtitles",
        "Gutenberg (PG-19)",  # 16, done
        "Ubuntu IRC",  # 17, done
        "BookCorpus2",  # 18, launched
        "EuroParl",  # 19, launch
        "YoutubeSubtitles",
        "PhilPapers",
    ]

    config = get_config_from_file(config_file, config_class=DoReMiConfig)
    print(f"raw_file_path: {raw_file_path}")

    raw_dataset = load_dataset("json", data_files=raw_file_path, num_proc=256)
    raw_dataset = Dataset.from_dict(raw_dataset["train"][:10])
    raw_dataset = raw_dataset.map(
        map_domain_ids,
        # num_proc=256
    )

    train_dataset = tokenize_dataset(config, raw_dataset=raw_dataset)

    cache_path = Path(save_path)
    os.makedirs(cache_path, exist_ok=True)
    train_dataset.save_to_disk(cache_path)
