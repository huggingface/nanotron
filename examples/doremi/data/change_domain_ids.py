import os
from pathlib import Path

from datasets import load_from_disk

if __name__ == "__main__":
    # domain_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    domain_idx = 8

    DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_splitted/tokenized_data"
    DOMAIN_KEYS = [
        "Github",
        "FreeLaw",
        "OpenWebText2",
        "PubMed Abstracts",
        "DM Mathematics",
        "OpenSubtitles",
        "HackerNews",
        "NIH ExPorter",
        "PubMed Central",
        "Enron Emails",
    ]
    NEW_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_splitted/tokenized_data_with_correct_domain"
    # TOKENIZED_DATASETS = [f"{DATASET_PATH}/{domain_name}" for domain_name in DOMAIN_KEYS]
    TOKENIZED_DATASETS = [f"{NEW_PATH}/{domain_name}" for domain_name in DOMAIN_KEYS]
    TARGET_PATH = TOKENIZED_DATASETS[domain_idx]

    d = load_from_disk(TARGET_PATH)
    domain_name = DOMAIN_KEYS[domain_idx]

    # def update_domain_idx(example, domain_ids):
    #     example['domain_ids'] = domain_ids
    #     return example

    # d.map(update_domain_idx, fn_kwargs={'domain_ids': domain_idx}, num_proc=1)

    from functools import partial

    # Define your batch processing function
    def set_domain_ids(batch, domain_ids):
        # Set the 'domain_ids' of each item in the batch to 'n'
        # batch["domain_ids"] = [domain_ids] * len(batch["domain_ids"])
        # batch["domain_ids"] = [domain_ids for _ in range(len(batch["domain_ids"]))]
        batch["domain_ids"] = domain_ids
        return batch

    # d = d.map(partial(set_domain_ids, domain_ids=domain_idx), batched=True)
    d = d.map(partial(set_domain_ids, domain_ids=domain_idx), num_proc=24)

    cache_path = Path(NEW_PATH) / f"{domain_name}"
    os.makedirs(cache_path, exist_ok=True)
    d.save_to_disk(cache_path)
