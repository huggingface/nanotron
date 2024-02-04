import os
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def find_subfolders(path):
    subfolders = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            subfolders.append(full_path)
    return subfolders


DOMAIN_KEYS = [
    "Books3",  # 0
    "ArXiv",  # 1
    "Gutenberg (PG-19)",  # 2
    "Ubuntu IRC",  # 17, done
    "BookCorpus2",  # 18, launched
    "EuroParl",  # 19, launch,
    "PhilPapers",
]

SHARD_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/tokenized_data_separate"
SAVE_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/tokenized_data/train"

# domain_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
# domain_idx = 5
domain_idx = 6


DOMAIN_PATH = os.path.join(SHARD_PATH, DOMAIN_KEYS[domain_idx])
saved_path = Path(f"{SAVE_PATH}/{DOMAIN_KEYS[domain_idx]}")


print(f"domain_idx: {domain_idx}")
print(f"domain name: {DOMAIN_KEYS[domain_idx]}")
print(f"DOMAIN_PATH: {DOMAIN_PATH}")
print(f"saved_path: {saved_path}")

dataset_paths = find_subfolders(DOMAIN_PATH)
ds = []

for path in dataset_paths:
    d = load_from_disk(path)
    ds.append(d)

raw_dataset = concatenate_datasets(ds)

if not os.path.exists(saved_path):
    os.makedirs(saved_path)

raw_dataset.save_to_disk(saved_path)
