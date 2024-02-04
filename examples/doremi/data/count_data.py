import os

from datasets import load_from_disk
from tqdm import tqdm


def find_subfolders(path):
    subfolders = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            subfolders.append(full_path)
    return subfolders


# DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_splitted/tokenized_data"
DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/splitted/Enron Emails"

dataset_paths = find_subfolders(DATASET_PATH)

d = load_from_disk("/fsx/phuc/project_data/doremi/datasets/the_pile_raw/tokenized_data/train/Enron Emails")

assert 1 == 1

ds = []
total = 0
for dataset_path in tqdm(dataset_paths, desc="Loading tokenized dataset from disk"):
    d = load_from_disk(dataset_path)
    total += len(d["train"])

assert 1 == 1
