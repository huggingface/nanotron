import datasets
from datasets import load_dataset

datasets.config.DOWNLOADED_DATASETS_PATH = "/fsx/phuc/.cache"
dataset = load_dataset("ArmelR/the-pile-splitted", split="train[:100]", num_proc=100)

print("done")
