# import json

# with open('/fsx/phuc/project_data/doremi/datasets/the_pile_raw/01.jsonl', 'r') as f:
#     for line in f:
#         json_data = json.loads(line)
#         print(json_data)


import os
from pathlib import Path

from datasets import load_dataset

# dataset = load_dataset("EleutherAI/pile", num_proc=256)

# ds = concatenate_datasets(
#     [
#         dataset["train"],
#         dataset["validation"],
#         dataset["test"]
#     ]
# )

SAVE_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/splitted_test"

DATA_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw_test/test.jsonl"

ds = load_dataset("json", data_files=DATA_PATH, num_proc=256)


def f(example):
    meta = example["meta"]
    example["domain"] = meta["pile_set_name"]
    return example


ds_m = ds.map(f, num_proc=256)

domains = [
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
    "Books3",
    "ArXiv",
    "DM Mathematics",
    "OpenSubtitles",
    "Gutenberg (PG-19)",
    "Ubuntu IRC",
    "BookCorpus2",
    "EuroParl",
    "YoutubeSubtitles",
    "PhilPapers",
]

for domain in domains:
    print(f"------ {domain} ------")
    saved_path = Path(f"{SAVE_PATH}/{domain}")
    dset = ds_m.filter(lambda x: x["domain"] == domain, num_proc=24)

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    dset.save_to_disk(saved_path)

print("done")
