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

SAVE_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/splitted"

paths = [
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/00.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/01.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/02.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/03.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/04.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/05.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/06.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/07.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/08.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/09.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/10.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/11.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/12.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/13.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/14.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/15.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/16.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/17.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/18.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/19.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/20.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/21.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/22.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/23.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/24.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/25.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/26.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/27.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/28.jsonl",
    "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/29.jsonl",
]

job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
path = paths[job_id]

print(f"job_id: {job_id}")
print(f"path: {path}")

ds = load_dataset("json", data_files=path, num_proc=256)


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
    saved_path = Path(f"{SAVE_PATH}/{domain}/{job_id}")
    dset = ds_m.filter(lambda x: x["domain"] == domain, num_proc=24)

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    dset.save_to_disk(saved_path)

print("done")
