# import json

# with open('/fsx/phuc/project_data/doremi/datasets/the_pile_raw/01.jsonl', 'r') as f:
#     for line in f:
#         json_data = json.loads(line)
#         print(json_data)


from datasets import load_dataset

# dataset = load_dataset("EleutherAI/pile", num_proc=256)

# ds = concatenate_datasets(
#     [
#         dataset["train"],
#         dataset["validation"],
#         dataset["test"]
#     ]
# )

ds = load_dataset("/fsx/phuc/project_data/doremi/datasets/the_pile_raw/01.jsonl", num_proc=256)


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
    dset = ds_m.filter(lambda x: x["domain"] == domain, num_proc=24)
    dset.to_parquet(f"split-{domain}-0.parquet")
