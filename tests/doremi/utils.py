import torch
from torch.utils.data import Dataset


def generate_toy_dataset(tokenizer, num_domains) -> Dataset:

    # # NOTE: later on, use ArmelR/the-pile-splitted as testing dataset
    class TokenizedDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings["input_ids"])

    def generate_dataset(sentence, repetitions, tokenizer):
        dataset = {}
        for i, rep in enumerate(repetitions):
            encoded_batch = tokenizer([sentence] * rep, padding=True, truncation=True, return_tensors="pt").to("cuda")
            dataset[f"domain_{i+1}"] = TokenizedDataset(encoded_batch)
        return dataset

    repetitions = [50 * 2**i for i in range(num_domains)]
    dataset = generate_dataset("hello world", repetitions, tokenizer)
    return dataset
