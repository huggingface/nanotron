import csv
import random
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset

# from arguments import DataTrainingArguments
from torch.nn import CrossEntropyLoss
from torch.utils.data import IterableDataset
from tqdm import tqdm

# from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedTokenizer


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name_or_path: str = field(
        default="bigcode/the-stack-dedup",
        metadata={"help": "Name or path of the dataset. The dataset can be on the hub or locally."},
    )
    split_file: str = field(
        default=".",
        metadata={
            "help": "Path to the file containing the name of the different splits of the dataset. It is useful for the argument data_dir of load_dataset and can be used \
                               to directly run the datasets from files."
        },
    )
    streaming: bool = field(default=True, metadata={"help": "Do we load the datasets in streaming mode."})
    dataset_seed: int = field(default=42, metadata={"help": "Seed parameter"})
    validation_dataset_path: str = field(
        default=None, metadata={"help": "Path to the validation set if it is local. "}
    )
    valid_set_size: float = field(
        default=0.05,
        metadata={
            "help": " Size of the test split If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split."
        },
    )
    number_of_domains: int = field(
        default=None,
        metadata={"help": " For debugging purposes or quicker training, truncate the number of domains to considers."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length: int = field(
        default=1024,
        metadata={"help": ("Input sequence length after tokenization. ")},
    )
    packing: bool = field(default=True, metadata={"help": "Whether to use packing or not."})
    num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    shuffle: bool = field(default=True, metadata={"help": "Shuffle the training data on the fly"})
    input_column_name: str = field(default="content", metadata={"help": "The column to consider for the training."})
    num_of_sequences: int = field(default=1024, metadata={"help": "Number of token sequences to keep in buffer."})


def chars_token_ratio(dataset, tokenizer, input_column_name="content", nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = example[input_column_name]
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        input_column_name="content",
        dataset_index=None,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.input_column_name = input_column_name
        self.dataset_index = dataset_index

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.input_column_name])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                if self.dataset_index is not None:
                    yield {
                        "index": torch.LongTensor([self.dataset_index]),
                        "input_ids": torch.LongTensor(example),
                        "labels": torch.LongTensor(example),
                    }
                else:
                    yield {
                        "input_ids": torch.LongTensor(example),
                        "labels": torch.LongTensor(example),
                    }


"""
class LinearWarmupExponentialLR(LRScheduler):
    \"""
    Exponential LR with linear warmup and decay to some end LR.
    \"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)
        # figure out decay rate to use to get within 1e-10 of lr_end at end of training
        self.gammas = [np.exp(np.log(1e-10 / (base_lr - self.lr_end)) / (self.num_training_steps - self.num_warmup_steps))
                       for base_lr in self.base_lrs]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.lr_end + (base_lr - self.lr_end) * gamma ** (self.last_epoch - self.num_warmup_steps) for gamma, base_lr in zip(self.base_lrs, self.gammas)]

class LinearWarmupCosineLR(LRScheduler):
    \"""
    Cosine LR with linear warmup and decay to some end LR.
    \"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.lr_end + (base_lr - self.lr_end) * (1 + math.cos(math.pi * (self.last_epoch - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps))) / 2 for base_lr in self.base_lrs]
"""


def merge(dico1, dico2):
    """
    Args:
        dico1 (dict) : It is a dictionary whose values are supposed to be of type List. Can be empty.
        dico2 (dict) : A dictionary with the same keys as dico1, and we want to merge dict2 in dict1.
    """
    if len(dico1) == 0:
        for key in dico2:
            dico1[key] = [dico2[key]]
    else:
        assert set(dico1.keys()) == set(dico2.keys())
        for key in dico1:
            dico1[key].append(dico2[key])
    return dico1


from copy import deepcopy
from typing import Iterator, List, Optional

from torch.utils.data import IterableDataset
from typing_extensions import Literal


class _HasNextIterator(Iterator):
    """Iterator with an hasnext() function. Taken from https://stackoverflow.com/questions/1966591/has-next-in-python-iterators."""

    def __init__(self, it):
        self.it = iter(it)
        self._hasnext = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._hasnext:
            result = self._thenext
        else:
            result = next(self.it)
        self._hasnext = None
        return result

    def hasnext(self):
        if self._hasnext is None:
            try:
                self._thenext = next(self.it)
            except StopIteration:
                self._hasnext = False
            else:
                self._hasnext = True
        return self._hasnext


class RandomlyCyclicChainDataset(IterableDataset):
    """
    Inspired by :
        - torch.utils.data.ChainDataset
        - Datasets.iterable_dataset.CyclingMultiSourcesExamplesIterable
        - Datasets.iterable_dataset.RandomlyCyclingMultiSourcesExamplesIterable
    """

    def __init__(
        self,
        datasets,
        generator: np.random.Generator,
        probabilities: Optional[List[float]] = None,
        stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.datasets = datasets
        self.stopping_strategy = stopping_strategy
        self.generator = deepcopy(generator)
        self.probabilities = probabilities

        # if undersampling ("first_exhausted"), we stop as soon as one dataset is exhausted
        # if oversampling ("all_exhausted"), we stop as soons as every dataset is exhausted, i.e as soon as every samples of every dataset has been visited at least once
        self.bool_strategy_func = np.all if (stopping_strategy == "all_exhausted") else np.any
        # TODO(QL): implement iter_arrow

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        random_batch_size=1000,
        p: Optional[List[float]] = None,
    ) -> Iterator[int]:
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        if p is None:
            while True:
                yield from (int(i) for i in rng.integers(0, num_sources, size=random_batch_size))
        else:
            while True:
                yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=p))

    def _get_indices_iterator(self):
        rng = deepcopy(self.generator)
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        return self._iter_random_indices(rng, len(self.datasets), p=self.probabilities)

    def __iter__(self):
        iterators = [_HasNextIterator(ex_iterable) for ex_iterable in self.datasets]

        indices_iterator = self._get_indices_iterator()

        is_exhausted = np.full(len(self.datasets), False)
        for i in indices_iterator:
            try:  # let's pick one example from the iterator at index i
                yield next(iterators[i])

                # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                if not iterators[i].hasnext():
                    is_exhausted[i] = True

                    if self.bool_strategy_func(is_exhausted):
                        # if the stopping criteria is met, break the main for loop
                        break
                    # otherwise reinitialise the iterator and yield the first example
                    iterators[i] = _HasNextIterator(self.datasets[i])

            except StopIteration:
                # here it means that the i-th iterabledataset is empty, i.e we never have the occasion to yield an element of the i-th dataset.
                # we still check if the stopping criteria is met and if we break out of the loop in case of an oversampling strategy
                is_exhausted[i] = True

                if self.bool_strategy_func(is_exhausted):
                    # if the stopping criteria is met, break the main for loop
                    break


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    probabilities: List[float],
):
    data_dirs = None

    if ".txt" in args.split_file or ".json" in args.split_file:
        with open(args.split_file, "r") as f:
            content = f.read()
        data_dirs = content.split("\n")
    elif ".csv" in args.split_file:
        with open(args.split_file) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                data_dirs = row
                break
    else:
        raise NotImplementedError(
            "The file containing the path to your data directories should have one of these extensions : .csv, .txt or .json."
        )

    if args.number_of_domains is not None:
        data_dirs = data_dirs[0 : args.number_of_domains]

    print("These are all the splits : " + str(data_dirs))

    if args.dataset_name_or_path:
        datasets = []
        for data_dir in data_dirs:
            import datasets as hf_db

            hf_db.config.DOWNLOADED_DATASETS_PATH = "/fsx/phuc/.cache"

            dataset = load_dataset(
                args.dataset_name_or_path,
                data_dir=data_dir,
                split="train",
                use_auth_token=True,
                num_proc=args.num_workers,
                streaming=False,
            )
            datasets.append(dataset)
    else:
        warnings.warn(
            """Your dataset in not stored on the datasets HUB. We are going to rely on JSON files. Each path in args.split_file should represent a valid path to a JSON/Parquet File with your examples in.
            """
        )
        datasets = []
        for data_dir in data_dirs:
            extension = data_dir[data_dir.rindex(".") + 1 :]
            dataset = load_dataset(extension, data_files=data_dir, streaming=False)
            datasets.append(dataset)

    if args.validation_dataset_path:
        valid_datasets = []
        for data_dir in data_dirs:
            dataset = load_dataset(
                args.validation_dataset_path,
                data_dir=data_dir,
                split="test",
                use_auth_token=True,
                num_proc=args.num_workers,
                streaming=False,
            )
            valid_datasets.append(dataset)
        train_datasets = datasets
    else:
        warnings.warn(
            f"""You did not provide a validation dataset path. In this case, we are going to create it from your training dataset, with the proportion {args.valid_set_size}.
            """
        )
        dataset_sizes = [len(dataset) for dataset in datasets]
        dataset_proportion = [
            (dataset_size * args.valid_set_size) / sum(dataset_sizes) for dataset_size in dataset_sizes
        ]

        train_datasets = []
        valid_datasets = []

        for i, dataset in enumerate(datasets):
            test_size = dataset_proportion[i]
            dataset = dataset.train_test_split(test_size=test_size, shuffle=True)
            train_datasets.append(dataset["train"])
            valid_datasets.append(dataset["test"])

    ex_iterables_train = []
    ex_iterables_val = []
    for j in range(len(train_datasets)):
        train = train_datasets[j].with_format("torch")
        val = valid_datasets[j].with_format("torch")

        if args.max_eval_samples is not None:
            val = val.select(np.arange(min(args.max_eval_samples, len(val))))
        if args.max_train_samples is not None:
            train = train.select(np.arange(min(args.max_train_samples, len(train))))

        if args.packing:
            chars_per_token_train = chars_token_ratio(train, tokenizer, args.input_column_name)
            chars_per_token_val = chars_token_ratio(val, tokenizer, args.input_column_name)
            train = ConstantLengthDataset(
                tokenizer=tokenizer,
                dataset=train,
                infinite=True,
                seq_length=args.max_length,
                num_of_sequences=args.num_of_sequences,
                chars_per_token=chars_per_token_train,
                shuffle=True,
                input_column_name=args.input_column_name,
                dataset_index=j,
            )
            val = ConstantLengthDataset(
                tokenizer=tokenizer,
                dataset=val,
                infinite=False,
                seq_length=args.max_length,
                num_of_sequences=args.num_of_sequences,
                chars_per_token=chars_per_token_val,
                shuffle=True,
                input_column_name=args.input_column_name,
                dataset_index=j,
            )
            ex_iterables_train.append(train)
            ex_iterables_val.append(val)
        else:

            def tokenize(element):
                outputs = tokenizer(
                    element[args.input_column_name],
                    truncation=True,
                    padding=True,
                    max_length=args.max_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                )
                return {
                    "index": j,
                    "input_ids": outputs["input_ids"],
                    "attention_mask": outputs["attention_mask"],
                    "labels": outputs["input_ids"],
                }

            train = train.map(
                tokenize,
                batched=True,
                remove_columns=train.column_names,
                num_proc=args.num_workers,
                batch_size=256,
            )
            val = val.map(
                tokenize,
                batched=True,
                remove_columns=val.column_names,
                num_proc=args.num_workers,
                batch_size=256,
            )
            ex_iterables_val.append(val)
            if args.streaming:
                ex_iterables_train.append(train.to_iterable_dataset())
            else:
                ex_iterables_train.append(train)

    generator = np.random.default_rng(args.dataset_seed)

    train_dataset = RandomlyCyclicChainDataset(
        datasets=ex_iterables_train,
        generator=generator,
        probabilities=probabilities,
        stopping_strategy="all_exhausted",
    )

    valid_dataset = RandomlyCyclicChainDataset(
        datasets=ex_iterables_val,
        generator=generator,
        probabilities=probabilities,
        stopping_strategy="all_exhausted",
    )

    return train_dataset, valid_dataset


from dataclasses import dataclass

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@dataclass
class DoReMiOutput(CausalLMOutputWithCrossAttentions):
    """
    loss, logits, hidden_states, attentions, cross_attentions, past_key_values
    """

    per_domain_loss: torch.FloatTensor = None

    def from_parent_instance(self, parent):
        self.loss = parent.loss
        self.logits = parent.logits
        self.hidden_states = parent.hidden_states
        self.attentions = parent.attentions
        self.cross_attentions = parent.cross_attentions
        self.past_key_values = parent.past_key_values


def _per_token_loss(inputs, logits):
    """
    https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    return a loss of size (batch size, sequence length)
    """
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fn = CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_logits.size(0), shift_logits.size(1))
    return loss


if __name__ == "__main__":
    args = DataTrainingArguments(
        dataset_name_or_path="ArmelR/the-pile-splitted",
        validation_dataset_path="ArmelR/the-pile-splitted",
        split_file="./pile.txt",
        # split_file="/fsx/phuc/.cache/fffdba1f8a000ef2024e31adaeb62adb88f06628ff886e3c9413665ec4fae0d5.json",
        input_column_name="text",
        valid_set_size=0.03,
        max_length=1024,
        num_workers=96,
        number_of_domains=22,
        # config_name="config.json",
        # tokenizer_name="bigcode-data/pile-1.3b",
        # number_of_rounds=2,
        # reference_model_name_or_path="ArmelR/doremi-280m",
        # step_size=1,
        # smoothing_parameter=1e-4,
        # train_batch_size=2,
        # valid_batch_size=2,
        # gradient_accumulation_steps=32,
        # max_train_steps=200000,
        # weight_decay=0.05,
        # learning_rate=1e-4,
        # lr_end=1e-4,
        # lr_scheduler_type="cosine",
        # num_warmup_steps=4000,
        # output_dir="./checkpoints-pile",
        # log_freq=100,
        # eval_freq=10000,
        # save_freq=10000,
        # run_name="DoReMi-pile",
        # report_to="wandb"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("stas/tiny-random-llama-2")
    probabilities = None
    train_dataset, valid_dataset = get_dataset(args, tokenizer, probabilities)

    assert 1 == 1
