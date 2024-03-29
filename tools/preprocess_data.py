import argparse
import multiprocessing
import os
import shutil

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from datasets import Dataset, concatenate_datasets, load_dataset


def preprocess_shard(
    dataset_shard: Dataset, output_file: str, tokenizer: PreTrainedTokenizerBase, column: str, add_special_tokens: bool
):
    dataset_shard = dataset_shard.map(
        lambda x: {"input_ids": tokenizer.encode(x, add_special_tokens=add_special_tokens)},
        input_columns=column,
        batched=False,
        remove_columns=[column],
    )
    input_ids_file = open(output_file, "wb")
    for sample in dataset_shard:
        np_array = np.array(sample["input_ids"], dtype=np.uint16)
        input_ids_file.write(np_array.tobytes(order="C"))
    input_ids_file.close()


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input", type=str, required=True, help="Path to local stored dataset or repository on the Hugging Face hub"
    )
    group.add_argument("--column", type=str, default="text", help="Column to preprocess from the Dataset")
    parser.add_argument("--split", type=str, default="train", help="Which split of the data to process")

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Whether or not to add special tokens when encoding the sequences. This will be passed to the Tokenizer",
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to the output processed dataset file")

    group = parser.add_argument_group(title="runtime")
    group.add_argument("--num-workers", type=int, default=8, help="Number of workers processing the dataset")

    args = parser.parse_args()

    return args


def main(args):

    # Check if output directory exists
    if not os.path.isdir(os.path.abspath(os.path.join(args.output_prefix, os.path.pardir))):
        print(f"Creating {os.path.abspath(os.path.join(args.output_prefix, os.path.pardir))} directory...")
        os.makedirs(os.path.abspath(os.path.join(args.output_prefix, os.path.pardir)), exist_ok=True)

    if args.input.endswith(".json"):  # For processing JSON files (Cross compatibility with other projects)
        ds = load_dataset("json", data_files=args.input)
        ds = concatenate_datasets([ds[splits] for splits in ds.keys()])  # Return DatasetDict
    else:
        ds = load_dataset(args.input, split=args.split, num_proc=args.num_workers)

    ds = ds.select_columns(args.column)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # Create tmp directory for worker outputs
    tmp_folder = os.path.abspath(os.path.join(args.output_prefix, os.pardir, "tmp"))
    os.makedirs(tmp_folder, exist_ok=True)
    workers_output_files = []
    processes = []

    print("Creating worker output files...")
    for worker in range(args.num_workers):
        worker_output_file = os.path.join(tmp_folder, f"worker_{worker}_input_ids.npy")
        workers_output_files.append(worker_output_file)

        p = multiprocessing.Process(
            target=preprocess_shard,
            args=(
                ds.shard(num_shards=args.num_workers, index=worker, contiguous=True),
                worker_output_file,
                tokenizer,
                args.column,
                args.add_special_tokens,
            ),
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Merging worker output files...")
    output_file = f"{args.output_prefix}_input_ids.npy"
    input_ids_file = open(output_file, "wb")
    for worker_output_file in workers_output_files:
        with open(worker_output_file, "rb") as f:
            shutil.copyfileobj(f, input_ids_file)
        os.remove(worker_output_file)

    input_ids_file.close()
    os.rmdir(tmp_folder)
    print(f"Done! {args.input} processed dataset stored in {output_file}")


if __name__ == "__main__":
    _args = get_args()
    main(_args)
