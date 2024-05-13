"""
Example:

torchrun --nproc-per-node 16 tools/preprocess_data.py \
       --input yelp_review_full \
       --split train \
       --output-prefix datasets/yelp_review_full \
       --tokenizer-name-or-path gpt2
       
torchrun --nproc-per-node 16 tools/preprocess_data.py \
       --input HuggingFaceH4/testing_alpaca_small \
       --split train \
       --column completion \
       --output-prefix datasets/testing_alpaca_small \
       --tokenizer-name-or-path gpt2
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import concatenate_datasets, load_dataset


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
        "--tokenizer-name-or-path",
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

    args = parser.parse_args()

    return args


def main(args):

    world_size, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])

    # Remove stdout from all processes except main to not flood the stdout
    if rank:
        sys.stdout = open(os.devnull, "w")

    # Check if output directory exists
    if not os.path.isdir(os.path.abspath(os.path.join(args.output_prefix, os.path.pardir))):
        print(f"Creating {os.path.abspath(os.path.join(args.output_prefix, os.path.pardir))} directory...")
        os.makedirs(os.path.abspath(os.path.join(args.output_prefix, os.path.pardir)), exist_ok=True)

    if args.input.endswith(".json"):  # For processing JSON files (Cross compatibility with other projects)
        ds = load_dataset("json", data_files=args.input)
        ds = concatenate_datasets(
            [ds[splits] for splits in ds.keys()]
        )  # load_dataset returns DatasetDict and we want a Dataset
    else:
        ds = load_dataset(args.input, split=args.split)

    ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)
    ds = ds.select_columns(args.column)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    token_dtype = np.int32 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else np.uint16

    # Create tmp directory for worker outputs
    tmp_folder = os.path.abspath(os.path.join(args.output_prefix, os.pardir, "tmp"))
    os.makedirs(tmp_folder, exist_ok=True)

    print("Creating workers output files...")
    worker_output_file = os.path.join(tmp_folder, f"worker_{rank}_input_ids.npy")
    ds = ds.map(
        lambda x: {"input_ids": tokenizer(x, add_special_tokens=args.add_special_tokens).input_ids},
        input_columns=args.column,
        batched=True,
        desc="Tokenizing Dataset",
        remove_columns=[args.column],
    )

    worker_input_ids_file = open(worker_output_file, "wb")
    for sample in ds:
        np_array = np.array(sample["input_ids"], dtype=token_dtype)
        worker_input_ids_file.write(np_array.tobytes(order="C"))
    worker_input_ids_file.close()

    # Wait for all workers to process each shard of the Dataset
    dist.barrier()

    # Only the main rank merges the worker files
    if not rank:
        output_file = f"{args.output_prefix}_input_ids.npy"
        input_ids_file = open(output_file, "wb")
        for worker_idx in tqdm(range(world_size), desc="Merging workers output files"):
            worker_output_file = os.path.join(tmp_folder, f"worker_{worker_idx}_input_ids.npy")
            with open(worker_output_file, "rb") as f:
                shutil.copyfileobj(f, input_ids_file)
            os.remove(worker_output_file)

        input_ids_file.close()
        os.rmdir(tmp_folder)
        print(f"Done! {args.input} processed dataset stored in {output_file}")

    else:  # Close devnull stdout redirect
        sys.stdout.close()


if __name__ == "__main__":
    _args = get_args()
    dist.init_process_group(backend="gloo")
    main(_args)
