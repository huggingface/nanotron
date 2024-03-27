import argparse
import glob
import importlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

package = importlib.import_module("nanotron")
package_path = Path(package.__file__).parent.parent
sys.path.append(str(package_path))

from nanotron.data import indexed_dataset
from transformers import AutoTokenizer


class Encoder(object):
    def __init__(self, pretrained_model_name_or_path: str, json_key: str, append_eos: bool):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.json_key = json_key
        self.append_eos = append_eos

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

    def encode(self, json_line: str):
        data = json.loads(json_line)
        text = data[self.json_key]

        text_ids = Encoder.tokenizer.encode(text)
        if self.append_eos:
            text_ids.append(Encoder.tokenizer.eos_token_id)

        return text_ids, len(text_ids), len(json_line)


class Partition(object):
    def __init__(
        self, workers: int, log_interval: int, pretrained_model_name_or_path: str, json_key: str, append_eos: bool
    ):
        self.workers = workers
        self.log_interval = log_interval
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.json_key = json_key
        self.append_eos = append_eos

    def print_processing_stats(self, count: int, proc_start: float, total_bytes_processed: int):
        if count % self.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {count} documents", f"({count/elapsed} docs/s, {mbs} MB/s).", file=sys.stderr)

    def process_json_file(self, file_name: str):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, "r", encoding="utf-8")

        startup_start = time.time()
        encoder = Encoder(self.pretrained_model_name_or_path, self.json_key, self.append_eos)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        output_bin_file = "{}_{}.bin".format(output_prefix, self.json_key)
        output_idx_file = "{}_{}.idx".format(output_prefix, self.json_key)

        builder = indexed_dataset.MMapIndexedDatasetBuilder(
            output_bin_file, dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
        )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (text_ids, len_text_ids, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            builder.add_document([text_ids], [len_text_ids])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        builder.finalize(output_idx_file)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON")
    group.add_argument("--json-key", type=str, default="text", help="Key to extract from json")

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument("--append-eos", action="store_true", help="Append an <eos> token to the end of a sample.")

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix", type=str, required=True, help="Path to binary and index output files without suffix"
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers",
        type=int,
        required=True,
        help=(
            "Number of worker processes to launch."
            "A good default for fast pre-processing "
            "is: (workers * partitions) = available CPU cores."
        ),
    )
    group.add_argument("--partitions", type=int, default=1, help="Number of file partitions")
    group.add_argument("--log-interval", type=int, default=1000, help="Interval between progress updates")

    args = parser.parse_args()

    return args


def get_file_name(input: str, output_prefix: str, file_id: int):
    file_name, extension = os.path.splitext(input)
    input_file_name = file_name + "_" + str(file_id) + extension
    output_prefix = output_prefix + "_" + str(file_id)
    file_names = {"partition": input_file_name, "output_prefix": output_prefix}
    return file_names


def main(args):
    # Check if json file is not empty
    assert os.path.getsize(args.input), f"{args.input} is empty!"
    # Check if output directory exists
    if not os.path.isdir(os.path.abspath(os.path.join(args.output_prefix, os.path.pardir))):
        print(f"Creating {os.path.abspath(os.path.join(args.output_prefix, os.path.pardir))} directory...")
        os.makedirs(os.path.abspath(os.path.join(args.output_prefix, os.path.pardir)), exist_ok=True)

    in_ss_out_names = []
    if args.partitions == 1:
        file_names = {
            "partition": args.input,
            "output_prefix": args.output_prefix,
        }
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)

        # Create .jsonl partition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args.input, args.output_prefix, idx)
            in_ss_out_names.append(in_ss_out_name)

        # Populate .jsonl partition files from parent files
        partitioned_input_files = []
        for idx in range(args.partitions):
            partitioned_input_file = open(in_ss_out_names[idx]["partition"], "w")
            partitioned_input_files.append(partitioned_input_file)

        index = 0
        for in_file_name in in_file_names:
            fin = open(in_file_name, "r", encoding="utf-8")

            for line in fin:
                partitioned_input_files[index].write(line)
                index = (index + 1) % args.partitions
            fin.close()

        for idx in range(args.partitions):
            partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(
        args.workers // args.partitions,
        args.log_interval,
        args.pretrained_model_name_or_path,
        args.json_key,
        args.append_eos,
    )

    # Encode partition files in parallel
    processes = []
    input_key = "partition"
    for name in in_ss_out_names:
        p = multiprocessing.Process(
            target=partition.process_json_file, args=((name[input_key], name["output_prefix"]),)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if args.partitions == 1:
        return

    # Merge bin/idx partitions
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    output_bin_file = "{}_{}.bin".format(args.output_prefix, args.json_key)
    output_idx_file = "{}_{}.idx".format(args.output_prefix, args.json_key)

    builder = indexed_dataset.MMapIndexedDatasetBuilder(
        output_bin_file, dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
    )

    for name in in_ss_out_names:
        parition_output_prefix = name["output_prefix"]
        full_partition_output_prefix = "{}_{}".format(parition_output_prefix, args.json_key)
        builder.add_index(full_partition_output_prefix)
    builder.finalize(output_idx_file)

    # Clean temporary files
    for name in in_ss_out_names:
        os.remove(name["partition"])
        for output in glob.glob(name["output_prefix"] + "*"):
            os.remove(output)


if __name__ == "__main__":
    _args = get_args()
    main(_args)
