import argparse
import glob
import json
import multiprocessing
import os
import sys
import time

# TODO tj-solergibert: Keep this hack or not? Hack to be able to process data without installing nanotron.
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)), "src"))

from nanotron.data import indexed_dataset
from transformers import AutoTokenizer


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.encode(sentence)

                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))

            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eos_token_id)
                sentence_lens[-1] += 1

            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {count} documents", f"({count/elapsed} docs/s, {mbs} MB/s).", file=sys.stderr)

    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, "r", encoding="utf-8")

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}.bin".format(output_prefix, key)
            output_idx_files[key] = "{}_{}.idx".format(output_prefix, key)
            builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        builders[key].finalize(output_idx_files[key])


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON")
    group.add_argument(
        "--json-keys", nargs="+", default=["text"], help="space separate listed of keys to extract from json"
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

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


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {"partition": input_file_name, "output_prefix": output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


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
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # Check to see if paritions were already created
        partitions_present = check_files_exist(in_ss_out_names, "partition", args.partitions)

        if not partitions_present:
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
    partition = Partition(args, args.workers // args.partitions)

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
    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}.bin".format(args.output_prefix, key)
        output_idx_files[key] = "{}_{}.idx".format(args.output_prefix, key)
        builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name["output_prefix"]
            full_partition_output_prefix = "{}_{}".format(parition_output_prefix, key)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    _args = get_args()
    main(_args)
