import argparse

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import DocumentTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Dataset reader")
    group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to local stored dataset or repository on the Hugging Face hub that can be loaded with datasets.load_dataset",
    )
    group.add_argument(
        "--column", type=str, default="text", help="Column to preprocess from the Dataset. Default: text"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Which split of the data to process. Default: train"
    )

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS token to add after each document. Default: None",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder", type=str, required=True, help="Path to the output folder to store the tokenized documents"
    )
    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Path to a folder for storing the logs of the preprocessing step. Default: None",
    )
    group.add_argument(
        "--n-tasks", type=int, default=8, help="Total number of tasks to run the preprocessing step. Default: 8"
    )

    args = parser.parse_args()

    return args


def main(args):

    preprocess_executor = LocalPipelineExecutor(
        pipeline=[
            HuggingFaceDatasetReader(
                dataset=args.dataset,
                text_key=args.column,
                dataset_options={"split": args.split},
            ),
            DocumentTokenizer(
                output_folder=args.output_folder,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                eos_token=args.eos_token,
            ),
        ],
        tasks=args.n_tasks,
        logging_dir=args.logging_dir,
    )
    preprocess_executor.run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)
