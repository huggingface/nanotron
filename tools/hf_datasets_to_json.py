import argparse

from datasets import concatenate_datasets, load_dataset


def main(args):
    # Load all the Dataset
    if args.split is None:
        ds = load_dataset(args.dataset_path, num_proc=args.num_workers)
        ds = concatenate_datasets([ds[splits] for splits in ds.keys()])
    # Load a split of the Dataset
    else:
        ds = load_dataset(args.dataset_path, split=args.split, num_proc=args.num_workers)

    ds = ds.select_columns(args.column_name)

    if args.column_name not in ["text"]:
        print(f"Renaming {ds.column_names[0]} to 'text'")
        ds = ds.rename_column(ds.column_names[0], "text")

    # Store dataset to json file
    ds.to_json(path_or_buf=args.output_json, num_proc=args.num_workers)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to local stored dataset or repository on the HF hub"
    )
    parser.add_argument("--split", type=str, help="Which split of the data to process")
    parser.add_argument(
        "--column-name", type=str, required=True, help="Name of the column containing the data to process"
    )
    parser.add_argument("--output-json", type=str, required=True, help="Path to the json output file")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of processes to load the dataset and store the json file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    _args = get_args()
    main(_args)
