# Nanosets
Nanotron incorporates [`Nanosets`](../src/nanotron/data/nanoset.py), a dataset for processing tokenized documents with [`datatrove`](https://github.com/huggingface/datatrove). They allow reading tokens from one or multiple datasets and even specifying the weight of each dataset when building batches.
## Install
To use `Nanosets`, it's necessary to install Nanotron with the `nanosets` flavor.
```
pip install nanotron[nanosets]
```
This will install the following dependencies:
- `datatrove`: To preprocess the datasets
- `numba`: To compile helper functions in order to speed up the creation of `Nanosets`
- `transformers`: For the tokenizers
## Data pre-processing
To use this dataset, first, we need to preprocess the data using `datatrove`'s `DocumentTokenizer` pipeline. We invite you to take a look at `datatrove`, since it contains multiple features that allow, for example, filter out documents based on specific rules/criteria, extract text content from raw formats or scheduling the preprocessing in a Slurm cluster. We have also added a simple script capable of tokenizing datasets.

The preprocessing is done using the [`tools/preprocess_data.py`](../tools/preprocess_data.py) script. The input format can either be a Hugging Face Dataset, a path to a `.jsonl` or a path to a folder containing multiple `.jsonl` files. Below we show an example for processing a Hugging Face Dataset from the Hub with the Llama3 tokenizer.

<pre>
python3 tools/preprocess_data.py \
       --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B \
       --output-folder datasets/emotion \
       --n-tasks 16 \
       hf \
       --dataset dair-ai/emotion \
</pre>

First with `--tokenizer-name-or-path` we will specify a tokenizer in the same way as we do when using `AutoTokenizers.from_pretrained(...)`. Then we specify the `--output-folder` where we will store the tokenized documents and the number of workers with `--n-tasks`. Finally we will indicate the type of dataset (whether if it's a Hugging Face Dataset ["**hf**"] or in jsonl ["**jsonl**"] format) and the dataset that we want to preprocess. Check the different settings with `python3 tools/preprocess_data.py --help`, `python3 tools/preprocess_data.py hf --help` & `python3 tools/preprocess_data.py jsonl --help`.

Every worker will store in `--output-folder` 3 different kind of files:
- `*.ds` Containing the tokenized documents
- `*.ds.index` Containing the bounds of each tokenized document
- `*.ds.metadata` Containing the number of tokens and tokenizer used

> [!IMPORTANT]
Remember to introduce the type of dataset to process. e.g. python3 tools/preprocess_data.py --tokenizer-name-or-path gpt2 --n-tasks 16 **jsonl** --dataset raw_datasets/c4-es-json-files

## Working with Nanosets

To work with `Nanosets`, we just need to configure 1 argument:
1. `dataset_folder`: This argument specifies the file or files that will compose the `Nanoset`. There are 3 ways to specify it:
   1. If we specify a single path, we will create a `Nanoset` from a single dataset file.
    ```yaml
    data_stages:
      - name: General purpose training (Single dataset)
        start_training_step: 1
        data:
          dataset:
            dataset_folder: datasets/SlimPajama-6B
          num_loading_workers: 0
          seed: 1234
    ```
   2. If we specify a list of paths, we will create a `Nanoset` from all the dataset files. In every epoch we will consume each and every sample from each dataset randomly.
    ```yaml
    data_stages:
      - name: Second purpose training (> 1 dataset)
        start_training_step: 15
        data:
          dataset:
            dataset_folder:
            - datasets/SlimPajama-6B
            - datasets/testing_alpaca_small
          num_loading_workers: 0
          seed: 1234
    ```
    3. If we specify a dictionary with paths and weights, we will create a `Nanoset` from the dataset files where each epoch will have a number of samples from each dataset according to the specified weights.
    ```yaml
    data_stages:
      - name: Third purpose training (Blended dataset)
        start_training_step: 25
        data:
          dataset:
            dataset_folder:
              datasets/SlimPajama-6B: 0.8
              datasets/testing_alpaca_small: 0.2
          num_loading_workers: 0
          seed: 1234
    ```
> [!IMPORTANT]
> Remember to set the `tokenizer.tokenizer_name_or_path` in the config file to the tokenizer used to preprocess the documents and set the `model.model_config.vocab_size` accordingly.

Finally, to use the `Nanosets`, launch the training with [`run_train.py`](../run_train.py).
```shell
torchrun --nproc-per-node 1 run_train.py --config examples/config_nanoset.yaml
```

## Under the hood
`Nanosets` are responsible of building samples of `sequence length + 1` tokens from the preprocessed dataset files. Despite most of the extracting logic lies in `DatatroveFolderDataset`, `Nanosets` will take care of the following:
1. Creating dataset mixtures from different dataset folder paths
2. Ensure that in each epoch, we consume each sample only once
3. Ensure that we never exhaust the `DataLoader`

Based on the `dataset lengths`, the `dataset weights` and the `number of samples per epoch` (defined as the `sum(dataset lengths)`), we build the two indexes we need in order to extract samples from the `Nanoset`  ([build_nanoset_index_helper](../src/nanotron/data/nanoset.py)):
- `dataset index`: Contains the index of the dataset from the list of `dataset paths` from which to extract the sample, respecting the established dataset weight.
```
Given:

D = [d0, d1, d2, d3]        # datasets
DL = [8, 2, 5, 5]           # dataset lengths
W = [0.1, 0.5, 0.3, 0.1]    # dataset weights
SPE = 20                    # number of samples per epoch

Then, for example:

dataset_index = [1, 2, 0, 1, 3, 1, 2, 1, 2, 1, 0, 1, 2, 1, 3, 1, 2, 1, 2, 1]
```
- `dataset sample index`: Contains the sample index to extract from the `dataset index[index]` dataset, always < `len(dataset)`.
```
dataset_index =         [1, 2, 0, 1, 3, 1, 2, 1, 2, 1, 0, 1, 2, 1, 3, 1, 2, 1, 2, 1]
dataset_sample_index =  [0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 1, 1, 3, 0, 1, 1, 4, 0, 0, 1]
```
Then, we **shuffle with the same permutation both indexes** and concatenate them `number of epochs` times, which is defined by `train split num samples` / `number of samples per epoch`.
```
Given:

N = 70                      # train split num samples

dataset_index =         [1, 2, 0, 1, 3, 1, 2, 1, 2, 1, 0, 1, 2, 1, 3, 1, 2, 1, 2, 1]
dataset_sample_index =  [0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 1, 1, 3, 0, 1, 1, 4, 0, 0, 1]

Shuffle dataset_index and dataset_sample_index:

dataset_index =         [1, 1, 0, 2, 3, 1, 3, 1, 2, 2, 1, 1, 0, 1, 1, 2, 1, 2, 2, 1]
dataset_sample_index =  [1, 0, 0, 4, 1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1, 0, 1, 3, 1, 1]

n_concatenations = (70/(20)) + 1 = 4
dataset_index = dataset_index concatenated 4 times
dataset_sample_index = dataset_sample_index concatenated 4 times

dataset_index = dataset_index[: N]
dataset_sample_index = dataset_sample_index[: N]
```
To query the `Nanoset` for the k-th sample we do the following:
- Use the `dataset_index` to retrieve the corresponding dataset from `D` and the `dataset_sample_index` to retrieve the corresponding sample from that dataset.
```
sample = D[dataset_index[k]][dataset_sample_index[k]]
```
