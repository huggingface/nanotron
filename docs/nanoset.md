# Nanosets
Nanotron incorporates [`Nanosets`](../src/nanotron/data/nanoset.py), a kind of datasets based on [numpy memory-mapped arrays](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html). `Nanosets` are capable of serving batches from files containing pre-tokenized datasets. They allow reading tokens from one or multiple datasets and even specifying the weight of each dataset when building batches.
## Install
To use `Nanosets`, it's necessary to install Nanotron with the `nanosets` flavor.
```
pip install -e '.[nanosets]'
```
This will install the following dependencies:
- `transformers`: To tokenize the datasets
- `datasets`: To preprocess the datasets
- `numba`: To compile helper functions in order to speed up the creation of `Nanosets`
## Data pre-processing
To use these datasets, first, we need to preprocess the data. The input format can either be a column of a Hugging Face Dataset or a .json file containing a text sample per line. For example:

<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The preprocessing is done using the [`tools/preprocess_data.py`](../tools/preprocess_data.py) script. Below we show an example for processing a corpus with the Llama2 tokenizer.

<pre>
python tools/preprocess_data.py \
       --input data/my_corpus.json \
       --output-prefix data/processed-datasets/my-llama2-dataset \
       --tokenizer-name-or-path meta-llama/Llama-2-7b-hf \
       --num-workers 128
</pre>

In `--tokenizer-name-or-path`, we will have to specify a tokenizer in the same way as we do when using `AutoTokenizers.from_pretrained(...)`.

The output will be one file named, in this case, `my-llama2-dataset_input_ids.npy`. We will then have to specify this file in the `dataset_path` field in the config file.

## Working with Nanosets

To work with `Nanosets`, we just need to configure 1 argument:
1. `dataset_path`: This argument specifies the file or files that will compose the `Nanoset`. There are 3 ways to specify it:
   1. If we specify a single path, we will create a `Nanoset` from a single dataset file.
    ```yaml
    data_stages:
      - name: General purpose training (Single dataset)
        start_training_step: 1
        data:
          dataset:
            dataset_path: nanosets/SlimPajama-6B_input_ids.npy
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
            dataset_path:
            - nanoset/SlimPajama-6B_input_ids.npy
            - nanoset/europarl_input_ids.npy
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
            dataset_path:
              nanoset/SlimPajama-6B_input_ids.npy: 0.8
              nanoset/europarl_input_ids.npy: 0.2
          num_loading_workers: 0
          seed: 1234
    ```

Finally, to use the `Nanosets`, launch the training with [`run_train.py`](../run_train.py).
```shell
torchrun --nproc-per-node 8 run_train.py --config configs/nanoset_llama2.yaml
```

## Under the hood
`Nanosets` are responsible of building samples of `sequence length + 1` tokens from the preprocessed dataset files. The `dataset lengths` of each dataset will be determined by the `number of total tokens / sequence length`, discarding the last sample since its length < `sequence length`.

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
