# Nanosets

## Install
To use `Nanosets`, it's necessary to install Nanotron with the `nanosets` flavor.
```
pip install -e '.[nanosets]'
```

## Data pre-processing

Nanotron incorporates [`Nanosets`](../src/nanotron/data/nanoset.py), a kind of datasets based on numpy memory-mapped arrays. Permite utilizar tanto un unico dataset como varios, incluso especificando los weights de cada dataset.


To use these datasets, first, we need to preprocess the data. The input format can either be a column of a Hugging Face Dataset or a .json file containing a text sample per line. For example:

<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The dataset is then processed into a mmap format for training using the [`tools/preprocess_data.py`](../tools/preprocess_data.py) script. Below we show an example for processing a corpus with the Llama2 tokenizer.

<pre>
python tools/preprocess_data.py \
       --input data/my_corpus.json \
       --output-prefix data/processed-datasets/my-llama2-dataset \
       --tokenizer-name-or-path meta-llama/Llama-2-7b-hf \
       --num-workers 128
</pre>

In `--tokenizer-name-or-path`, we will have to specify a tokenizer in the same way as we do when using `AutoTokenizers.from_pretrained(...)`.

The output will be one file named, in this case, `my-llama2-dataset_input_ids.npy`. We will then have to specify this file in the `data_path` field in the config file.

## Working with Nanosets

To work with Nanosets, we just need to configure 1 argument:
1. `data_path`: This argument specifies the file/files that will compose the `Nanoset`. There are 3 ways to specify it:
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

Finally, to use the Nanosets, launch the training with [`run_train.py`](../run_train.py).
```shell
torchrun --nproc-per-node 8 run_train.py --config configs/nanoset_llama2.yaml
```

## Under the hood
### Number of samples

When using Nanosets, we specify the `data_path` to the preprocessed dataset which contains all the tokens. The number of samples will be the `number of tokens / sequence lenght`.

For the train split, the number of samples consumed from the Nanoset will be determined by the `number of train steps * global batch size`, so if this number is higher than the number of samples in the train split, we will see the dataset samples more than once (> 1 epoch). In the case of the valid and test split, we will see all the samples only once.

In the case of the `BlendedNanoset`, we will also indicate the weight of each dataset to construct data batches according to the specified proportion. In this case, the train split will respect this proportion, considering that the number of samples will be computed in the same way as in the `Nanosets`, so it may happen that we consume one dataset for 3 epochs and another larger dataset for only one epoch. For the valid and test splits, the same as in the `Nanosets` will occur; we will consume all the samples only once.

### Nanoset
A `Nanoset` is paremeterized by the following variables:
- The underlying `MMapIndexedDataset` instance (`indexed_dataset`)
- The sequence length `S`
- The split indices `indexed_indices` (the congituous subset of sample indices used for training, validation, and testing)
- The total number of samples `N` of the Nanoset that we will consume during training. In the case of the valid and test splits, we will only consume the dataset once
- The random seed `R`

The `Nanoset` creates a single index (`shuffle_index`) to map the indices of the Nanoset (0, 1, 2, ... `N`) to the indices of the `MMapIndexedDataset` for the specific split (`indexed_indices`).

In the train split, the shuffle index (`shuffle_index`) is a 1-D array mapping from _k_ to _j_ of length `n_concatenations * len(indexed_indices)`, where `n_concatenations` is defined as `(N / len(indexed_indices)) + 1`, so that `len(shuffle_index)` is always greater than `N`. While for the valid and test splits, `len(shuffle_index) == len(indexed_indices)`. Before concatenating the full array, `shuffle_index` is shuffled according to `R`.
```
Given:

N = 70

indexed_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

Then, for example:

shuffle_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

Shuffle the indices -> shuffle_index = [19, 9, 3, 18, 15, 12, 5, 10, 17, 1, 4, 8, 11, 16, 13, 7, 2, 14, 6, 0]

n_concatenations = (70/(20)) + 1 = 4
shuffle_index = shuffle_index concatenated 4 times

len(shuffle_index) = 80 > N
```

To query the `Nanoset` for the k-th sample we do the following:
1. Use the `shuffle_index` to get the index _j_ into the `sample_index`
```
j = shuffle_index[k]
```
2. To retrieve `S + 1` tokens from the `indexed_dataset` we have to specify the `offset` (`idx * sequence length` for CausalLM) and the length (the number of tokens to extract)
```
offset = j * sequence_length
sample = indexed_dataset[offset:offset + sequence_length + 1]
```

Despite having repeated indices in the `shuffle_index`, throughout 1 epoch, we will only observe each sample once. We achieve this by deactivating shuffling in the `DistributedDataSampler`, so that the indices of the `shuffle_index` are consumed in the order they appear by the multiple processes. It is worth noting that the samples are already shuffled in the `shuffle_index`.
```
Given:

4 Processes loading data

[19, 9, 3, 18, 15, 12, 5, 10, 17, 1, 4, 8, 11, 16, 13, 7, 2, 14, 6, 0]

(P1) idx_list = [0, 4, 8, 12, 16, ...]    -> shuffle_index[idx_list] = [19, 15, 17, 11, 2, 19, ...]
(P2) idx_list = [1, 5, 9, 13, 17, ...]    -> shuffle_index[idx_list] = [9, 12, 1, 16, 14, 9, ...]
(P3) idx_list = [2, 6, 10, 14, 18, ...]   -> shuffle_index[idx_list] = [3, 5, 4, 13, 6, 3, ...]
(P4) idx_list = [3, 7, 11, 15, 19, ...]   -> shuffle_index[idx_list] = [18, 10, 8, 7, 0, 18, ...]
```
### BlendedNanoset
The `BlendedNanoset` is parameterized by the following variables:
- The underlying `Nanoset` instances `D`
- The weights `W` (one per dataset)
- The number of samples `U`

The `BlendedNanoset` creates two "blending" indices to facilitate lookup: (1) The `dataset_index` and (2) the `dataset_sample_index`.

1. The `dataset_index` is a 1-D array mapping from _i_ to dataset index from `D` of length `U`.
```
Given:

D = [d0, d1, d2, d3]
W = [0.1, 0.5, 0.3, 0.1]
U = 20

Then, for example:

dataset_index = [1, 2, 0, 1, 3, 1, 2, 1, 2, 1, 0, 1, 2, 1, 3, 1, 2, 1, 2, 1]
```
2. The `dataset_sample_index` is a 1-D mapping from _i_ to the sample index for dataset_index[_i_] of length `U`.
```
dataset_index =         [1, 2, 0, 1, 3, 1, 2, 1, 2, 1, 0, 1, 2, 1, 3, 1, 2, 1, 2, 1]
dataset_sample_index =  [0, 0, 0, 1, 0, 2, 1, 3, 2, 4, 1, 5, 3, 6, 1, 7, 4, 8, 5, 9]
```
To query the `BlendedNanoset` for the k-th sample we do the following:
- Use the `dataset_index` to retrieve the corresponding dataset from `D` and the `dataset_sample_index` to retrieve the corresponding sample from that dataset.
```
sample = D[dataset_index[k]][dataset_sample_index[k]]
```
