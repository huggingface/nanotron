# Data Pipeline

## Data pre-processing

The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:

<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in preprocess_data.py The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap format use [`preprocess_data.py`](/tools/preprocess_data.py). An example script to prepare data for Llama2 training is:

<pre>
python tools/preprocess_data.py \
       --input my_corpus.json \
       --output-prefix my-llama2-dataset \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model /models/Llama-2-7b-chat-hf/tokenizer.model \
       --append-eod \
       --workers 6
</pre>

The output will be two files named, in this case, `my-llama2-dataset_text_document.bin` and `my-llama2-dataset_text_document.idx`. The `--data-path` specified later in training is the full path and new filename, but without the file extension.

----

Data preprocessing is built around the following classes:

1. `MMapIndexedDatasetBuilder`
2. `MMapIndexedDataset`

#### MMapIndexedDatasetBuilder

The `MMapIndexedDatasetBuilder` is capable of building and merging `MMapIndexedDataset` instances.

#### MMapIndexedDataset

The `MMapIndexedDataset` class is the lowest-level data interface. Internally, an `MMapIndexedDataset` instance references two binaries: the data file (`.bin`) contains document/sequence data and the index file (`.idx`) contains document/sequence metadata.

The index file stores dataset-level metadata first:
- The index header, for backward compatibility
- The index version, for backward compatibility
- A numeric code corresponding to the data type used to write data to the data file
- The number of sequences in the dataset
- The number of documents in the dataset

The index file stores document-level and sequence-level metadata second:
- In order, the number of elements per sequence
- In order, the byte offset (pointer) per sequence
- In order, the consecutive sequence index range `[...)` per document
- In order, the mode per sequence (in the multimodal case)

## Data loading: construction

Building the data loaders is a distributed-aware process built around the following classes:

1. `NanosetConfig`
2. `NanosetBuilder`
3. `MMapIndexedDataset`
3. `Nanoset`

See the class docstrings for more details.

#### NanosetConfig

The `NanosetConfig` class parameterizes the `NanosetBuilder` and in turn the `Nanoset`.

Different training/inference regimes will require different extensions e.g. the `NanosetConfig`

#### NanosetBuilder

The `NanosetBuilder` class builds the highest-level data interfaces.

**NB:** All ranks should attempt to build the dataset via the `NanosetBuilder` or the program will hang. Which ranks follow through on their attempts can be controlled via the `NanosetConfig`.

#### MMapIndexedDataset

The `MMapIndexedDataset` class is the lowest-level data interface.

The `MMapIndexedDataset` should already exist on disk before attempting to build any of the high-level data interfaces.


#### Nanoset

The `Nanoset` abstract class is a high-level data interface. It is built upon the `MMapIndexedDataset`.

Different training/inference regimes will require different extensions e.g. the `Nanoset`

## Data loading: implementation

### Nanoset

The `Nanoset` is parameterized by the following variables: the underlying `MMapIndexedDataset` instance `indexed_dataset`, the split indices `indexed_indices` (the congituous subset of document or sequence indices used for training, validation, and testing), the number of samples `N`, the sequence length `S`, and the random seed `R`.

The `Nanoset` creates three index mappings to facilitate lookup: (1) the document index, (2) the sample index, and (3) the shuffle index.

1. The document index _Do_idx_ is a 1-D array mapping from _i_ to document index of length `E * |indexed_indices|` where `E` corresponds to the minimum number of epochs such that `E * |indexed_indices| >= N`. The document index is shuffled according to `R`.

    ```
    Given:

    N = 15
    indexed_indices = [5, 6, 7, 8, 9]
    E = 3

    Then, for example:

    Do_idx = [8, 8, 9, 6, 7, 5, 8, 5, 6, 6, 5, 9, 7, 7, 9]
    ```

2. The sample index _Sa_idx_ is a 2-D array mapping from _j_ to pairs of (_i_, _Do_idx_[ _i_ ] offset) of shape `[N + 1, 2]`. The rows _j_ and _j_ + 1 serve as the left and right bounds for the _j_-th sample. 

    ```
    Given:

    S = 1024

    Then, for example:

    Sa_idx[0] = (0, 0)
    Sa_idx[1] = (0, 1024)       => Do_idx[0] has length greater than S
    Sa_idx[2] = (1, 512)        => Do_idx[0] has length 1536
    Sa_idx[3] = (2, 0)          => Do_idx[1] has length 1536
    Sa_idx[4] = (5, 300)        => Do_idx[2:5] are shorter documents relative to Do_idx[0:2]
    Sa_idx[5] = (6, 24)         => Do_idx[5] has length 1300
    ```

3. The shuffle index _Sh_idx_ is a 1-D array mapping from _k_ to _j_ of length `N`. The shuffle index is shuffled according to `R`.

    ```
    Given

    N = 10

    Then, for example:

    Sh_idx = [4, 0, 2, 6, 1, 9, 5, 8, 7, 3]
    ```

To query the `Nanoset` for the _k_-th sample we do the following

-  Use the shuffle index to get the index _j_ into the sample index.

    ```
    j = Sh_idx[k]
    ```
- Use the sample index to get the left and right sample-bounding indices into the document index and the starting token offset for each document.

    ```
    i, offset = Sa_idx[j]
    i_next, offset_next = Sa_idx[j + 1]
    ```
- Use the document index to retrieve `S` tokens from consecutive (in the document index) documents.

    ```
    sample = []
    sample += indexed_dataset[Do_idx[i]][offset:]
    if i != i_next:
        sample += indexed_dataset[Do_idx[i + 1:i_next]]
    sample += indexed_dataset[Do_idx[i_next]][:offset_next]
    ```

To save time during initialization, each index is built/cached sequentially on one process rank and subsequently loaded in parallel on other process ranks. The cached indices are unique to a hash generated in the `Nanoset.__init__` function.

