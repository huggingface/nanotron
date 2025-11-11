# Using Megatron `IndexedDataset` With `run_train.py`

These steps explain how to plug NeMo/Megatron `*.bin` + `*.idx` datasets into Nanotron’s `run_train.py`, so you can reuse your existing `IndexedDataset` shards instead of Datatrove `.ds` chunks.

---

## 1. Build or reuse the indexed shards
1. Convert your corpus into Megatron format using the helper in `nanotron/data/nemo_dataset/indexed_dataset.py`:
   ```python
   from nanotron.data.nemo_dataset.indexed_dataset import make_indexed_dataset
   make_indexed_dataset(
       data_prefix="/data/my_corpus",   # produces /data/my_corpus.bin + .idx
       skip_warmup=False
   )
   ```
2. Compile the C++ helpers that power the dataset utilities:
   ```bash
   make -C src/nanotron/data/nemo_dataset
   ```
   (`compile_helper()` in `nanotron/data/nemo_dataset/dataset_utils.py:24` calls this at run time, but pre-building avoids stalls.)

You can repeat the builder for every source you plan to blend (e.g., code, math, multilingual web).

---

## 2. Add an `IndexedDatasetsArgs` dataclass
Edit `src/nanotron/config/config.py` and add a new dataclass next to `PretrainDatasetsArgs` and `NanosetDatasetsArgs`:

```python
@dataclass
class IndexedDatasetsArgs:
    data_prefix: Union[List[str], Dict[str, List[str]]]
    splits_string: str = "969,30,1"
    validation_drop_last: bool = True
    eod_mask_loss: bool = False
    no_seqlen_plus_one_input_tokens: bool = False
    index_mapping_dir: Optional[str] = None
    fim_rate: float = 0.0
    fim_spm_rate: float = 0.0
    fim_split_sample: Optional[str] = None
    fragment_fim_rate: float = 0.0
    no_fim_prefix: bool = False
    skip_warmup: bool = False
```

- Update the `Union[...]` types (`DataArgs.dataset`, `TrainerConfig.data_stages`, etc.) so the config parser accepts the new dataclass.
- In the `__post_init__` block around line `507`, reuse the existing tokenizer/vocab checks for this type just like `NanosetDatasetsArgs` already does.

---

## 3. Teach `run_train.py` about indexed datasets
Inside `get_dataloader_from_data_stage` (`run_train.py:80-230`) insert a new branch before the TokenizedBytes block:

```python
elif isinstance(data.dataset, IndexedDatasetsArgs):
    from nanotron.data.nemo_dataset import build_dataset

    tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_dataset = build_dataset(
        cfg=data.dataset,
        tokenizer=tokenizer,
        data_prefix=data.dataset.data_prefix,
        num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
        seq_length=trainer.sequence_length,
        seed=data.seed,
        skip_warmup=data.dataset.skip_warmup,
        name="train",
        parallel_context=trainer.parallel_context,
    )

    dataloader = get_train_dataloader(
        train_dataset=train_dataset,
        sequence_length=trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        consumed_train_samples_stage=consumed_train_samples_stage,
        dataloader_num_workers=data.num_loading_workers,
        seed_worker=data.seed,
        dataloader_drop_last=True,
        use_position_ids=isinstance(trainer.model_config, Qwen2Config),
    )
```

Notes:
- If you need validation/test splits, call `build_train_valid_test_datasets` instead (same module).
- The builder returns a `BlendableDataset` automatically when `data_prefix` contains weight/prefix pairs, so the downstream `get_train_dataloader` logic works unchanged.

---

## 4. Describe the dataset stage in YAML
With the dataclass and loader in place, you can define a stage that reads indexed data:

```yaml
data_stages:
  - name: indexed_stage
    start_training_step: 1
    data:
      dataset:
        _target_: IndexedDatasetsArgs   # if you rely on OmegaConf instantiation
        data_prefix:
          - 0.6
          - /data/fineweb/fw_edu
          - 0.4
          - /data/code/stack_mix   # expects stack_mix.bin / stack_mix.idx
        splits_string: "999,1,0"
        validation_drop_last: true
        eod_mask_loss: false
        index_mapping_dir: /scratch/nanotron_index_cache
        fim_rate: 0.0
        skip_warmup: false
      num_loading_workers: 8
      seed: 6
```

If you are not using `_target_`, make sure your config loader constructs `IndexedDatasetsArgs` explicitly before passing the object into Nanotron.

---

## 5. Keep metadata in sync
- Checkpoint resumes rely on token accounting per dataset folder. Update the helper functions in `nanotron/helpers.py` if they currently assume every stage is `NanosetDatasetsArgs`.
- If you enable FIM or change `sequence_length`, ensure the indexed corpus was tokenized with the exact same tokenizer (`trainer.config.tokenizer`) so the vocabulary matches (the sanity checks around `run_train.py:143` will catch mismatches).

Once these steps are in place, `run_train.py` can seamlessly stream Megatron-style `IndexedDataset`s alongside the existing Datatrove TokenizedBytes loader.
