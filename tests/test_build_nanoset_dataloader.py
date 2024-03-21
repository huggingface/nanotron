import pytest
from helpers.context import TestContext
from helpers.data import (
    assert_batch_dataloader,
    create_dataset_paths,
    create_dummy_json_dataset,
    preprocess_dummy_dataset,
)
from helpers.utils import available_gpus, get_all_3d_configurations, init_distributed, rerun_if_address_is_in_use
from nanotron.data.blended_nanoset import BlendedNanoset
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.data.dataset_builder import NanosetBuilder
from nanotron.data.nanoset import Nanoset
from nanotron.data.nanoset_configs import NanosetConfig
from nanotron.data.utils import compile_helpers, compute_datasets_num_samples
from nanotron.parallel import ParallelContext


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_build_nanoset_dataloader(tp: int, dp: int, pp: int):
    test_context = TestContext()

    # Compile helpers & Create dataset files
    compile_helpers()

    json_paths, bin_paths = create_dataset_paths(tmp_dir=test_context.get_auto_remove_tmp_dir(), quantity=2)

    # Create dummy json datasets
    for idx, json_path in enumerate(json_paths):
        create_dummy_json_dataset(path_to_json=json_path, dummy_text=f"Nanoset {idx}!")

    # Preprocess dummy json datasets
    for json_path in json_paths:
        preprocess_dummy_dataset(path_to_json=json_path)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_build_nanoset_dataloader)(
        test_context=test_context, path_to_bin_files=bin_paths
    )


def _test_build_nanoset_dataloader(
    parallel_context: ParallelContext, test_context: TestContext, path_to_bin_files: str
):
    TRAIN_STEPS = 10000
    VAL_CHECK_INTERVAL = TRAIN_STEPS / 4
    VAL_STEPS = TRAIN_STEPS / 10
    MICRO_BATCH_SIZE = 4
    N_MICRO_BATCHES_PER_BATCH = 8
    GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * N_MICRO_BATCHES_PER_BATCH * parallel_context.dp_pg.size()

    SEED = 1234
    SEQ_LENGTH = 8192
    SPLIT = "950,40,10"

    input_pp_rank, output_pp_rank = 0, int(parallel_context.pp_pg.size() - 1)

    # Create Nanoset configs: 1. Normal 2. Blended
    split_num_samples = compute_datasets_num_samples(
        train_iters=TRAIN_STEPS,
        eval_interval=VAL_CHECK_INTERVAL,
        eval_iters=VAL_STEPS,
        global_batch_size=GLOBAL_BATCH_SIZE,
    )

    nanoset_config = NanosetConfig(
        random_seed=SEED,
        sequence_length=SEQ_LENGTH,
        data_path=path_to_bin_files[0],
        split=SPLIT,
        split_num_samples=split_num_samples,
        path_to_cache=test_context.get_auto_remove_tmp_dir(),
    )

    blended_nanoset_config = NanosetConfig(
        random_seed=SEED,
        sequence_length=SEQ_LENGTH,
        data_path={bin_path: 5 - idx for idx, bin_path in enumerate(path_to_bin_files)},
        split=SPLIT,
        split_num_samples=split_num_samples,
        path_to_cache=test_context.get_auto_remove_tmp_dir(),
    )

    configs = [nanoset_config, blended_nanoset_config]
    dataset_types = [Nanoset, BlendedNanoset]

    for config, dataset_type in zip(configs, dataset_types):
        # Create Nanosets
        train_dataset, valid_dataset, test_dataset = NanosetBuilder(config).build()
        # Check the type of Nanoset and the quantity of Nanosets in BlendedNanoset
        assert isinstance(train_dataset, dataset_type)
        if isinstance(train_dataset, BlendedNanoset):
            assert len(train_dataset.datasets) > 1

        # Create Dataloaders
        dataloader = build_nanoset_dataloader(
            train_dataset,
            sequence_length=SEQ_LENGTH,
            parallel_context=parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=MICRO_BATCH_SIZE,
            dataloader_num_workers=0,
            seed_worker=SEED,
            dataloader_drop_last=True,
        )

        # Check a batch produced by the Dataloader
        batch = next(iter(dataloader))
        assert_batch_dataloader(batch=batch, parallel_context=parallel_context)

    parallel_context.destroy()
