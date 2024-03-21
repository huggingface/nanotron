import os

import pytest
import torch
from helpers.context import TestContext
from helpers.data import assert_batch_dataloader, create_dummy_json_dataset, preprocess_dummy_dataset
from helpers.utils import get_all_3d_configurations, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.data.blended_nanoset import BlendedNanoset
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.data.dataset_builder import NanosetBuilder
from nanotron.data.nanoset import Nanoset
from nanotron.data.nanoset_configs import NanosetConfig
from nanotron.data.utils import compute_datasets_num_samples
from nanotron.parallel import ParallelContext


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(12, 8) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_build_nanoset_dataloader(tp: int, dp: int, pp: int):
    test_context = TestContext()
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_build_nanoset_dataloader)(test_context=test_context)


def _test_build_nanoset_dataloader(parallel_context: ParallelContext, test_context: TestContext):
    TRAIN_STEPS = 10000
    VAL_CHECK_INTERVAL = TRAIN_STEPS / 4
    VAL_STEPS = TRAIN_STEPS / 10
    MICRO_BATCH_SIZE = 4
    N_MICRO_BATCHES_PER_BATCH = 8
    GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * N_MICRO_BATCHES_PER_BATCH * parallel_context.dp_pg.size()

    SEED = 1234
    SEQ_LENGTH = 8192
    SPLIT = "8,1,1"

    dataset_1_json_path = os.path.join(test_context.get_auto_remove_tmp_dir(), "pytest_1")
    dataset_2_json_path = os.path.join(test_context.get_auto_remove_tmp_dir(), "pytest_2")
    dataset_1_bin_path = dataset_1_json_path + "_text"
    dataset_2_bin_path = dataset_2_json_path + "_text"

    # Compile helpers & Create dataset files
    if dist.get_rank() == 0:
        from nanotron.data.utils import compile_helpers

        compile_helpers()

        # Create dummy json datasets
        create_dummy_json_dataset(path_to_json=dataset_1_json_path, dummy_text="Nanoset 1!")
        create_dummy_json_dataset(path_to_json=dataset_2_json_path, dummy_text="Nanoset 2!")

        # Preprocess dummy json datasets
        preprocess_dummy_dataset(path_to_json=dataset_1_json_path)
        preprocess_dummy_dataset(path_to_json=dataset_2_json_path)

    torch.distributed.barrier()

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
        data_path=dataset_1_bin_path,
        split=SPLIT,
        split_num_samples=split_num_samples,
        path_to_cache=test_context.get_auto_remove_tmp_dir(),
    )

    blended_nanoset_config = NanosetConfig(
        random_seed=SEED,
        sequence_length=SEQ_LENGTH,
        data_path={dataset_1_bin_path: 0.2, dataset_2_bin_path: 0.2},
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
