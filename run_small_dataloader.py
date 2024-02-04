import torch
from nanotron import distributed as dist
from nanotron.doremi.dataloader import CombinedDataset, DistributedSamplerForDoReMi
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.parallel import ParallelContext
from torch.utils.data import DataLoader

if __name__ == "__main__":
    DP_SIZE = 4

    from datasets import load_dataset

    dataset = load_dataset("stas/c4-en-10k", split="train")
    domain_weights = torch.tensor([0.6, 0.4])
    datasets = [dataset for _ in range(len(domain_weights))]

    parallel_context = ParallelContext(
        data_parallel_size=DP_SIZE,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )

    # global_batch_size = 512
    num_microbatches = 5
    # batch_size = global_batch_size // (num_microbatches * DP_SIZE)
    batch_size = 10

    # assert global_batch_size == num_microbatches * batch_size * DP_SIZE

    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)
    domain_keys = [f"domain {i}" for i in range(len(datasets))]
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )
    global_rank = dist.get_rank(parallel_context.world_pg)

    print(f"global_rank={global_rank}, num_samples_per_step: {sampler.num_samples_per_global_step}")

    comebined_dataset = CombinedDataset(datasets)

    dataloader = DataLoader(
        comebined_dataset,
        # batch_size=batch_size,
        sampler=sampler,
        # collate_fn=data_collator,
        # drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
        # num_workers=1,
        # pin_memory=True,
        # worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
    )

    # microbatch_idx = 0
    # yielded_idxs = []
    # for idxs in sampler:
    #     # NOTE: check that the indicies are not repeated
    #     assert not set(idxs).intersection(
    #         yielded_idxs
    #     ), f"microbatch_idx: {microbatch_idx}, yielded_idxs: {yielded_idxs}, idxs: {idxs}"

    #     microbatch_idx += 1
    #     yielded_idxs.extend(idxs)

    # iter_sampler = iter(sampler)
    epoch = 0
    yieled_idxs = []

    # def sanity(dataloader):
    #     for batch in dataloader:
    #         yield batch

    # dataloader = sanity(dataloader)
    # dataloader = iter(dataloader)

    step = 0
    for idxs in dataloader:
        # # idxs = (next(sampler) for _ in range(8))

        # # idxs = []
        # for _ in range(num_microbatches):
        #     _ = next(dataloader)

        # # NOTE: check not repeating idxs
        # # assert not set(idxs).intersection(yieled_idxs), f"epoch: {epoch}"

        # if epoch % 1000 == 0:
        #     print(f"rank: {dist.get_rank(parallel_context.dp_pg)}, epoch: {epoch} \n \n")

        # epoch += 1
        # # yieled_idxs.extend(idxs)

        # _ = next(dataloader)
        dist.barrier()
        if dist.get_rank(parallel_context.world_pg) == 0:
            print("\n\n\n\n ------------------- \n ")
            print(f"step = {step}, microbatch_idx = {sampler.microbatch_idx} \n")
            print(f"step = {step}, domain_counters = {sampler.domain_counters} \n")
            print(f"step = {step}, domain_batch_sizes = {sampler.domain_batch_sizes} \n")

            if step % num_microbatches == 0:
                if dp_rank == 0:
                    epoch = step / num_microbatches
                    print(f"################# epoch = {epoch} \n")

        dist.barrier()

        step += 1

        if step == 10:
            break
