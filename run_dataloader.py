import torch
from datasets import load_from_disk
from nanotron import distributed as dist
from nanotron.dataloader import get_dataloader_worker_init
from nanotron.doremi.dataloader import CombinedDataset, DistributedSamplerForDoReMi
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.parallel import ParallelContext
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    DP_SIZE = 16
    # # domain_weights = torch.tensor(
    # #     [
    # #         0.34356916553540745,
    # #         # 0.16838812972610234,
    # #         # 0.24711766854236725,
    # #         # 0.0679225638705455,
    # #         # 0.059079828519653675,
    # #         # 0.043720261601881555,
    # #         # 0.01653850841342608,
    # #         # 0.00604146633842096,
    # #         # 0.04342813428189645,
    # #         # 0.0041942731702987,
    # #     ]
    # # )
    # domain_weights = torch.tensor([0.6, 0.4])

    # dataset1 = load_dataset("stas/c4-en-10k", split="train[:100]")
    # datasets = [dataset1 for _ in range(len(domain_weights))]

    # DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_splitted/tokenized_data"
    # DOMAIN_KEYS = [
    #     "Github",
    #     "FreeLaw",
    #     "OpenWebText2",
    #     "PubMed Abstracts",
    #     "DM Mathematics",
    #     "OpenSubtitles",
    #     "HackerNews",
    #     "NIH ExPorter",
    #     "PubMed Central",
    #     "Enron Emails",
    # ]

    DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_raw/tokenized_data/train"
    DOMAIN_KEYS = [
        "Pile-CC",
        "Github",
        "OpenWebText2",
        "StackExchange",
        "Wikipedia (en)",
        "PubMed Abstracts",
        "USPTO Backgrounds",
        "FreeLaw",
        "PubMed Central",
        "Enron Emails",
        "HackerNews",
        "NIH ExPorter",
        "Books3",  # 12
        "ArXiv",  # 13 , launched
        "DM Mathematics",
        "OpenSubtitles",
        "Gutenberg (PG-19)",  # 16, done
        "Ubuntu IRC",  # 17, done
        "BookCorpus2",  # 18, launched
        "EuroParl",  # 19, launch
        "YoutubeSubtitles",
        "PhilPapers",
    ]

    TOKENIZED_DATASETS = [f"{DATASET_PATH}/{domain_name}" for domain_name in DOMAIN_KEYS]
    # domain_weights = torch.tensor(
    #     [
    #         0.34356916553540745,
    #         0.16838812972610234,
    #         0.24711766854236725,
    #         0.0679225638705455,
    #         0.059079828519653675,
    #         0.043720261601881555,
    #         0.01653850841342608,
    #         0.00604146633842096,
    #         0.04342813428189645,
    #         0.0041942731702987,
    #     ]
    # )

    # domain_weights = torch.tensor([
    #     0.1500, 0.1213, 0.0872, 0.0631, 0.0340, 0.0240, 0.0281, 0.0594, 0.1599,
    #     0.0015, 0.0058, 0.0021, 0.0605, 0.1136, 0.0209, 0.0154, 0.0202, 0.0037,
    #     0.0065, 0.0100, 0.0093, 0.0036
    # ])
    domain_weights = torch.tensor(
        [
            0.3267,
            0.003165,
            0.1223,
            0.0465,
            0.06024,
            0.06611,
            0.06174,
            0.0659,
            0.01737,
            0.005272,
            0.004745,
            0.00686,
            0.01651,
            0.08172,
            0.0009354,
            0.002027,
            0.013,
            0.0609,
            0.002643,
            0.01381,
            0.0004395,
            0.02115,
        ]
    )

    datasets = []
    for dataset_path in tqdm(TOKENIZED_DATASETS, desc="Loading tokenized dataset from disk"):
        d = load_from_disk(dataset_path)
        datasets.append(d)

    # from datasets import load_dataset
    # dataset = load_dataset("stas/c4-en-10k", split="train")
    # domain_weights = torch.tensor
    # datasets = [dataset for _ in range(len(domain_weights))]

    parallel_context = ParallelContext(
        data_parallel_size=DP_SIZE,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )

    # global_batch_size = 512
    # batch_size = global_batch_size // (num_microbatches * DP_SIZE)
    # NOTE: this cause 0 loss in some domains
    # num_microbatches = 4
    # batch_size = 8

    num_microbatches = 1
    batch_size = 32

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
        num_workers=1,
        pin_memory=True,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
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
        # if dist.get_rank(parallel_context.world_pg) == 0:
        #     # print(f"-------------------")
        #     # print(f"step = {step}, microbatch_idx = {sampler.microbatch_idx}")
        #     # print(f"step = {step}, domain_counters = {sampler.domain_counters}")
        #     # print(f"step = {step}, domain_batch_sizes = {sampler.domain_batch_sizes}")

        #     if step % num_microbatches:
        #         if dp_rank == 0:
        #             epoch = step / num_microbatches
        #             print(f"################# epoch = {epoch}")
        if step % 1000:
            print(f"################# epoch = {step / num_microbatches}")

        step += 1

        # if step == 20:
        #     break

    # step = 0
    # while True:
    #     # # idxs = (next(sampler) for _ in range(8))

    #     # # idxs = []
    #     # for _ in range(num_microbatches):
    #     #     _ = next(dataloader)

    #     # # NOTE: check not repeating idxs
    #     # # assert not set(idxs).intersection(yieled_idxs), f"epoch: {epoch}"

    #     # if epoch % 1000 == 0:
    #     #     print(f"rank: {dist.get_rank(parallel_context.dp_pg)}, epoch: {epoch} \n \n")

    #     # epoch += 1
    #     # # yieled_idxs.extend(idxs)

    #     _ = next(dataloader)
    #     if dist.get_rank(parallel_context.world_pg) == 0:
    #         print(f"-------------------")
    #         print(f"step = {step}, microbatch_idx = {sampler.microbatch_idx}")
    #         print(f"step = {step}, domain_counters = {sampler.domain_counters}")
    #         print(f"step = {step}, domain_batch_sizes = {sampler.domain_batch_sizes}")

    #         if step % num_microbatches:
    #             if dp_rank == 0:
    #                 epoch = step / num_microbatches
    #                 print(f"################# epoch = {epoch}")

    #     step += 1
