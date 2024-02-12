import torch

domain_weights = torch.load(
    "/fsx/phuc/checkpoints/doremi/big-run-02/proxy-280m-llama_with_100k_reference/doremi_domain_weights_100000.pt"
)


total_weights = sum(d["domain_weights"] for d in domain_weights)
avg_weights = total_weights / len(domain_weights)

assert 1 == 1
