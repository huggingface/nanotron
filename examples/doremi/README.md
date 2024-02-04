# DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

You might think that the one of key ways for speeding up pretraining performance is either finding more quality data, increase FLOPs, or chaging model architecture but it's actually these are not all of them. DoReMi shows that given the same source of training data, a model using an optimal data mixing could outperform its equivalent model with random sampling by 2x-2.s5x across all domains's cross entropy loss, and downstream evaluations without any knowledge of the downstream evaluation tasks.

Step 0: Preprocessing data

Step 1: Train a small reference model using uniform sampling from each domain (for a given global batch size, you equally sample `x` samples across all domains, or in some cases, a domain has smaller amount of samples than other domains, this leads to some domain run out of samples early, so you could enable automatic domain weights based on the token count)


Step 2: Use the trained reference model from step 1 to train a identical model, and use its performance to dynamically tuning the domain weights during training


Step 3: We calculale the optimal domain weights by averaing domain weights across all training steps from step 1

Step 4: Use the optimal domain weights to train a larger model (could be 10x or 30x larger)

In our implementation, experiment results show that


### Tips

Since in the proxy model training, the domain weights are dynamically tune during training, that means there is a possiblity for a domain with low amount of samples running out of data, for guarantee no running out data during training, we recommend to check if the global_batch_size * total_training steps is smaller than the number of smaples in the smallest domain.
