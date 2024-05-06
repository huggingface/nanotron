# Use a custom dataloader with Nanotron

This example shows how to use a custom dataloader with Nanotron. We will use a simple dataloader that loads a random tokenized dataset and feeds it to a Nanotron model.
https://github.com/huggingface/nanotron/blob/2e21db0db46a40bedbd03714616dd0ae4ea75914/examples/custom-dataloader/run_train.py#L72-L84

`DataCollatorForCLM` is a custom data collator that takes a list of input_ids and returns a dictionary with the input_ids and the labels on the ranks which need it. For example `input_ids` are only needed in the first PP rank, while `labels` are needed in the last PP rank.

## Troubleshooting

### `return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)`
```
  File "/fsx/nouamane/projects/nanotron/src/nanotron/parallel/tensor_parallel/nn.py", line 284, in forward
    out = super().forward(masked_input)
  File "/fsx/nouamane/miniconda/envs/2-1-cu121/lib/python3.10/site-packages/torch/nn/modules/sparse.py", line 162, in forward
    return F.embedding(
  File "/fsx/nouamane/miniconda/envs/2-1-cu121/lib/python3.10/site-packages/torch/nn/functional.py", line 2233, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: CUDA error: device-side assert triggered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

If you encounter an error with `torch.embedding`, it's probable you're feeding a token which is bigger than the model's vocabulary size. Check your model's vocab size and tokenizer
