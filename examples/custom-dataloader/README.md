# Use a custom dataloader with Nanotron

## Usage
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=2 examples/custom-dataloader/run_train.py --config-file examples/custom-dataloader/config_custom_dl.yaml
```

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
