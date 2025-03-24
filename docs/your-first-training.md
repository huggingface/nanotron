# Your First Training with Nanotron

This guide will walk you through the necessary steps to train your first model with Nanotron, a high-performance library for pretraining transformer models.

## Prerequisites

Before you begin, make sure you have:
- Python 3.10 or later (but less than 3.12)
- CUDA-enabled GPU(s)
- Nanotron installed (see [Installation](../README.md#installation))

## Single Node Training

Training a model on a single node involves two main steps:
1. Creating a configuration
2. Running the training script

### Step 1: Creating a Configuration

Nanotron uses YAML configuration files to define training parameters. You can either:
- Use an existing YAML config directly
- Generate a YAML config from a Python script

#### Option A: Using a Python Script to Generate Config

Creating a config with Python offers more flexibility and allows for programmatic configuration generation. Here's how:

1. Create a Python script similar to [`examples/config_tiny_llama.py`](../examples/config_tiny_llama.py):

2. Run the Python script to generate the YAML config:

```bash
python my_config.py
```

#### Option B: Use an Existing Config File

You can also use one of the provided example configurations directly, such as `examples/config_tiny_llama.yaml`.

### Step 2: Running the Training

Once you have your configuration file ready, you can start training using `torchrun`:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```

Where:
- `CUDA_DEVICE_MAX_CONNECTIONS=1`: Important environment variable for some distributed operations
- `--nproc_per_node=8`: Specifies the number of processes (GPUs) you want to use. Make sure this matches DPxTPxPP parallelism sizes.
- `run_train.py`: Main training script
- `--config-file examples/config_tiny_llama.yaml`: Path to your configuration file

### Additional Configuration Notes

1. **Parallelism**: Adjust `dp`, `tp`, and `pp` in the configuration based on your hardware:
   - `dp`: Data Parallelism - How many replicas of your model
   - `tp`: Tensor Parallelism - How to split individual tensors
   - `pp`: Pipeline Parallelism - How to split the model across stages

2. **Batch Size**: The global batch size is calculated as:
   ```
   micro_batch_size * batch_accumulation_per_replica * dp
   ```

3. **Checkpointing**: Set `checkpoint_interval` to control how often models are saved.

## Using a Custom Dataloader

If you want to use your own dataset instead of the built-in Hugging Face datasets support, you can create a custom dataloader:

1. Set your dataset configuration to `null`:
   ```yaml
   data:
     dataset: null # Custom dataloader will be used
     num_loading_workers: 1
     seed: 42
   name: Stable Training Stage
   start_training_step: 1
   ```

2. Implement a custom dataloader similar to the example in `examples/custom-dataloader/run_train.py`.

For detailed instructions, refer to `examples/custom-dataloader/README.md`.

## Multi-node Training

Check out the [Multi-node Training](multi-node-training.md) guide for more information.

## Troubleshooting

If you encounter issues with token sizes in your dataloader, ensure that your tokens do not exceed the model's vocabulary size. This is a common source of errors in training.

For more help, check the [Troubleshooting](multi-node-training.md#troubleshooting) section in the Multi-Node Training guide.
