# ClimLLaMA Training Pipeline

This directory contains scripts for converting HuggingFace checkpoints to Nanotron format and setting up finetuning configurations.

## Workflow Overview

```
┌─────────────────────┐       ┌──────────────────────┐       ┌─────────────────┐
│  1. Convert HF to   │       │  2. Prepare Training │       │  3. Run Fine-   │
│     Nanotron        │  ───> │     Config           │  ───> │     tuning      │
│                     │       │                      │       │                 │
│ convert_hf_to_nt.sh │       │ prepare_finetune_    │       │ run_finetune.sh │
│                     │       │ config.sh            │       │                 │
└─────────────────────┘       └──────────────────────┘       └─────────────────┘
        │                              │                              │
        │                              │                              │
        ▼                              ▼                              ▼
  checkpoint_nt/                config_finetune.yaml          Trained model!
  ├── model_config.json         (auto-generated from
  ├── model.safetensors          checkpoint metadata)
  └── ...
```

## Key Features

- **Automatic architecture detection**: Reads model config from checkpoint
- **Smart defaults**: Optimized hyperparameters for finetuning vs pretraining
- **Flexible configuration**: Easy to customize via command-line args
- **Production-ready**: Supports multi-GPU, distributed training

## Files

- **[convert_hf_to_nt.sh](convert_hf_to_nt.sh)** - Converts HuggingFace checkpoint to Nanotron format
- **[prepare_training_config.py](prepare_training_config.py)** - Python script that generates training config from checkpoint
- **[prepare_finetune_config.sh](prepare_finetune_config.sh)** - Shell script wrapper for easy config generation
- **[run_finetune.sh](run_finetune.sh)** - Launches the finetuning job

## Quick Start

### Step 1: Convert Checkpoint

Convert your HuggingFace checkpoint to Nanotron format:

```bash
bash climllama/convert_hf_to_nt.sh
```

This will:
- Read the checkpoint from: `/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_hf`
- Save to: `/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_nanotron`

### Step 2: Generate Training Config

Generate a training configuration file:

```bash
bash climllama/prepare_finetune_config.sh
```

This will:
- Read the model architecture from the Nanotron checkpoint
- Create a YAML config file at `climllama/config_finetune.yaml`
- Set up proper hyperparameters for finetuning

**Configuration options** (edit in `prepare_finetune_config.sh`):
- `DATASET` - HuggingFace dataset name (default: "trl-lib/tldr")
- `TRAIN_STEPS` - Number of training steps (default: 5000)
- `LEARNING_RATE` - Learning rate (default: 1e-5)
- `MICRO_BATCH_SIZE` - Batch size per GPU (default: 2)
- `SEQUENCE_LENGTH` - Max sequence length (default: 4096)
- `DP`, `TP`, `PP` - Parallelism degrees (default: 4, 2, 1)

### Step 3: Run Finetuning

Start the finetuning job:

```bash
bash climllama/run_finetune.sh
```

This will:
- Activate the nanotron environment
- Set required environment variables
- Launch distributed training with `torchrun`

## Advanced Usage

### Using `prepare_training_config.py` Directly

For more control, you can use the Python script directly:

```bash
python climllama/prepare_training_config.py \
    --checkpoint_path /path/to/nanotron/checkpoint \
    --tokenizer_path /path/to/tokenizer \
    --output_config my_config.yaml \
    --mode finetune \
    --dataset "my-org/my-dataset" \
    --train_steps 10000 \
    --learning_rate 5e-6 \
    --dp 8 --tp 2 --pp 1
```

**All available options:**

```
Required:
  --checkpoint_path PATH      Path to Nanotron checkpoint directory

Optional:
  --output_config PATH        Output YAML config path (default: config_finetune.yaml)
  --tokenizer_path PATH       Tokenizer path (default: uses checkpoint_path)
  --mode {finetune,pretrain}  Training mode (default: finetune)
  --dataset NAME              HF dataset name (default: trl-lib/tldr)
  --train_steps INT           Training steps (default: 5000)
  --learning_rate FLOAT       Learning rate (default: auto - 1e-5 for finetune, 3e-4 for pretrain)
  --micro_batch_size INT      Micro batch size (default: 2)
  --sequence_length INT       Sequence length (default: 4096)
  --dp INT                    Data parallelism (default: 4)
  --tp INT                    Tensor parallelism (default: 2)
  --pp INT                    Pipeline parallelism (default: 1)
  --batch_accumulation INT    Gradient accumulation steps (default: 1)
  --checkpoint_interval INT   Save checkpoint every N steps (default: 500)
  --seed INT                  Random seed (default: 42)
```

### Pretrain Mode

To continue pretraining instead of finetuning:

```bash
python climllama/prepare_training_config.py \
    --checkpoint_path /path/to/checkpoint \
    --mode pretrain \
    --dataset "HuggingFaceFW/fineweb-edu" \
    --learning_rate 3e-4
```

### Custom Dataset Format

For **Supervised Fine-Tuning (SFT)**, your dataset should have:
- `prompt` field - The instruction/input
- `completion` field - The target output

Example:
```json
{
  "prompt": "What is the capital of France?",
  "completion": "The capital of France is Paris."
}
```

For **Pretraining**, your dataset should have:
- `text` field - The raw text to train on

## What the Scripts Do

### `prepare_training_config.py` Features

1. **Reads checkpoint metadata**: Parses `model_config.json` from the Nanotron checkpoint
2. **Extracts model architecture**: Gets hidden_size, num_layers, vocab_size, etc.
3. **Calculates parameters**: Estimates total parameter count
4. **Generates training config**: Creates a complete YAML config with:
   - Proper parallelism settings
   - Optimizer configuration (AdamW with cosine decay)
   - Learning rate scheduler
   - Dataset configuration (SFT or pretrain)
   - Checkpoint settings (resume from converted checkpoint)
   - Tokenizer configuration

### Model Architecture Detection

The script automatically detects and uses:
- Hidden dimension
- Number of layers
- Number of attention heads
- Intermediate (FFN) size
- Vocabulary size
- Max position embeddings
- RoPE settings
- Normalization epsilon

## Example Output

When you run `prepare_finetune_config.sh`, you'll see:

```
Creating finetune configuration from checkpoint: /path/to/checkpoint
Tokenizer path: /path/to/tokenizer
Dataset: trl-lib/tldr
Training steps: 5000
Learning rate: auto
Parallelism: DP=4, TP=2, PP=1

Model configuration loaded: 3B parameters
  - Hidden size: 3072
  - Num layers: 28
  - Num attention heads: 32
  - Intermediate size: 8192
  - Vocab size: 32768
  - Max position embeddings: 4096

✓ Config saved to climllama/config_finetune.yaml

You can now start training with:
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  torchrun --nproc_per_node=8 run_train.py --config-file climllama/config_finetune.yaml
```

## Customizing the Config

After generation, you can manually edit `config_finetune.yaml` to:
- Change the dataset
- Adjust hyperparameters
- Modify parallelism settings
- Add data stages (e.g., different learning rate phases)
- Configure logging and monitoring

## Troubleshooting

**Config file not found**:
- Make sure you run `prepare_finetune_config.sh` before `run_finetune.sh`

**GPU count mismatch**:
- Ensure `NPROC` in `run_finetune.sh` matches `DP × TP × PP` from your config

**Checkpoint not found**:
- Verify the paths in `convert_hf_to_nt.sh` point to valid directories
- Make sure conversion completed successfully

**Out of memory**:
- Reduce `micro_batch_size` in the config
- Increase `batch_accumulation` to maintain effective batch size
- Increase parallelism (TP or PP)

## Next Steps

After training completes, you can:

1. **Convert back to HuggingFace format** for inference:
   ```bash
   torchrun --nproc_per_node=1 examples/llama/convert_nanotron_to_hf.py \
       --nanotron_checkpoint /path/to/finetuned/checkpoint \
       --hf_output_path /path/to/hf/output
   ```

2. **Resume training** from any checkpoint by updating `resume_checkpoint_path` in the config

3. **Evaluate** the model using your evaluation pipeline

## Resources

- [Nanotron Documentation](../../docs/your-first-training.md)
- [Multi-node Training Guide](../../docs/multi-node-training.md)
- [Custom Dataloader Examples](../../examples/custom-dataloader/README.md)
- [Main Training Script](../../run_train.py)
