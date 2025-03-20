# Multi-Node Training with Nanotron

This guide explains how to train models with Nanotron across multiple compute nodes using Slurm, a popular workload manager for high-performance computing (HPC) clusters.

## Using the Slurm Launcher

Nanotron provides a convenient script (`slurm_launcher.py`) to simplify launching multi-node training jobs on Slurm clusters. This script handles configuration generation, resource allocation, and job submission in one step.

### Basic Usage

```bash
python slurm_launcher.py --run_name my_experiment --nodes 4 --model_size base
```

This will:
1. Generate a Nanotron configuration file based on your parameters
2. Create a Slurm job script with appropriate settings
3. Submit the job to the Slurm scheduler
4. Save everything needed for reproducibility

### Important Parameters

The launcher supports many parameters, organized into logical groups:

#### Required Parameters
- `--run_name`: Name for your experiment (will be used in logs and checkpoints)

#### Slurm Configuration
- `--nodes`: Number of nodes to use (default: 2)
- `--gpus_per_node`: Number of GPUs per node (default: 8)
- `--partition`: Slurm partition to use (default: "hopper-prod")
- `--qos`: Slurm QOS to use (default: "normal")
- `--time_limit`: Time limit for the job in HH:MM:SS format (default: "1:00:00")
- `--email`: Email address for job notifications
- `--tmp_dir`: Temporary directory on compute nodes (default: "/tmp")
- `--pre_launch_commands`: Commands to run before job launch
- `--extra_env`: Additional environment variables to set

#### Model Configuration
- `--model_size`: Predefined size (`tiny`, `small`, `base`, `large`)
- `--hidden_size`, `--intermediate_size`, `--num_layers`, etc.: Custom model dimensions

#### Training Configuration
- `--seed`: Random seed for reproducibility (default: 42)
- `--train_steps`: Number of training steps (default: 10000)
- `--micro_batch_size`: Size of micro batches (default: 2)
- `--grad_accum_steps`: Gradient accumulation steps (default: 8)
- `--learning_rate`: Peak learning rate (default: 3e-4)
- `--min_lr`: Minimum learning rate for decay (default: 3e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--grad_clip`: Gradient clipping (default: 1.0)
- `--warmup_steps`: Learning rate warmup steps (default: 1000)

#### Parallelism Strategy
- `--dp`: Data parallelism (DP) degree (default: 8)
- `--pp`: Pipeline parallelism (PP) degree (default: 1)
- `--tp`: Tensor parallelism (TP) degree (default: 2)

**Note**: Make sure that DP × PP × TP does not exceed your total number of GPUs (nodes × gpus_per_node).

#### Dataset Configuration
- `--dataset`: Hugging Face dataset name or path (default: "stas/openwebtext-10k")
- `--text_column`: Column name for text in the dataset (default: "text")
- `--tokenizer`: Tokenizer name or path (default: "robot-test/dummy-tokenizer-wordlevel")

#### Path Configuration
- `--project`: Project name for logging (default: "nanotron")
- `--configs_path`: Where to save configurations (default: "logs/configs")
- `--slurm_logs_path`: Where to save Slurm output logs (default: "logs/slurm_logs")
- `--checkpoints_path`: Where to save model checkpoints (default: "checkpoints")
- `--slurm_scripts_dir`: Directory to save generated Slurm scripts (default: "logs/slurm_scripts")
- `--run_train_script`: Custom training script path (default: "run_train.py")
- `--save_interval`: Interval for saving checkpoints in steps (default: 1000)
- `--save_initial_state`: Save initial model state before training

#### Logging Configuration
- `--wandb_disabled`: Disable logging to Weights & Biases
- `--profiler_export_path`: Path to export the profiler tensorboard data

#### Execution Control
- `--dry_run`: Generate configs but don't submit job
- `--show_logs`: Show output of the job as it runs

### Examples

#### Training a Small Model for Testing

```bash
python slurm_launcher.py \
  --run_name quick_test \
  --nodes 2 \
  --model_size tiny \
  --train_steps 100 \
  --dataset stas/openwebtext-10k
```

#### Training a Large Model for Production

```bash
python slurm_launcher.py \
  --run_name production_run \
  --nodes 8 \
  --model_size large \
  --dp 4 \
  --pp 2 \
  --tp 2 \
  --train_steps 50000 \
  --learning_rate 2e-4 \
  --warmup_steps 2000 \
  --dataset my_dataset \
  --tokenizer my_tokenizer \
  --email researcher@example.com \
  --time_limit 72:00:00
```

**Note**: In this example, we're using 16 GPUs for training (4×2×2 = 16) out of 64 available GPUs (8 nodes × 8 GPUs).

#### Custom Model Architecture

```bash
python slurm_launcher.py \
  --run_name custom_arch \
  --nodes 4 \
  --hidden_size 1536 \
  --num_layers 24 \
  --num_heads 16 \
  --num_kv_heads 4 \
  --train_steps 20000
```

#### Dry Run (Generate Config Without Submitting)

```bash
python slurm_launcher.py \
  --run_name test_config \
  --model_size base \
  --dry_run
```

#### Using a Custom Training Script

```bash
python slurm_launcher.py \
  --run_name custom_script \
  --nodes 2 \
  --model_size base \
  --run_train_script path/to/my_custom_train.py \
  --slurm_scripts_dir slurm_scripts
```

#### Monitoring Training Output in Real-time

```bash
python slurm_launcher.py \
  --run_name monitored_run \
  --nodes 2 \
  --model_size base \
  --show_logs
```

## Manual Multi-Node Configuration

If you prefer to set up multi-node training manually, follow these steps:

1. Create a Nanotron configuration file (YAML or Python)
2. Set appropriate parallelism parameters:
   ```python
   parallelism = ParallelismArgs(
       dp=8,  # Adjust based on (total_gpus / (pp * tp))
       pp=2,  # Pipeline parallelism degree
       tp=2,  # Tensor parallelism degree
       pp_engine="1f1b",
       tp_mode="REDUCE_SCATTER",
       tp_linear_async_communication=True,
   )
   ```
3. Create a Slurm batch script:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=nanotron-training
   #SBATCH --nodes=4
   #SBATCH --ntasks-per-node=1
   #SBATCH --gpus-per-node=8
   #SBATCH --partition=your_partition
   #SBATCH --output=logs/%x-%j.out

   export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
   export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
   export MASTER_PORT=6000
   export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

   export TMPDIR=/tmp
   export CUDA_DEVICE_MAX_CONNECTIONS=1

   srun bash -c "torchrun \
       --nproc_per_node 8 \
       --nnodes $COUNT_NODE \
       --rdzv_backend c10d \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --max_restarts 0 \
       run_train.py --config-file your_config.yaml"
   ```

4. Submit the job:
   ```bash
   sbatch your_job_script.sh
   ```

## Tips for Multi-Node Training

1. **Node Communication**: Ensure your cluster has a high-speed interconnect (like InfiniBand) for efficient multi-node communication.

2. **Balanced Parallelism**:
   - For small models (< 1B parameters): Focus on data parallelism
   - For medium models (1-10B): Use TP=2, PP=1 or PP=2
   - For large models (>10B): Increase both TP and PP

3. **Fault Tolerance**: Configure `--save_interval` to save regularly in case of job failures.

4. **Monitoring**: Use `--show_logs` to monitor training progress in real-time.

5. **Resource Efficiency**: Balance your parallelism settings (--dp, --pp, --tp) to maximize GPU utilization.

6. **Environment Variables**: Use `--extra_env` to set additional environment variables like NCCL settings.

## Troubleshooting

### Common Issues

1. **GPU Communication Errors**
   - Check network connectivity between nodes
   - Try setting `CUDA_DEVICE_MAX_CONNECTIONS=1`
   - Use environment variables like `NCCL_DEBUG=WARN` for debugging

2. **Out of Memory Errors**
   - Reduce batch size or sequence length
   - Increase pipeline parallelism
   - Consider gradient checkpointing

3. **Job Timeouts**
   - Increase `--time_limit` parameter
   - Set appropriate checkpointing intervals with `--save_interval`

For more detailed information, refer to the [Nanotron documentation](https://github.com/huggingface/nanotron) and your cluster's specific Slurm documentation.
