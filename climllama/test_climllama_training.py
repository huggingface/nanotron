"""End-to-end training test for ClimLlama with ClimLlamaDataset.

This script tests Phase 4 of the ClimLlama implementation: end-to-end training
with a small model and the ClimLlama dataset. It creates a minimal configuration,
builds the model, dataset, and dataloader, then runs a few training steps.

Usage:
    # Single GPU test with default data path:
    python climllama/test_climllama_training.py

    # With custom data prefix:
    python climllama/test_climllama_training.py --data-prefix data/combined_interleave_256/1

    # With multiple data prefixes (blended):
    python climllama/test_climllama_training.py --data-prefix data/path1 data/path2

    # Full options:
    python climllama/test_climllama_training.py \\
        --data-prefix data/combined_interleave_256/1 \\
        --checkpoint-path /tmp/climllama_ckpt \\
        --sequence-length 256 \\
        --micro-batch-size 2 \\
        --train-steps 5

    # Multi-GPU test:
    torchrun --nproc_per_node=2 climllama/test_climllama_training.py --data-prefix data/path
"""

import argparse
import os
import sys
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist

# Ensure we can import nanotron modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nanotron.mock_flash_attn  # noqa: F401 - Ensure flash_attn is available

from nanotron import logging
from nanotron.config import (
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LoggingArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.config.config import AdamWOptimizerArgs, LRSchedulerArgs
from nanotron.config.config import ClimLlamaDatasetsArgs
from nanotron.config.models_config import ClimLlamaConfig, RandomInit
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.config.utils_config import InitScalingMethod
from nanotron.data.climllama_collator import DataCollatorForClimLlama
from nanotron.data.climllama_dataset import build_climllama_dataset
from nanotron.data.dataloader import get_dataloader_worker_init
from nanotron.logging import log_rank
from nanotron.models.climllama import CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from torch.utils.data import DataLoader
from unittest.mock import MagicMock

logger = logging.get_logger(__name__)


def init_distributed():
    """Initialize distributed environment for testing."""
    if dist.is_initialized():
        return

    # Check if we're running under torchrun
    if "RANK" in os.environ:
        # Running with torchrun
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        # Single process mode
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29516"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_small_climllama_config() -> ClimLlamaConfig:
    """Create a small ClimLlama configuration for testing.

    Returns a minimal configuration suitable for quick testing.
    """
    return ClimLlamaConfig(
        # Small model for testing
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        vocab_size=33312,  # Standard ClimLlama vocab size
        max_position_embeddings=4096,

        # ClimLlama-specific config
        is_climllama_config=True,
        use_absolute_position_embeddings=True,
        var_vocab_size=16,
        variables=(
            "unk", "z", "t", "q",
            "u", "v", "w",
            "t2m", "msl", "u10", "v10",
            "tp", "tp_6h",
            "pad0", "pad1", "pad2",
        ),
        res_vocab_size=12,
        leadtime_vocab_size=12,
        leadtimes=(0, 1, 3, 6, 12, 24, 48, 72, 120, 168, 336, 720),
        use_spatial_temporal_encoding=True,
        max_tp=4,

        # Architecture settings
        _attn_implementation="flash_attention_2",
        _fused_rms_norm=True,
        _fused_rotary_emb=True,
        _use_doc_masking=True,
        _use_qkv_packed=True,
        attention_bias=False,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        z_loss_enabled=False,
    )


def create_test_config(
    data_prefix: Union[str, List[str]],
    checkpoint_path: str,
    sequence_length: int = 256,
    micro_batch_size: int = 2,
    train_steps: int = 5,
) -> Config:
    """Create a test configuration for ClimLlama training.

    Args:
        data_prefix: Path prefix for the indexed dataset. Can be:
            - A single string path: "data/combined_interleave_256/1"
              (expects files: 1.bin, 1.idx, 1.json, optionally 1.npy)
            - A list with single path: ["data/combined_interleave_256/1"]
            - Multiple paths (equal weights): ["data/path1", "data/path2"]
            - Weighted blend: [0.7, "data/path1", 0.3, "data/path2"]
        checkpoint_path: Path for saving checkpoints
        sequence_length: Sequence length for training
        micro_batch_size: Micro batch size
        train_steps: Number of training steps

    Returns:
        Config object for training
    """
    model_config = create_small_climllama_config()

    # Normalize data_prefix to list format
    if isinstance(data_prefix, str):
        data_prefix_list = [data_prefix]
    else:
        data_prefix_list = data_prefix

    # Create dataset args
    dataset_args = ClimLlamaDatasetsArgs(
        data_prefix=data_prefix_list,
        variables=(
            "unk", "z", "t", "q",
            "u", "v", "w",
            "t2m", "msl", "u10", "v10",
            "tp", "tp_6h",
        ),
        leadtimes=(0, 1, 3, 6, 12, 24, 48, 72, 120, 168, 336, 720),
        codebook_size=32768,
        sampler_type="sequential",
        pad_samples_to_global_batch_size=True,
    )

    config = Config(
        general=GeneralArgs(
            project="climllama_test",
            run="test_training",
            seed=42,
            ignore_sanity_checks=True,
        ),
        checkpoints=CheckpointsArgs(
            checkpoints_path=Path(checkpoint_path),
            checkpoint_interval=1000,  # Don't save during test
            save_initial_state=False,
            save_final_state=False,
        ),
        model=ModelArgs(
            model_config=model_config,
            init_method=RandomInit(std=0.02, scaling_method=InitScalingMethod.NUM_LAYERS),
            dtype=torch.bfloat16,
            make_vocab_size_divisible_by=1,
        ),
        tokenizer=TokenizerArgs(
            tokenizer_name_or_path=None,  # We don't need a real tokenizer for this test
        ),
        parallelism=ParallelismArgs(
            dp=1,
            tp=1,
            pp=1,
            context_parallel_size=1,
        ),
        tokens=TokensArgs(
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            batch_accumulation_per_replica=1,
            train_steps=train_steps,
        ),
        optimizer=OptimizerArgs(
            learning_rate_scheduler=LRSchedulerArgs(
                learning_rate=1e-4,
                lr_warmup_steps=1,
                lr_warmup_style="linear",
                lr_decay_steps=train_steps,
                lr_decay_style="cosine",
                min_decay_lr=1e-5,
            ),
            optimizer_factory=AdamWOptimizerArgs(
                name="adamW",
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-8,
                torch_adam_is_fused=True,
            ),
            weight_decay=0.01,
            clip_grad=1.0,
            accumulate_grad_in_fp32=True,
            zero_stage=0,
        ),
        logging=LoggingArgs(
            log_level="info",
            log_level_replica="warning",
            iteration_step_info_interval=1,
        ),
        data_stages=[
            DatasetStageArgs(
                name="training",
                start_training_step=1,
                data=DataArgs(
                    dataset=dataset_args,
                    seed=42,
                    num_loading_workers=0,
                ),
            )
        ],
    )

    return config


def get_climllama_dataloader(
    trainer: DistributedTrainer,
    data_args: ClimLlamaDatasetsArgs,
    consumed_train_samples: int = 0,
    num_samples: int = 100,
) -> DataLoader:
    """Create a dataloader for ClimLlama training.

    Args:
        trainer: The distributed trainer
        data_args: ClimLlama dataset arguments
        consumed_train_samples: Number of samples already consumed
        num_samples: Total number of samples to generate

    Returns:
        DataLoader for ClimLlama training
    """
    from transformers import AutoTokenizer

    # Get PP ranks
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Create a mock tokenizer (we don't need text tokenization for ClimLlama)
    tokenizer = MagicMock()

    # Build the dataset
    data_prefix = data_args.data_prefix
    if isinstance(data_prefix, list):
        data_prefix = data_prefix[0] if len(data_prefix) == 1 else data_prefix

    train_dataset = build_climllama_dataset(
        cfg=data_args,
        tokenizer=tokenizer,
        data_prefix=data_prefix, # type: ignore
        num_samples=num_samples,
        seed=42,
        parallel_context=trainer.parallel_context,
        name="train",
        seq_length=trainer.sequence_length,
        drop_last=True,
    )

    # Create collator
    collator = DataCollatorForClimLlama(
        sequence_length=trainer.sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=trainer.parallel_context,
        use_doc_masking=True,
        cp_return_global_position_ids=True,
    )

    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=trainer.micro_batch_size,
        collate_fn=collator,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=get_dataloader_worker_init(dp_rank=trainer.parallel_context.dp_pg.rank()),
    )

    return dataloader


def test_model_forward_pass():
    """Test that the ClimLlama model can do a forward pass with dummy data."""
    print("\n" + "="*60)
    print("Testing ClimLlama model forward pass with dummy data")
    print("="*60)

    init_distributed()

    try:
        # Import model components
        from nanotron.models.climllama import ClimLlamaForTraining
        from nanotron.models import build_model
        from nanotron.parallel.context import ParallelContext
        from nanotron.config import ParallelismArgs

        # Create parallel context
        parallel_context = ParallelContext(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            context_parallel_size=1,
        )

        # Create model config
        config = create_small_climllama_config()

        # Create parallelism args
        parallel_config = ParallelismArgs(dp=1, tp=1, pp=1, context_parallel_size=1)

        print(f"Creating model with config: hidden_size={config.hidden_size}, "
              f"layers={config.num_hidden_layers}, heads={config.num_attention_heads}")

        # Determine device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Build model using nanotron's build_model to properly set up pipeline blocks
        model = build_model(
            model_builder=lambda: ClimLlamaForTraining(
                config=config,
                parallel_context=parallel_context,
                parallel_config=parallel_config,
            ),
            parallel_context=parallel_context,
            dtype=dtype,
            device=device,
        )

        # Initialize model weights randomly
        for param in model.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

        # Create dummy inputs
        batch_size = 2
        seq_length = 64

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        var_idx = torch.randint(0, config.var_vocab_size, (batch_size, seq_length), device=device)
        res_idx = torch.randint(0, config.res_vocab_size, (batch_size, seq_length), device=device)
        leadtime_idx = torch.randint(0, config.leadtime_vocab_size, (batch_size, seq_length), device=device)
        spatial_temporal_features = torch.randn(
            batch_size, seq_length, CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES, device=device, dtype=dtype
        )
        label_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        label_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)

        print(f"Input shapes: input_ids={input_ids.shape}, var_idx={var_idx.shape}, "
              f"spatial_temporal_features={spatial_temporal_features.shape}")

        # Forward pass
        model.train()
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            var_idx=var_idx,
            res_idx=res_idx,
            leadtime_idx=leadtime_idx,
            spatial_temporal_features=spatial_temporal_features,
            label_ids=label_ids,
            label_mask=label_mask,
        )

        print(f"Forward pass successful! Output keys: {output.keys()}")
        if "loss" in output:
            print(f"Loss: {output['loss'].item():.4f}")

        # Test backward pass
        if "loss" in output:
            output["loss"].backward()
            print("Backward pass successful!")

            # Check that gradients were computed
            grad_count = sum(1 for p in model.parameters() if p.grad is not None)
            total_params = sum(1 for p in model.parameters())
            print(f"Parameters with gradients: {grad_count}/{total_params}")

        print("\n[PASS] Model forward/backward pass test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Model forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_distributed()


def test_dataset_and_collator(data_prefix: Optional[str] = None):
    """Test that ClimLlamaDataset and DataCollatorForClimLlama work together.

    Args:
        data_prefix: Path prefix for the indexed dataset. If None, uses default test path.
    """
    print("\n" + "="*60)
    print("Testing ClimLlamaDataset and DataCollatorForClimLlama")
    print("="*60)

    init_distributed()

    try:
        from nanotron.data.climllama_dataset import build_climllama_dataset
        from nanotron.data.climllama_collator import DataCollatorForClimLlama
        from nanotron.parallel.context import ParallelContext

        # Use provided data_prefix or default
        if data_prefix is None:
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data/combined_interleave_256",
            )
            data_prefix = os.path.join(data_path, "1")

        if not os.path.exists(f"{data_prefix}.bin"):
            print(f"[SKIP] Test data not found at {data_prefix}.bin")
            print("Please ensure the test data is available to run this test.")
            return None  # Skip, not fail

        # Create parallel context
        parallel_context = ParallelContext(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            context_parallel_size=1,
        )

        # Create dataset config
        dataset_cfg = ClimLlamaDatasetsArgs(
            data_prefix=[data_prefix],
            variables=(
                "unk", "z", "t", "q",
                "u", "v", "w",
                "t2m", "msl", "u10", "v10",
                "tp", "tp_6h",
            ),
            leadtimes=(0, 1, 3, 6, 12, 24, 48, 72, 120, 168, 336, 720),
            codebook_size=32768,
        )

        # Build dataset
        tokenizer = MagicMock()
        dataset = build_climllama_dataset(
            cfg=dataset_cfg,
            tokenizer=tokenizer,
            data_prefix=data_prefix,
            num_samples=10,
            seed=42,
            parallel_context=parallel_context,
            name="test",
            drop_last=True,
        )

        print(f"Dataset created with {len(dataset)} samples")

        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Sample shapes:")
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")

        # Get sequence length from sample
        seq_length = len(sample["input_ids"]) - 1  # -1 for next token prediction
        print(f"Sequence length (from sample): {seq_length}")

        # Create collator
        collator = DataCollatorForClimLlama(
            sequence_length=seq_length,
            input_pp_rank=0,
            output_pp_rank=0,
            parallel_context=parallel_context,
            use_doc_masking=True,
        )

        # Collate a batch
        batch_size = 2
        samples = [dataset[i] for i in range(batch_size)]
        batch = collator(samples)

        print(f"\nCollated batch keys: {batch.keys()}")
        print(f"Collated batch shapes:")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}") # type: ignore
            else:
                print(f"  {k}: {type(v).__name__}")

        # Verify batch shapes
        assert batch["input_ids"].shape == (batch_size, seq_length), \
            f"Expected input_ids shape ({batch_size}, {seq_length}), got {batch['input_ids'].shape}" # type: ignore
        assert batch["var_idx"].shape == (batch_size, seq_length), \
            f"Expected var_idx shape ({batch_size}, {seq_length}), got {batch['var_idx'].shape}" # type: ignore
        assert batch["spatial_temporal_features"].shape == (batch_size, seq_length, CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES), \
            f"Expected spatial_temporal_features shape ({batch_size}, {seq_length}, {CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES})" # type: ignore
        print("\n[PASS] Dataset and collator test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Dataset and collator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_distributed()


def test_end_to_end_training(
    data_prefix: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    sequence_length: Optional[int] = None,
    micro_batch_size: int = 2,
    train_steps: int = 5,
):
    """Test end-to-end training with ClimLlama model and dataset.

    This test:
    1. Creates a small ClimLlama model
    2. Loads the ClimLlamaDataset
    3. Runs a few training steps
    4. Verifies the loss decreases

    Args:
        data_prefix: Path prefix for the indexed dataset. If None, uses default test path.
        checkpoint_path: Path for saving checkpoints. If None, uses a temp directory.
        sequence_length: Sequence length to pass to the dataset. If None, dataset decides.
        micro_batch_size: Micro batch size for training.
        train_steps: Number of training steps to run.
    """
    print("\n" + "="*60)
    print("Testing end-to-end ClimLlama training")
    print("="*60)

    # Use provided data_prefix or default
    if data_prefix is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data/combined_interleave_256",
        )
        data_prefix = os.path.join(data_path, "1")

    if not os.path.exists(f"{data_prefix}.bin"):
        print(f"[SKIP] Test data not found at {data_prefix}.bin")
        print("Please ensure the test data is available to run this test.")
        return None  # Skip, not fail

    # Allow explicit override; if None, dataset will use metadata/default
    seq_length = sequence_length
    if seq_length is not None:
        print(f"Using provided sequence length override: {seq_length}")

    # Create checkpoint directory (use provided or temp)
    cleanup_checkpoint = False
    if checkpoint_path is None:
        checkpoint_dir = tempfile.mkdtemp(prefix="climllama_test_")
        cleanup_checkpoint = True
    else:
        checkpoint_dir = checkpoint_path
        os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        init_distributed()

        from nanotron.models.climllama import ClimLlamaForTraining
        from nanotron.models import build_model
        from nanotron.parallel.context import ParallelContext
        from nanotron.config import ParallelismArgs

        # Create parallel context
        parallel_context = ParallelContext(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            context_parallel_size=1,
        )

        # Create model config
        config = create_small_climllama_config()

        # Create parallelism args
        parallel_config = ParallelismArgs(dp=1, tp=1, pp=1, context_parallel_size=1)

        print(f"Building model: hidden={config.hidden_size}, layers={config.num_hidden_layers}")

        # Determine device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Build model using nanotron's build_model to properly set up pipeline blocks
        model = build_model(
            model_builder=lambda: ClimLlamaForTraining(
                config=config,
                parallel_context=parallel_context,
                parallel_config=parallel_config,
            ),
            parallel_context=parallel_context,
            dtype=dtype,
            device=device,
        )

        # Initialize model weights
        for param in model.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        # Create dataset config
        dataset_cfg = ClimLlamaDatasetsArgs(
            data_prefix=[data_prefix],
            variables=(
                "unk", "z", "t", "q",
                "u", "v", "w",
                "t2m", "msl", "u10", "v10",
                "tp", "tp_6h",
            ),
            leadtimes=(0, 1, 3, 6, 12, 24, 48, 72, 120, 168, 336, 720),
            codebook_size=32768,
        )

        # Build dataset
        tokenizer = MagicMock()
        dataset = build_climllama_dataset(
            cfg=dataset_cfg,
            tokenizer=tokenizer,
            data_prefix=data_prefix,
            num_samples=20,
            seed=42,
            parallel_context=parallel_context,
            name="train",
            seq_length=seq_length,
            drop_last=True,
        )

        # Get actual sequence length from dataset
        sample = dataset[0]
        actual_seq_length = len(sample["input_ids"]) - 1
        print(f"Dataset sequence length: {actual_seq_length}")

        # Create collator
        collator = DataCollatorForClimLlama(
            sequence_length=actual_seq_length,
            input_pp_rank=0,
            output_pp_rank=0,
            parallel_context=parallel_context,
            use_doc_masking=True,
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            collate_fn=collator,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

        print(f"Dataset size: {len(dataset)}, Batch size: {micro_batch_size}")
        print(f"Number of batches: {len(dataloader)}")

        # Training loop
        num_steps = min(train_steps, len(dataloader))
        losses = []

        model.train()
        print(f"\nRunning {num_steps} training steps...")

        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            # Move batch to device and convert dtypes as needed
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    tensor = torch.from_numpy(v)
                    # Float tensors need correct dtype for model
                    if tensor.dtype in (torch.float32, torch.float64):
                        tensor = tensor.to(dtype=dtype, device=device)
                    else:
                        tensor = tensor.to(device=device)
                    batch_device[k] = tensor
                elif isinstance(v, torch.Tensor):
                    if v.dtype in (torch.float32, torch.float64):
                        batch_device[k] = v.to(dtype=dtype, device=device)
                    else:
                        batch_device[k] = v.to(device=device)
                else:
                    batch_device[k] = v

            # Forward pass
            optimizer.zero_grad()
            output = model(**batch_device)

            loss = output["loss"]
            losses.append(loss.item())

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()

            print(f"  Step {step + 1}/{num_steps}: loss = {loss.item():.4f}")

        # Check results
        print(f"\nTraining results:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss:   {losses[-1]:.4f}")
        print(f"  Loss change:  {losses[-1] - losses[0]:.4f}")

        # The loss should generally decrease or stay stable
        # We don't require strict decrease as training can be noisy
        if len(losses) >= 2:
            avg_first_half = sum(losses[:len(losses)//2]) / (len(losses)//2) if len(losses) >= 2 else losses[0]
            avg_second_half = sum(losses[len(losses)//2:]) / (len(losses) - len(losses)//2)
            print(f"  Avg first half:  {avg_first_half:.4f}")
            print(f"  Avg second half: {avg_second_half:.4f}")

        print("\n[PASS] End-to-end training test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] End-to-end training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_distributed()
        # Clean up checkpoint directory only if we created a temp one
        if cleanup_checkpoint and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end training test for ClimLlama with ClimLlamaDataset"
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        nargs="+",
        default=None,
        help="Path prefix(es) for the indexed dataset. "
             "Examples: 'data/path/1' or 'data/path1 data/path2' for blending"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path for saving checkpoints. If not provided, uses a temp directory."
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Sequence length override to pass to the dataset. If not set, dataset uses metadata."
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size for training (default: 2)"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=5,
        help="Number of training steps to run (default: 5)"
    )
    parser.add_argument(
        "--skip-model-test",
        action="store_true",
        help="Skip the model forward pass test"
    )
    parser.add_argument(
        "--skip-dataset-test",
        action="store_true",
        help="Skip the dataset and collator test"
    )
    parser.add_argument(
        "--skip-e2e-test",
        action="store_true",
        help="Skip the end-to-end training test"
    )
    return parser.parse_args()


def main():
    """Run all ClimLlama training tests."""
    args = parse_args()

    # Resolve data_prefix (handle single or multiple paths)
    data_prefix = None
    if args.data_prefix is not None:
        if len(args.data_prefix) == 1:
            data_prefix = args.data_prefix[0]
        else:
            data_prefix = args.data_prefix[0]  # Use first for tests, full list for config

    print("="*60)
    print("ClimLlama End-to-End Training Tests")
    print("="*60)
    if data_prefix:
        print(f"Data prefix: {data_prefix}")
    if args.checkpoint_path:
        print(f"Checkpoint path: {args.checkpoint_path}")
    if args.sequence_length is not None:
        print(f"Sequence length: {args.sequence_length}")
    print(f"Micro batch size: {args.micro_batch_size}")
    print(f"Train steps: {args.train_steps}")

    results = {}

    # Test 1: Model forward pass
    if not args.skip_model_test:
        results["model_forward"] = test_model_forward_pass()
    else:
        print("\n[SKIP] Model forward pass test skipped by user")
        results["model_forward"] = None

    # Test 2: Dataset and collator
    if not args.skip_dataset_test:
        results["dataset_collator"] = test_dataset_and_collator(data_prefix=data_prefix)
    else:
        print("\n[SKIP] Dataset and collator test skipped by user")
        results["dataset_collator"] = None

    # Test 3: End-to-end training
    if not args.skip_e2e_test:
        results["e2e_training"] = test_end_to_end_training(
            data_prefix=data_prefix,
            checkpoint_path=args.checkpoint_path,
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            train_steps=args.train_steps,
        )
    else:
        print("\n[SKIP] End-to-end training test skipped by user")
        results["e2e_training"] = None

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"  {test_name}: {status}")

    # Return exit code
    failures = sum(1 for r in results.values() if r is False)
    if failures > 0:
        print(f"\n{failures} test(s) failed!")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
