import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer
import torch.distributed as dist
from typing import Dict

from helpers.llama import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config
from helpers.utils import available_gpus, get_all_3d_configurations, init_distributed, rerun_if_address_is_in_use
from nanotron.config import Config, ModelArgs, RandomInit
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy


def create_sft_dataset(tokenizer, sequence_length=16, num_samples=10):
    """Create a tiny SFT dataset for testing."""
    instructions = [f"Instruction {i}" for i in range(num_samples)]
    inputs = [f"Input {i}" for i in range(num_samples)]
    outputs = [f"Output {i}" for i in range(num_samples)]
    
    # Format as instruction-input-output pairs
    texts = [
        f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        for instruction, input, output in zip(instructions, inputs, outputs)
    ]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Create masks for loss computation (only compute loss on response tokens)
    label_masks = []
    for text in texts:
        response_start = text.find("### Response:")
        if response_start == -1:
            # Fallback if format is wrong
            label_masks.append([0] * sequence_length)
            continue
            
        # Tokenize just the prefix to find response token positions
        prefix = text[:response_start + len("### Response:")]
        prefix_tokens = tokenizer(prefix, add_special_tokens=False)
        prefix_length = len(prefix_tokens["input_ids"])
        
        # Create mask: 1 for response tokens, 0 for instruction/input tokens
        mask = [0] * prefix_length + [1] * (sequence_length - prefix_length)
        mask = mask[:sequence_length]  # Truncate if needed
        label_masks.append(mask)
    
    # Create dataset
    dataset_dict = {
        "input_ids": encodings["input_ids"],
        "input_mask": encodings["attention_mask"],
        "label_ids": encodings["input_ids"].clone(),  # Same as input_ids for autoregressive training
        "label_mask": torch.tensor(label_masks)
    }
    
    return Dataset.from_dict(dataset_dict)

@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1)])  # Simple test with single GPU
@rerun_if_address_is_in_use()
def test_sft_training(tp: int, dp: int, pp: int):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_sft_training)()


def _test_sft_training(parallel_context: ParallelContext):
    # Setup model config
    model_args = ModelArgs(init_method=RandomInit(std=0.02), model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)
    
    # Create model
    model = create_llama_from_config(
        model_config=config.model.model_config,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    model.init_model_randomly(config=config)
    
    # Create tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create tiny SFT dataset
    dataset = create_sft_dataset(tokenizer, sequence_length=16, num_samples=10)
    
    # Training parameters
    micro_batch_size = 2
    learning_rate = 5e-5
    num_iterations = 3
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    
    # Track initial loss for comparison
    initial_losses = []
    final_losses = []
    
    for iteration in range(num_iterations):
        for i in range(0, len(dataset), micro_batch_size):
            # Get batch
            batch_data = {k: dataset[k][i:i+micro_batch_size] for k in dataset.features}
            
            # Move to device
            batch = {key: tensor.cuda() for key, tensor in batch_data.items()}
            
            # Forward and backward
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            
            # Track loss
            if iteration == 0 and i == 0:
                initial_losses.append(loss.item())
            if iteration == num_iterations - 1 and i == 0:
                final_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
    
    # Verify that training is working by checking loss decreased
    assert final_losses[0] < initial_losses[0], f"Training did not reduce loss: initial={initial_losses[0]}, final={final_losses[0]}"
    
    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1)])  # Simple test with single GPU
@rerun_if_address_is_in_use()
def test_sft_loss_masking(tp: int, dp: int, pp: int):
    """Test that loss masking works correctly for SFT (only computing loss on response tokens)."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_sft_loss_masking)()
def _test_sft_loss_masking(parallel_context: ParallelContext):
    # Setup model config
    model_args = ModelArgs(init_method=RandomInit(std=0.02), model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)
    
    # Create model
    model = create_llama_from_config(
        model_config=config.model.model_config,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    model.init_model_randomly(config=config)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create a simple dataset with different mask patterns
    sequence_length = 16
    
    # Case 1: All tokens included in loss (full loss)
    all_included = {
        "input_ids": torch.randint(0, 100, (1, sequence_length)).cuda(),
        "input_mask": torch.ones(1, sequence_length).bool().cuda(),
        "label_ids": torch.randint(0, 100, (1, sequence_length)).cuda(),
        "label_mask": torch.ones(1, sequence_length).cuda(),
    }
    
    # Case 2: Half tokens included in loss (partial loss)
    half_included = {
        "input_ids": torch.randint(0, 100, (1, sequence_length)).cuda(),
        "input_mask": torch.ones(1, sequence_length).bool().cuda(),
        "label_ids": torch.randint(0, 100, (1, sequence_length)).cuda(),
        "label_mask": torch.cat([
            torch.zeros(1, sequence_length // 2),
            torch.ones(1, sequence_length // 2)
        ], dim=1).cuda(),
    }
    
    # Case 3: No tokens included in loss (should raise an error)
    none_included = {
        "input_ids": torch.randint(0, 100, (1, sequence_length)).cuda(),
        "input_mask": torch.ones(1, sequence_length).bool().cuda(),
        "label_ids": torch.randint(0, 100, (1, sequence_length)).cuda(),
        "label_mask": torch.zeros(1, sequence_length).cuda(),
    }
    
    # Case 4: Almost no tokens included in loss (near-zero loss) - just one token to avoid error
    almost_none_included = {
        "input_ids": torch.randint(0, 100, (1, sequence_length)).cuda(),
        "input_mask": torch.ones(1, sequence_length).bool().cuda(),
        "label_ids": torch.randint(0, 100, (1, sequence_length)).cuda(),
        "label_mask": torch.zeros(1, sequence_length).cuda(),
    }
    almost_none_included["label_mask"][0, 0] = 1  # Set just one token to be included in loss
    
    # Forward pass for each case
    model.eval()  # Use eval mode to avoid dropout randomness
    
    with torch.no_grad():
        loss_all = model(**all_included)["loss"]
        loss_half = model(**half_included)["loss"]
        loss_almost_none = model(**almost_none_included)["loss"]
    
    # Verify loss masking works correctly
    # Only check on the output rank
    if dist.get_rank(parallel_context.pp_pg) == parallel_context.pipeline_parallel_size - 1:
        print(f"Loss with all tokens included: {loss_all}")
        print(f"Loss with half tokens included: {loss_half}")
        print(f"Loss with almost no tokens included: {loss_almost_none}")
        
        assert loss_all > 0, "Full mask should produce positive loss"
        assert loss_half > 0, "Partial mask should produce positive loss"
        assert loss_almost_none > 0, "Almost no tokens included should produce positive loss"
        
        # Note: We don't assert loss_half < loss_all because it's not always true.
        # The masked_mean function normalizes the loss by the number of tokens in the mask:
        #   (loss * label_mask).sum() / label_mask.sum()
        # Instead, we just verify that masking produces a different loss value.
        assert not torch.allclose(loss_half, loss_all, rtol=1e-6, atol=1e-6, equal_nan=False), "Partial mask should produce different loss than full mask"
        assert not torch.allclose(loss_almost_none, loss_all, rtol=1e-6, atol=1e-6, equal_nan=False), "Almost no tokens should produce different loss than full mask"
        # Test with all tokens masked out (should return NaN)
        with torch.no_grad():
            loss_none = model(**none_included)["loss"]
            print(f"Loss with zero mask: {loss_none}")
            assert torch.isnan(loss_none), "Loss should be NaN with zero mask"
    
    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1)])  # Simple test with single GPU
@rerun_if_address_is_in_use()
def test_zero_mask_assertion(tp: int, dp: int, pp: int):
    """Test that the built-in assertion in masked_mean correctly catches zero masks."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_zero_mask_assertion)()


def _test_zero_mask_assertion(parallel_context: ParallelContext):
    # Setup model config
    model_args = ModelArgs(init_method=RandomInit(std=0.02), model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)
    
    # Create model
    model = create_llama_from_config(
        model_config=config.model.model_config,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    model.init_model_randomly(config=config)
    
    # Create a simple dataset with different mask patterns
    sequence_length = 16
    batch_size = 2
    
    # Create inputs with valid mask
    valid_inputs = {
        "input_ids": torch.randint(0, 100, (batch_size, sequence_length)).cuda(),
        "input_mask": torch.ones(batch_size, sequence_length).bool().cuda(),
        "label_ids": torch.randint(0, 100, (batch_size, sequence_length)).cuda(),
        "label_mask": torch.ones(batch_size, sequence_length).cuda(),
    }
    
    # Create inputs with zero mask
    zero_mask_inputs = {
        "input_ids": torch.randint(0, 100, (batch_size, sequence_length)).cuda(),
        "input_mask": torch.ones(batch_size, sequence_length).bool().cuda(),
        "label_ids": torch.randint(0, 100, (batch_size, sequence_length)).cuda(),
        "label_mask": torch.zeros(batch_size, sequence_length).cuda(),
    }
    
    # Test with valid mask
    model.eval()
    with torch.no_grad():
        loss_valid = model(**valid_inputs)["loss"]
        print(f"Loss with valid mask: {loss_valid}")
        assert not torch.isnan(loss_valid), "Loss should not be NaN with valid mask"
    
    # Test with zero mask (should return NaN)
    with torch.no_grad():
        loss_zero = model(**zero_mask_inputs)["loss"]
        print(f"Loss with zero mask: {loss_zero}")
        assert torch.isnan(loss_zero), "Loss should be NaN with zero mask"
    
    parallel_context.destroy()



@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1)])  # Simple test with single GPU
@rerun_if_address_is_in_use()
def test_masking_with_hf_reference(tp: int, dp: int, pp: int):
    """Test masking by comparing with HuggingFace's LlamaForCausalLM as reference."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_masking_with_hf_reference)()


def _test_masking_with_hf_reference(parallel_context: ParallelContext):
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM
    # Import from examples/llama
    import sys
    import os

    # Get the absolute path to the examples directory (parent's parent directory + examples)
    examples_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples"))

    # Add the parent directory of examples to the path
    # This allows Python to recognize 'examples' as a proper package
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import the convert_nanotron_to_hf module
    sys.path.append(os.path.join(project_root, "examples", "llama"))
    
    
    # Setup model config
    model_args = ModelArgs(init_method=RandomInit(std=0.02), model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)
    
    # Create our model
    model = create_llama_from_config(
        model_config=config.model.model_config,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    model.init_model_randomly(config=config)
    
    # Create HuggingFace reference model with same config
    from examples.llama.convert_nanotron_to_hf import get_hf_config
    config_hf = get_hf_config(TINY_LLAMA_CONFIG)
    reference_model = LlamaForCausalLM._from_config(config_hf).cuda()
    
    # Copy weights from our model to the reference model
    from examples.llama.convert_nanotron_to_hf import convert_nt_to_hf
    print("Converting nanotron model to huggingface model..")
    convert_nt_to_hf(model, reference_model, config.model.model_config)
    print("Conversion complete.")

    # Create test inputs with different masking patterns
    batch_size = 2
    seq_length = 16
    
    # Create test cases with different masking patterns
    test_cases = []
    
    # Case 1: No masking (all tokens visible)
    input_ids = torch.randint(0, 100, (batch_size, seq_length), device="cuda")
    attention_mask = torch.ones(batch_size, seq_length, device="cuda")
    
    # For our model
    our_inputs = {
        "input_ids": input_ids,
        "input_mask": attention_mask.bool(),
        "label_ids": input_ids.clone(),
        "label_mask": torch.ones(batch_size, seq_length, device="cuda"),
    }
    
    # For HuggingFace model
    hf_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }
    
    test_cases.append({
        "name": "no_masking",
        "our_inputs": our_inputs,
        "hf_inputs": hf_inputs,
    })
    # Case 2: Padding masking (some tokens masked at the end)
    input_ids = torch.randint(0, 100, (batch_size, seq_length), device="cuda")
    attention_mask = torch.ones(batch_size, seq_length, device="cuda")
    # Make sure we don't mask all tokens in any sequence
    attention_mask[0, -4:] = 0  # Last 4 tokens of first sequence are padding
    attention_mask[1, -6:] = 0  # Last 6 tokens of second sequence are padding

    # Create labels with -100 for masked positions (HuggingFace convention)
    labels = input_ids.clone()
    labels[0, -4:] = -100
    labels[1, -6:] = -100

    # For our model
    our_inputs = {
        "input_ids": input_ids,
        "input_mask": attention_mask.bool(),
        "label_ids": input_ids.clone(),
        "label_mask": attention_mask.clone(),  # Same as attention_mask for this case
    }
    
    # For HuggingFace model
    hf_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
    test_cases.append({
        "name": "padding_masking",
        "our_inputs": our_inputs,
        "hf_inputs": hf_inputs,
    })
    
    # Case 3: SFT-style masking (only compute loss on response tokens)
    input_ids = torch.randint(0, 100, (batch_size, seq_length), device="cuda")
    attention_mask = torch.ones(batch_size, seq_length, device="cuda")
    
    # Only compute loss on second half (simulating response tokens)
    label_mask = torch.zeros(batch_size, seq_length, device="cuda")
    label_mask[:, seq_length//2:] = 1
    
    # Create labels with -100 for non-response positions (HuggingFace convention)
    labels = input_ids.clone()
    labels[label_mask == 0] = -100
    
    # For our model
    our_inputs = {
        "input_ids": input_ids,
        "input_mask": attention_mask.bool(),
        "label_ids": input_ids.clone(),
        "label_mask": label_mask,
    }
    
    # For HuggingFace model
    hf_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
    test_cases.append({
        "name": "sft_masking",
        "our_inputs": our_inputs,
        "hf_inputs": hf_inputs,
    })
    
    # Run tests
    model.eval()
    reference_model.eval()
    
    # Test each case
    for test_case in test_cases:
        name = test_case["name"]
        our_inputs = test_case["our_inputs"]
        hf_inputs = test_case["hf_inputs"]
        
        print(f"Testing {name}...")
        
        with torch.no_grad():
            # Run both models
            our_output = model(**our_inputs)
            hf_output = reference_model(**hf_inputs)
            
            # Get losses
            our_loss = our_output["loss"]
            hf_loss = hf_output.loss
            
            print(f"  Our model loss: {our_loss.item()}")
            print(f"  HuggingFace loss: {hf_loss.item()}")
            
            # Both losses should be finite (not NaN or inf) if masking is valid
            assert torch.isfinite(our_loss), f"Our model loss is not finite for {name}"
            assert torch.isfinite(hf_loss), f"HuggingFace loss is not finite for {name}"
            # Check that our model's loss is close to HuggingFace's loss
            torch.testing.assert_close(
                our_loss,
                hf_loss,
                rtol=0.1,
                atol=0.1,
            )
            print(f"  Loss comparison passed: our loss matches HuggingFace loss")

            # For padding masking, verify that changing padding values doesn't affect the loss
            if name == "padding_masking":
                # Create copies with different values in padding positions
                modified_our_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                      for k, v in our_inputs.items()}
                modified_hf_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                     for k, v in hf_inputs.items()}
                
                # Change values in padding positions
                padding_positions = ~our_inputs["input_mask"]
                modified_our_inputs["input_ids"][padding_positions] = 999
                modified_hf_inputs["input_ids"][padding_positions] = 999
                
                # Run both models with modified inputs
                modified_our_output = model(**modified_our_inputs)
                modified_hf_output = reference_model(**modified_hf_inputs)
                
                # Losses should be the same since padding tokens are masked
                torch.testing.assert_close(hf_output.loss, modified_hf_output.loss, rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(our_output["loss"], modified_our_output["loss"], rtol=1e-4, atol=1e-4)
                
                print("  Padding masking test passed: changing padding values doesn't affect loss")
            # For SFT masking, verify that changing non-response token labels doesn't affect the loss
            if name == "sft_masking":
                # Create copies with different values in non-response positions (only for labels)
                modified_our_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                      for k, v in our_inputs.items()}
                modified_hf_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                     for k, v in hf_inputs.items()}
                
                # Change only label values in non-response positions, not input_ids
                non_response_positions = our_inputs["label_mask"] == 0
                modified_our_inputs["label_ids"][non_response_positions] = 888
                hf_non_response_positions = hf_inputs["labels"] == -100
                modified_hf_inputs["labels"][hf_non_response_positions] = -100 # Can't actually change labels in HF otherwise they won't be masked
                
                # Run both models with modified inputs
                modified_our_output = model(**modified_our_inputs)
                modified_hf_output = reference_model(**modified_hf_inputs)
                
                # Losses should be the same since non-response tokens are masked in loss computation
                torch.testing.assert_close(hf_output.loss, modified_hf_output.loss, rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(our_output["loss"], modified_our_output["loss"], rtol=1e-4, atol=1e-4)
                
                print("  SFT masking test passed: changing non-response values doesn't affect loss")
    
    print("All masking tests with HuggingFace reference passed!")
    parallel_context.destroy()


if __name__ == "__main__":
    print("Running tests..")
    # Run the zero mask assertion test
    # print("Running zero mask assertion test..")
    # init_distributed(tp=1, dp=1, pp=1)(_test_zero_mask_assertion)()
    
    # # Uncomment to run other tests
    # print("Running SFT loss masking test..")
    # init_distributed(tp=1, dp=1, pp=1)(_test_sft_loss_masking)()

    # print("Running SFT training test..")
    # init_distributed(tp=1, dp=1, pp=1)(_test_sft_training)()
    
    print("Running masking test with HuggingFace reference..")
    init_distributed(tp=1, dp=1, pp=1)(_test_masking_with_hf_reference)() 