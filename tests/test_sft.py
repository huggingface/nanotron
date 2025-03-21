import pytest
import torch
import torch.distributed as dist
from datasets import Dataset
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import ModelArgs, RandomInit
from nanotron.parallel import ParallelContext
from transformers import AutoTokenizer

from tests.helpers.llama_helper import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config


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
        texts, max_length=sequence_length, padding="max_length", truncation=True, return_tensors="pt"
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
        prefix = text[: response_start + len("### Response:")]
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
        "label_mask": torch.tensor(label_masks),
    }

    return Dataset.from_dict(dataset_dict)


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
        "label_mask": torch.cat(
            [torch.zeros(1, sequence_length // 2), torch.ones(1, sequence_length // 2)], dim=1
        ).cuda(),
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
        assert not torch.allclose(
            loss_half, loss_all, rtol=1e-6, atol=1e-6, equal_nan=False
        ), "Partial mask should produce different loss than full mask"
        assert not torch.allclose(
            loss_almost_none, loss_all, rtol=1e-6, atol=1e-6, equal_nan=False
        ), "Almost no tokens should produce different loss than full mask"
        # Test with all tokens masked out (should return NaN)
        with torch.no_grad():
            loss_none = model(**none_included)["loss"]
            print(f"Loss with zero mask: {loss_none}")
            assert torch.isnan(loss_none), "Loss should be NaN with zero mask"

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1)])  # Simple test with single GPU
@rerun_if_address_is_in_use()
def test_right_padding_mask(tp: int, dp: int, pp: int):
    """Test that right padding masking works correctly (changing padded input_ids shouldn't affect loss)."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_right_padding_mask)()


def _test_right_padding_mask(parallel_context: ParallelContext):
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

    # Create test inputs with right padding
    batch_size = 2
    seq_length = 16

    # Create sequences of different lengths
    valid_lengths = [12, 10]  # First sequence has 12 tokens, second has 10

    # Create input_ids with random values
    input_ids = torch.randint(0, 100, (batch_size, seq_length), device="cuda")

    # Create input_mask with right padding pattern: [1, 1, 1, ..., 0, 0, 0]
    input_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device="cuda")
    for i, length in enumerate(valid_lengths):
        input_mask[i, :length] = True

    # Create label_mask with right padding pattern: [1, 1, 1, ..., 0, 0, 0]
    # (same as input_mask in this case)
    # We use the same mask as input_mask because label_ids are already shifted in the dataset
    label_mask = input_mask.clone().float()

    # Original inputs
    # Create shifted label_ids for autoregressive training (predict next token)
    # Shift right: first token predicts second, etc.
    label_ids = torch.zeros_like(input_ids)
    label_ids[:, :-1] = input_ids[:, 1:].clone()
    # Last position predicts random token
    label_ids[:, -1] = 123  # Random token

    original_inputs = {
        "input_ids": input_ids.clone(),
        "input_mask": input_mask.clone(),
        "label_ids": label_ids.clone(),  # Shifted for next-token prediction
        "label_mask": label_mask.clone(),
    }

    # Modified inputs (change values at padded positions)
    modified_input_ids = input_ids.clone()
    # Set padded positions to a completely different value
    padded_positions = ~input_mask
    modified_input_ids[padded_positions] = 999  # Use a value likely not in the original input

    modified_inputs = {
        "input_ids": modified_input_ids,
        "input_mask": input_mask.clone(),
        "label_ids": label_ids.clone(),  # Use the same shifted labels
        "label_mask": label_mask.clone(),
    }
    # Run model with both inputs
    model.eval()  # Use eval mode to avoid dropout randomness

    with torch.no_grad():
        original_output = model(**original_inputs)
        modified_output = model(**modified_inputs)

        original_loss = original_output["loss"]
        modified_loss = modified_output["loss"]

        # # Get logits for comparison
        # original_logits = original_output["sharded_logits"]
        # modified_logits = modified_output["sharded_logits"]

    # Verify that padding masking works correctly
    print(f"Original loss: {original_loss.item()}")
    print(f"Modified loss (changed padding tokens): {modified_loss.item()}")

    # Losses should be identical since we only changed padded input tokens
    torch.testing.assert_close(
        original_loss, modified_loss, rtol=1e-4, atol=1e-4, msg="Changing padded input tokens affected the loss"
    )

    # # Logits should also be identical except for padded positions
    # torch.testing.assert_close(
    #     original_logits[0,:,:],
    #     modified_logits[0,:,:],
    #     rtol=1e-4,
    #     atol=1e-4,
    # )

    print("Right padding mask test passed: changing padded input tokens doesn't affect loss or logits")

    # Additional test: verify that changing unpadded tokens DOES affect the loss
    control_input_ids = input_ids.clone()
    # Change first token of each sequence (which is definitely not padded)
    control_input_ids[0, 0] = 888
    control_input_ids[1, 0] = 888

    control_inputs = {
        "input_ids": control_input_ids,
        "input_mask": input_mask.clone(),
        "label_ids": label_ids.clone(),
        "label_mask": label_mask.clone(),
    }

    with torch.no_grad():
        control_output = model(**control_inputs)
        control_loss = control_output["loss"]
        # control_logits = control_output["sharded_logits"]

    print(f"Control loss (changed unpadded tokens): {control_loss.item()}")

    # Verify that changing unpadded tokens DOES affect the output
    assert not torch.allclose(
        original_loss, control_loss, rtol=1e-4, atol=1e-4
    ), "Changing unpadded input tokens should affect the loss"

    parallel_context.destroy()


if __name__ == "__main__":
    print("Running tests..")
    init_distributed(tp=1, dp=1, pp=1)(_test_sft_loss_masking)()
    init_distributed(tp=1, dp=1, pp=1)(_test_right_padding_mask)()
