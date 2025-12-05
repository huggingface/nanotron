"""Unit tests for ClimLlama model with hybrid positional embeddings.

Tests cover:
1. ClimLlamaConfig validation
2. ClimLlamaEmbedding layer
3. DataCollatorForCLMWithPositionIds
4. Hybrid PE (absolute + RoPE) verification

Tests requiring distributed context use the init_distributed pattern like other nanotron tests.
"""

import pytest
import torch
import numpy as np

from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config.models_config import ClimLlamaConfig
from nanotron.parallel import ParallelContext


class TestClimLlamaConfig:
    """Tests for ClimLlamaConfig dataclass (no GPU required)."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClimLlamaConfig()

        assert config.is_climllama_config is True
        assert config.use_absolute_position_embeddings is True
        assert config.use_spatial_temporal_encoding is True
        assert config.var_vocab_size == 13
        assert config.res_vocab_size == 12
        assert config.leadtime_vocab_size == 13
        assert config.spatial_temporal_encoding_dim == 128
        assert config.max_tp == 4
        assert len(config.variables) == config.var_vocab_size

    def test_variable_vocab_size_validation(self):
        """Test that variable vocab size must match variables tuple length."""
        with pytest.raises(ValueError, match="Number of variables"):
            ClimLlamaConfig(var_vocab_size=5, variables=("a", "b", "c"))

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_vars = ("unk", "temp", "pressure")
        config = ClimLlamaConfig(
            var_vocab_size=3,
            variables=custom_vars,
            res_vocab_size=8,
            leadtime_vocab_size=6,
            spatial_temporal_encoding_dim=64,
            hidden_size=512,
        )

        assert config.var_vocab_size == 3
        assert config.variables == custom_vars
        assert config.res_vocab_size == 8
        assert config.leadtime_vocab_size == 6
        assert config.spatial_temporal_encoding_dim == 64
        assert config.hidden_size == 512

    def test_inherits_from_qwen2(self):
        """Test that ClimLlamaConfig inherits Qwen2Config fields."""
        config = ClimLlamaConfig(
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            vocab_size=50000,
        )

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 16
        assert config.vocab_size == 50000

    def test_disable_absolute_position_embeddings(self):
        """Test config with absolute position embeddings disabled."""
        config = ClimLlamaConfig(use_absolute_position_embeddings=False)
        assert config.use_absolute_position_embeddings is False

    def test_disable_spatial_temporal_encoding(self):
        """Test config with spatial-temporal encoding disabled."""
        config = ClimLlamaConfig(use_spatial_temporal_encoding=False)
        assert config.use_spatial_temporal_encoding is False

    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        import dataclasses
        import json

        config = ClimLlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            max_position_embeddings=128,
            var_vocab_size=5,
            variables=("unk", "t", "q", "u", "v"),
            res_vocab_size=4,
            leadtime_vocab_size=6,
            spatial_temporal_encoding_dim=32,
            use_absolute_position_embeddings=True,
            use_spatial_temporal_encoding=True,
            _fused_rms_norm=False,
            _fused_rotary_emb=False,
        )

        # Convert to dict
        config_dict = dataclasses.asdict(config)

        # Simulate JSON round-trip
        json_str = json.dumps(config_dict)
        loaded_dict = json.loads(json_str)

        # Handle tuple conversion
        loaded_dict["variables"] = tuple(loaded_dict["variables"])

        # Remove private fields that are set in __post_init__
        private_fields = [k for k in loaded_dict.keys() if k.startswith('_')]
        for field in private_fields:
            if field not in ['_attn_implementation', '_fused_rms_norm', '_fused_rotary_emb',
                            '_use_qkv_packed', '_use_doc_masking']:
                loaded_dict.pop(field, None)

        # Recreate config
        new_config = ClimLlamaConfig(**loaded_dict)

        assert new_config.hidden_size == config.hidden_size
        assert new_config.var_vocab_size == config.var_vocab_size
        assert new_config.variables == config.variables


class TestSpatialTemporalFeatures:
    """Tests for spatial-temporal feature handling (no GPU required)."""

    def test_feature_dimensions(self):
        """Test that spatial-temporal features have correct dimensions."""
        # Features: [x, y, z, cos_hour, sin_hour, cos_day, sin_day]
        batch_size = 2
        seq_len = 16

        # Create sample features
        x = torch.rand(batch_size, seq_len, 1)  # Normalized x coordinate
        y = torch.rand(batch_size, seq_len, 1)  # Normalized y coordinate
        z = torch.rand(batch_size, seq_len, 1)  # Normalized z (pressure level)

        hour = torch.rand(batch_size, seq_len, 1) * 24  # Hour of day
        cos_hour = torch.cos(2 * np.pi * hour / 24)
        sin_hour = torch.sin(2 * np.pi * hour / 24)

        day = torch.rand(batch_size, seq_len, 1) * 365  # Day of year
        cos_day = torch.cos(2 * np.pi * day / 365)
        sin_day = torch.sin(2 * np.pi * day / 365)

        features = torch.cat([x, y, z, cos_hour, sin_hour, cos_day, sin_day], dim=-1)

        assert features.shape == (batch_size, seq_len, 7)

    def test_cyclic_encoding_values(self):
        """Test that cyclic encodings have correct value ranges."""
        # Hour 0 and hour 24 should produce same encoding
        hour_0 = torch.tensor([0.0])
        hour_24 = torch.tensor([24.0])

        cos_0 = torch.cos(2 * np.pi * hour_0 / 24)
        sin_0 = torch.sin(2 * np.pi * hour_0 / 24)

        cos_24 = torch.cos(2 * np.pi * hour_24 / 24)
        sin_24 = torch.sin(2 * np.pi * hour_24 / 24)

        assert torch.allclose(cos_0, cos_24, atol=1e-6)
        assert torch.allclose(sin_0, sin_24, atol=1e-6)

    def test_spatial_encoding_normalization(self):
        """Test that spatial coordinates should be normalized."""
        # Example: latitude [-90, 90] -> [-1, 1]
        lat = torch.tensor([-90.0, 0.0, 90.0])
        lat_normalized = lat / 90.0

        assert torch.all(lat_normalized >= -1)
        assert torch.all(lat_normalized <= 1)

        # Example: longitude [-180, 180] -> [-1, 1]
        lon = torch.tensor([-180.0, 0.0, 180.0])
        lon_normalized = lon / 180.0

        assert torch.all(lon_normalized >= -1)
        assert torch.all(lon_normalized <= 1)


# ============================================================================
# Tests requiring distributed context (GPU required, assumes 4 GPUs)
# ============================================================================

# Standard 4-GPU parallelism configurations for testing
PARALLEL_CONFIGS_4GPU = [(2, 2, 1), (2, 1, 2), (1, 2, 2), (4, 1, 1)]


def get_small_climllama_config():
    """Create a small ClimLlama config for testing."""
    # Note: vocab sizes must be divisible by max TP (4) for TensorParallelEmbedding
    return ClimLlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
        max_position_embeddings=128,
        var_vocab_size=8,
        variables=("unk", "t", "q", "u", "v", "w", "z", "sp"),
        res_vocab_size=4,
        leadtime_vocab_size=8,
        leadtimes=(0, 6, 12, 24, 48, 72, 96, 120),
        spatial_temporal_encoding_dim=32,
        use_absolute_position_embeddings=True,
        use_spatial_temporal_encoding=True,
        _fused_rms_norm=False,
        _fused_rotary_emb=False,
    )


# --- ClimLlamaEmbedding Tests ---

@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_embedding_output_shape(tp: int, dp: int, pp: int):
    """Test that embedding layer produces correct output shape."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_embedding_output_shape)()


def _test_embedding_output_shape(parallel_context: ParallelContext):
    from nanotron.models.climllama import ClimLlamaEmbedding

    config = get_small_climllama_config()

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    var_idx = torch.randint(0, config.var_vocab_size, (batch_size, seq_len), device="cuda")
    res_idx = torch.randint(0, config.res_vocab_size, (batch_size, seq_len), device="cuda")
    leadtime_idx = torch.randint(0, config.leadtime_vocab_size, (batch_size, seq_len), device="cuda")
    spatial_temporal_features = torch.randn(batch_size, seq_len, 7, device="cuda")

    output = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        var_idx=var_idx,
        res_idx=res_idx,
        leadtime_idx=leadtime_idx,
        spatial_temporal_features=spatial_temporal_features,
    )

    assert "input_embeds" in output
    assert "position_ids" in output
    assert output["input_embeds"].shape == (batch_size * seq_len, config.hidden_size)
    assert torch.equal(output["position_ids"], position_ids)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_embedding_without_discrete_positions(tp: int, dp: int, pp: int):
    """Test embedding layer when discrete position indices are None."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_embedding_without_discrete_positions)()


def _test_embedding_without_discrete_positions(parallel_context: ParallelContext):
    from nanotron.models.climllama import ClimLlamaEmbedding

    config = get_small_climllama_config()

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)

    # Call without discrete position indices
    output = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        var_idx=None,
        res_idx=None,
        leadtime_idx=None,
        spatial_temporal_features=None,
    )

    assert output["input_embeds"].shape == (batch_size * seq_len, config.hidden_size)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_embedding_only_with_spatial_temporal(tp: int, dp: int, pp: int):
    """Test embedding with only spatial-temporal encoding enabled."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_embedding_only_with_spatial_temporal)()


def _test_embedding_only_with_spatial_temporal(parallel_context: ParallelContext):
    from nanotron.models.climllama import ClimLlamaEmbedding

    config = ClimLlamaConfig(
        hidden_size=64,
        vocab_size=100,
        var_vocab_size=5,
        variables=("unk", "t", "q", "u", "v"),
        use_absolute_position_embeddings=False,
        use_spatial_temporal_encoding=True,
        spatial_temporal_encoding_dim=32,
    )

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    # Should not have discrete embedding layers
    assert not hasattr(embedding_layer, 'var_embedding')
    assert not hasattr(embedding_layer, 'res_embedding')
    assert not hasattr(embedding_layer, 'leadtime_embedding')

    # Should have spatial-temporal projection
    assert hasattr(embedding_layer, 'spatial_temporal_proj')
    assert hasattr(embedding_layer, 'spatial_temporal_proj2')

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_spatial_temporal_projection_dimensions(tp: int, dp: int, pp: int):
    """Test spatial-temporal projection layer dimensions."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_spatial_temporal_projection_dimensions)()


def _test_spatial_temporal_projection_dimensions(parallel_context: ParallelContext):
    from nanotron.models.climllama import ClimLlamaEmbedding

    # Note: vocab sizes must be divisible by max TP (4) for TensorParallelEmbedding
    config = ClimLlamaConfig(
        hidden_size=128,
        vocab_size=100,
        var_vocab_size=8,
        variables=("unk", "t", "q", "u", "v", "w", "z", "sp"),
        res_vocab_size=4,
        leadtime_vocab_size=8,
        spatial_temporal_encoding_dim=64,
        use_spatial_temporal_encoding=True,
    )

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    # Check projection layer dimensions
    assert embedding_layer.spatial_temporal_proj.in_features == 7
    assert embedding_layer.spatial_temporal_proj.out_features == 64
    assert embedding_layer.spatial_temporal_proj2.in_features == 64
    assert embedding_layer.spatial_temporal_proj2.out_features == 128

    parallel_context.destroy()


# --- Hybrid Positional Embedding Tests ---

@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_embedding_combines_all_components(tp: int, dp: int, pp: int):
    """Test that embedding combines token, discrete, and spatial-temporal embeddings."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_embedding_combines_all_components)()


def _test_embedding_combines_all_components(parallel_context: ParallelContext):
    from nanotron.models.climllama import ClimLlamaEmbedding

    config = get_small_climllama_config()

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    batch_size = 1
    seq_len = 4

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
    var_idx = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
    res_idx = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
    leadtime_idx = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
    spatial_temporal_features = torch.ones(batch_size, seq_len, 7, device="cuda")

    output = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        var_idx=var_idx,
        res_idx=res_idx,
        leadtime_idx=leadtime_idx,
        spatial_temporal_features=spatial_temporal_features,
    )

    # Verify the embedding is non-zero and has expected shape
    embeds = output["input_embeds"]
    assert embeds.shape == (batch_size * seq_len, config.hidden_size)
    assert not torch.all(embeds == 0)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_position_ids_passed_through(tp: int, dp: int, pp: int):
    """Test that position_ids are passed through for RoPE in attention layers."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_position_ids_passed_through)()


def _test_position_ids_passed_through(parallel_context: ParallelContext):
    from nanotron.models.climllama import ClimLlamaEmbedding

    config = get_small_climllama_config()

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)

    output = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
    )

    # Position IDs should be returned unchanged for use in RoPE
    assert torch.equal(output["position_ids"], position_ids)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_different_position_ids_per_batch(tp: int, dp: int, pp: int):
    """Test handling of different position IDs per batch element."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_different_position_ids_per_batch)()


def _test_different_position_ids_per_batch(parallel_context: ParallelContext):
    from nanotron.models.climllama import ClimLlamaEmbedding

    config = get_small_climllama_config()

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")

    # Different position IDs per batch (e.g., different document starts)
    position_ids = torch.stack([
        torch.arange(seq_len, device="cuda"),
        torch.cat([torch.arange(4, device="cuda"), torch.arange(4, device="cuda")]),  # Two documents
    ])

    output = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
    )

    # Should handle different position patterns per batch element
    assert output["input_embeds"].shape == (batch_size * seq_len, config.hidden_size)
    assert torch.equal(output["position_ids"], position_ids)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_embedding_forward_backward(tp: int, dp: int, pp: int):
    """Test forward and backward pass through embedding layer."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_embedding_forward_backward)()


def _test_embedding_forward_backward(parallel_context: ParallelContext):
    from nanotron.models.climllama import ClimLlamaEmbedding

    config = get_small_climllama_config()

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    var_idx = torch.randint(0, config.var_vocab_size, (batch_size, seq_len), device="cuda")
    res_idx = torch.randint(0, config.res_vocab_size, (batch_size, seq_len), device="cuda")
    leadtime_idx = torch.randint(0, config.leadtime_vocab_size, (batch_size, seq_len), device="cuda")
    spatial_temporal_features = torch.randn(batch_size, seq_len, 7, device="cuda", requires_grad=True)

    output = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        var_idx=var_idx,
        res_idx=res_idx,
        leadtime_idx=leadtime_idx,
        spatial_temporal_features=spatial_temporal_features,
    )

    # Backward pass
    loss = output["input_embeds"].sum()
    loss.backward()

    # Check gradients flow through
    assert spatial_temporal_features.grad is not None
    assert embedding_layer.spatial_temporal_proj.weight.grad is not None

    parallel_context.destroy()


# --- DataCollator Tests ---

@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_collator_output_structure(tp: int, dp: int, pp: int):
    """Test that collator produces correct output structure."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_collator_output_structure)()


def _test_collator_output_structure(parallel_context: ParallelContext):
    from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
    from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
    from nanotron import distributed as dist

    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    collator = DataCollatorForCLMWithPositionIds(
        sequence_length=16,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=parallel_context,
    )

    if current_pp_rank == 0:
        # Create sample data for participating rank
        examples = [
            {
                "input_ids": np.arange(17),  # seq_len + 1
                "positions": np.arange(17),
            },
            {
                "input_ids": np.arange(17),
                "positions": np.arange(17),
            },
        ]

        result = collator(examples)

        assert "input_ids" in result
        assert "position_ids" in result
        assert "label_ids" in result
        assert "label_mask" in result
    else:
        # Non-participating ranks should provide empty examples
        examples = [{}, {}]
        result = collator(examples)

        # Should return TensorPointers for non-participating ranks
        assert isinstance(result["input_ids"], TensorPointer)
        assert isinstance(result["label_ids"], TensorPointer)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_collator_position_ids_shape(tp: int, dp: int, pp: int):
    """Test position IDs shape in collator output."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_collator_position_ids_shape)()


def _test_collator_position_ids_shape(parallel_context: ParallelContext):
    from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
    from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
    from nanotron import distributed as dist

    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    seq_len = 32
    batch_size = 4

    collator = DataCollatorForCLMWithPositionIds(
        sequence_length=seq_len,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=parallel_context,
    )

    if current_pp_rank == 0:
        examples = [
            {
                "input_ids": np.arange(seq_len + 1),
                "positions": np.arange(seq_len + 1),
            }
            for _ in range(batch_size)
        ]

        result = collator(examples)

        assert result["input_ids"].shape == (batch_size, seq_len)
        assert result["position_ids"].shape == (batch_size, seq_len)
    else:
        # Non-participating ranks should provide empty examples
        examples = [{} for _ in range(batch_size)]
        result = collator(examples)

        # Should return TensorPointers for non-participating ranks
        assert isinstance(result["input_ids"], TensorPointer)
        assert isinstance(result["label_ids"], TensorPointer)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_collator_label_mask_with_doc_boundaries(tp: int, dp: int, pp: int):
    """Test label mask handles document boundaries correctly."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_collator_label_mask_with_doc_boundaries)()


def _test_collator_label_mask_with_doc_boundaries(parallel_context: ParallelContext):
    from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
    from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
    from nanotron import distributed as dist

    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    seq_len = 16

    collator = DataCollatorForCLMWithPositionIds(
        sequence_length=seq_len,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=parallel_context,
        use_doc_masking=True,
    )

    if current_pp_rank == 0:
        # Create example with document boundary at position 8
        # Position IDs: 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7,0
        positions = np.concatenate([np.arange(8), np.arange(9)])

        examples = [
            {
                "input_ids": np.arange(seq_len + 1),
                "positions": positions,
            }
        ]

        result = collator(examples)

        # Label mask should be False at document boundaries
        label_mask = result["label_mask"]
        assert label_mask.dtype == np.bool_

        # Position 7 should be masked (before the 0 at position 8)
        # After shifting, the label at index 7 corresponds to position 8 (which is 0)
        assert label_mask[0, 7] == False  # Document boundary
    else:
        # Non-participating ranks should provide empty examples
        examples = [{}]
        result = collator(examples)

        # Should return TensorPointers for non-participating ranks
        assert isinstance(result["label_mask"], TensorPointer)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", PARALLEL_CONFIGS_4GPU)
@rerun_if_address_is_in_use()
def test_collator_without_doc_masking(tp: int, dp: int, pp: int):
    """Test collator without document masking."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_collator_without_doc_masking)()


def _test_collator_without_doc_masking(parallel_context: ParallelContext):
    from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
    from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
    from nanotron import distributed as dist

    seq_len = 16
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    collator = DataCollatorForCLMWithPositionIds(
        sequence_length=seq_len,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=parallel_context,
        use_doc_masking=False,
    )

    if current_pp_rank == 0:
        examples = [
            {
                "input_ids": np.arange(seq_len + 1),
                "positions": np.arange(seq_len + 1),
            }
        ]

        result = collator(examples)

        # Without doc masking, all labels should be used
        assert np.all(result["label_mask"])
    else:
        # Non-participating ranks should provide empty examples
        examples = [{}]
        result = collator(examples)

        # Should return TensorPointers for non-participating ranks
        assert isinstance(result["label_mask"], TensorPointer)

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [(2, 1, 2), (1, 2, 2)])
@rerun_if_address_is_in_use()
def test_collator_non_participating_rank(tp: int, dp: int, pp: int):
    """Test collator behavior for non-participating PP ranks."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_collator_non_participating_rank)()


def _test_collator_non_participating_rank(parallel_context: ParallelContext):
    from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
    from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
    from nanotron import distributed as dist

    # This test runs with pp=2, so rank 1 is not the input rank (0) nor output rank (0)
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    collator = DataCollatorForCLMWithPositionIds(
        sequence_length=16,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=parallel_context,
    )

    if current_pp_rank == 0:
        # Rank 0 is both input and output rank
        examples = [
            {
                "input_ids": np.arange(17),
                "positions": np.arange(17),
            },
            {
                "input_ids": np.arange(17),
                "positions": np.arange(17),
            },
        ]
        result = collator(examples)

        # Should return actual tensors
        assert not isinstance(result["input_ids"], TensorPointer)
        assert not isinstance(result["label_ids"], TensorPointer)
    else:
        # Other ranks don't participate
        examples = [{}, {}]
        result = collator(examples)

        # Should return TensorPointers (collator returns "positions" for non-participating ranks)
        assert isinstance(result["input_ids"], TensorPointer)
        assert isinstance(result["positions"], TensorPointer)
        assert isinstance(result["label_ids"], TensorPointer)
        assert isinstance(result["label_mask"], TensorPointer)

    parallel_context.destroy()


# --- Full Model Forward Pass Integration Tests ---

@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1)])
@rerun_if_address_is_in_use()
def test_full_model_forward_pass(tp: int, dp: int, pp: int):
    """Integration test: Full ClimLlama model forward pass with hybrid PE."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_full_model_forward_pass)()


def _test_full_model_forward_pass(parallel_context: ParallelContext):
    """Test full model forward pass with all positional embedding components."""
    from math import ceil
    from nanotron.models.climllama import ClimLlamaForTraining
    from nanotron.config import ParallelismArgs
    from nanotron.parallel.pipeline_parallel.block import PipelineBlock

    config = get_small_climllama_config()

    # Create parallel config
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
    )

    # Initialize model
    model = ClimLlamaForTraining(
        config=config,
        parallel_context=parallel_context,
        parallel_config=parallel_config,
        random_states=None,
    )

    # Build and set rank for all pipeline blocks
    pipeline_blocks = [module for name, module in model.named_modules() if isinstance(module, PipelineBlock)]
    contiguous_size = ceil(len(pipeline_blocks) / parallel_context.pp_pg.size())
    for i, block in enumerate(pipeline_blocks):
        rank = i // contiguous_size
        block.build_and_set_rank(rank)

    model = model.to("cuda").to(torch.bfloat16)

    batch_size = 2
    seq_len = 16

    # Prepare inputs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    var_idx = torch.randint(0, config.var_vocab_size, (batch_size, seq_len), device="cuda")
    res_idx = torch.randint(0, config.res_vocab_size, (batch_size, seq_len), device="cuda")
    leadtime_idx = torch.randint(0, config.leadtime_vocab_size, (batch_size, seq_len), device="cuda")
    spatial_temporal_features = torch.randn(batch_size, seq_len, 7, device="cuda", dtype=torch.bfloat16)
    label_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    label_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device="cuda")

    # Forward pass
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

    # Verify output structure and shapes
    assert "loss" in output

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [(2, 1, 1), (1, 2, 1)])
@rerun_if_address_is_in_use()
def test_full_model_forward_pass_parallel(tp: int, dp: int, pp: int):
    """Integration test: Full ClimLlama model forward pass with parallelism."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_full_model_forward_pass_parallel)()


def _test_full_model_forward_pass_parallel(parallel_context: ParallelContext):
    """Test full model forward pass with tensor/data parallelism."""
    from math import ceil
    from nanotron.models.climllama import ClimLlamaForTraining
    from nanotron.config import ParallelismArgs
    from nanotron import distributed as dist
    from nanotron.parallel.pipeline_parallel.block import PipelineBlock

    config = get_small_climllama_config()

    tp_size = dist.get_world_size(parallel_context.tp_pg)
    dp_size = dist.get_world_size(parallel_context.dp_pg)

    parallel_config = ParallelismArgs(
        dp=dp_size,
        pp=1,
        tp=tp_size,
    )

    model = ClimLlamaForTraining(
        config=config,
        parallel_context=parallel_context,
        parallel_config=parallel_config,
        random_states=None,
    )

    # Build and set rank for all pipeline blocks
    pipeline_blocks = [module for name, module in model.named_modules() if isinstance(module, PipelineBlock)]
    contiguous_size = ceil(len(pipeline_blocks) / parallel_context.pp_pg.size())
    for i, block in enumerate(pipeline_blocks):
        rank = i // contiguous_size
        block.build_and_set_rank(rank)

    model = model.to("cuda").to(torch.bfloat16)

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    var_idx = torch.randint(0, config.var_vocab_size, (batch_size, seq_len), device="cuda")
    res_idx = torch.randint(0, config.res_vocab_size, (batch_size, seq_len), device="cuda")
    leadtime_idx = torch.randint(0, config.leadtime_vocab_size, (batch_size, seq_len), device="cuda")
    spatial_temporal_features = torch.randn(batch_size, seq_len, 7, device="cuda", dtype=torch.bfloat16)
    label_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    label_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device="cuda")

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

    # Verify forward pass completes without error
    assert "loss" in output

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1)])
@rerun_if_address_is_in_use()
def test_hybrid_pe_absolute_plus_rope(tp: int, dp: int, pp: int):
    """Verify hybrid PE: absolute position embeddings combined with RoPE."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_hybrid_pe_absolute_plus_rope)()


def _test_hybrid_pe_absolute_plus_rope(parallel_context: ParallelContext):
    """
    Verify that the model uses both absolute position embeddings and RoPE together.

    The hybrid approach:
    1. Absolute embeddings (var, res, leadtime, spatial-temporal) are added to token embeddings
    2. RoPE is applied in attention layers using position_ids
    """
    from nanotron.models.climllama import ClimLlamaEmbedding

    config = get_small_climllama_config()

    embedding_layer = ClimLlamaEmbedding(
        tp_pg=parallel_context.tp_pg,
        config=config,
        parallel_config=None,
    ).to("cuda")

    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    var_idx = torch.randint(0, config.var_vocab_size, (batch_size, seq_len), device="cuda")
    res_idx = torch.randint(0, config.res_vocab_size, (batch_size, seq_len), device="cuda")
    leadtime_idx = torch.randint(0, config.leadtime_vocab_size, (batch_size, seq_len), device="cuda")
    spatial_temporal_features = torch.randn(batch_size, seq_len, 7, device="cuda")

    # Test 1: Verify embedding output contains position_ids for RoPE
    output = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        var_idx=var_idx,
        res_idx=res_idx,
        leadtime_idx=leadtime_idx,
        spatial_temporal_features=spatial_temporal_features,
    )

    # Position IDs should be passed through for RoPE in attention layers
    assert "position_ids" in output
    assert torch.equal(output["position_ids"], position_ids)

    # Test 2: Verify absolute embeddings affect the output
    # Run with different var_idx and verify embeddings differ
    var_idx_alt = (var_idx + 1) % config.var_vocab_size

    output_alt = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        var_idx=var_idx_alt,
        res_idx=res_idx,
        leadtime_idx=leadtime_idx,
        spatial_temporal_features=spatial_temporal_features,
    )

    # Embeddings should differ when absolute position indices differ
    assert not torch.allclose(output["input_embeds"], output_alt["input_embeds"])

    # Test 3: Verify spatial-temporal features affect the output
    spatial_temporal_features_alt = spatial_temporal_features + 1.0

    output_st_alt = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        var_idx=var_idx,
        res_idx=res_idx,
        leadtime_idx=leadtime_idx,
        spatial_temporal_features=spatial_temporal_features_alt,
    )

    # Embeddings should differ when spatial-temporal features differ
    assert not torch.allclose(output["input_embeds"], output_st_alt["input_embeds"])

    parallel_context.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
