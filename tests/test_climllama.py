"""Unit tests for ClimLlama model with hybrid positional embeddings.

Tests cover:
1. ClimLlamaEmbedding layer
2. DataCollatorForCLMWithPositionIds
3. Integration test for full forward pass
4. Hybrid PE (absolute + RoPE) verification
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from nanotron.config.models_config import ClimLlamaConfig


class TestClimLlamaConfig:
    """Tests for ClimLlamaConfig dataclass."""

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


class TestClimLlamaEmbedding:
    """Tests for ClimLlamaEmbedding layer."""

    @pytest.fixture
    def mock_process_group(self):
        """Create a mock process group for tensor parallelism."""
        pg = MagicMock()
        pg.size.return_value = 1
        pg.rank.return_value = 0
        return pg

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ClimLlamaConfig(
            hidden_size=64,
            vocab_size=100,
            var_vocab_size=5,
            variables=("unk", "t", "q", "u", "v"),
            res_vocab_size=4,
            leadtime_vocab_size=6,
            spatial_temporal_encoding_dim=32,
            use_absolute_position_embeddings=True,
            use_spatial_temporal_encoding=True,
        )

    def test_embedding_output_shape(self, config, mock_process_group):
        """Test that embedding layer produces correct output shape."""
        from nanotron.models.climllama import ClimLlamaEmbedding

        with patch('nanotron.parallel.tensor_parallel.nn.TensorParallelEmbedding') as MockTPEmbed:
            # Mock TensorParallelEmbedding to return regular embeddings
            def create_mock_embedding(num_embeddings, embedding_dim, **kwargs):
                embed = torch.nn.Embedding(num_embeddings, embedding_dim)
                return embed

            MockTPEmbed.side_effect = create_mock_embedding

            embedding_layer = ClimLlamaEmbedding(
                tp_pg=mock_process_group,
                config=config,
                parallel_config=None,
            )

            batch_size = 2
            seq_len = 16

            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            var_idx = torch.randint(0, config.var_vocab_size, (batch_size, seq_len))
            res_idx = torch.randint(0, config.res_vocab_size, (batch_size, seq_len))
            leadtime_idx = torch.randint(0, config.leadtime_vocab_size, (batch_size, seq_len))
            spatial_temporal_features = torch.randn(batch_size, seq_len, 7)

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

    def test_embedding_without_discrete_positions(self, config, mock_process_group):
        """Test embedding layer when discrete position indices are None."""
        from nanotron.models.climllama import ClimLlamaEmbedding

        with patch('nanotron.parallel.tensor_parallel.nn.TensorParallelEmbedding') as MockTPEmbed:
            def create_mock_embedding(num_embeddings, embedding_dim, **kwargs):
                return torch.nn.Embedding(num_embeddings, embedding_dim)

            MockTPEmbed.side_effect = create_mock_embedding

            embedding_layer = ClimLlamaEmbedding(
                tp_pg=mock_process_group,
                config=config,
                parallel_config=None,
            )

            batch_size = 2
            seq_len = 16

            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

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

    def test_embedding_only_with_spatial_temporal(self, mock_process_group):
        """Test embedding with only spatial-temporal encoding enabled."""
        config = ClimLlamaConfig(
            hidden_size=64,
            vocab_size=100,
            var_vocab_size=5,
            variables=("unk", "t", "q", "u", "v"),
            use_absolute_position_embeddings=False,
            use_spatial_temporal_encoding=True,
            spatial_temporal_encoding_dim=32,
        )

        from nanotron.models.climllama import ClimLlamaEmbedding

        with patch('nanotron.parallel.tensor_parallel.nn.TensorParallelEmbedding') as MockTPEmbed:
            def create_mock_embedding(num_embeddings, embedding_dim, **kwargs):
                return torch.nn.Embedding(num_embeddings, embedding_dim)

            MockTPEmbed.side_effect = create_mock_embedding

            embedding_layer = ClimLlamaEmbedding(
                tp_pg=mock_process_group,
                config=config,
                parallel_config=None,
            )

            # Should not have discrete embedding layers
            assert not hasattr(embedding_layer, 'var_embedding')
            assert not hasattr(embedding_layer, 'res_embedding')
            assert not hasattr(embedding_layer, 'leadtime_embedding')

            # Should have spatial-temporal projection
            assert hasattr(embedding_layer, 'spatial_temporal_proj')
            assert hasattr(embedding_layer, 'spatial_temporal_proj2')

    def test_spatial_temporal_projection_dimensions(self, mock_process_group):
        """Test spatial-temporal projection layer dimensions."""
        config = ClimLlamaConfig(
            hidden_size=128,
            vocab_size=100,
            var_vocab_size=5,
            variables=("unk", "t", "q", "u", "v"),
            spatial_temporal_encoding_dim=64,
            use_spatial_temporal_encoding=True,
        )

        from nanotron.models.climllama import ClimLlamaEmbedding

        with patch('nanotron.parallel.tensor_parallel.nn.TensorParallelEmbedding') as MockTPEmbed:
            def create_mock_embedding(num_embeddings, embedding_dim, **kwargs):
                return torch.nn.Embedding(num_embeddings, embedding_dim)

            MockTPEmbed.side_effect = create_mock_embedding

            embedding_layer = ClimLlamaEmbedding(
                tp_pg=mock_process_group,
                config=config,
                parallel_config=None,
            )

            # Check projection layer dimensions
            assert embedding_layer.spatial_temporal_proj.in_features == 7
            assert embedding_layer.spatial_temporal_proj.out_features == 64
            assert embedding_layer.spatial_temporal_proj2.in_features == 64
            assert embedding_layer.spatial_temporal_proj2.out_features == 128


class TestDataCollatorForCLMWithPositionIds:
    """Tests for DataCollatorForCLMWithPositionIds."""

    @pytest.fixture
    def mock_parallel_context(self):
        """Create a mock parallel context."""
        pc = MagicMock()
        pc.pp_pg = MagicMock()
        pc.cp_pg = MagicMock()
        pc.context_parallel_size = 1
        return pc

    def test_collator_output_structure(self, mock_parallel_context):
        """Test that collator produces correct output structure."""
        from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
        from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer

        # Set up mock to return input rank
        with patch('nanotron.distributed.get_rank') as mock_get_rank:
            mock_get_rank.return_value = 0  # Input PP rank

            collator = DataCollatorForCLMWithPositionIds(
                sequence_length=16,
                input_pp_rank=0,
                output_pp_rank=0,
                parallel_context=mock_parallel_context,
            )

            # Create sample data
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

    def test_collator_position_ids_shape(self, mock_parallel_context):
        """Test position IDs shape in collator output."""
        from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds

        with patch('nanotron.distributed.get_rank') as mock_get_rank:
            mock_get_rank.return_value = 0

            seq_len = 32
            batch_size = 4

            collator = DataCollatorForCLMWithPositionIds(
                sequence_length=seq_len,
                input_pp_rank=0,
                output_pp_rank=0,
                parallel_context=mock_parallel_context,
            )

            examples = [
                {
                    "input_ids": np.arange(seq_len + 1),
                    "positions": np.arange(seq_len + 1),
                }
                for _ in range(batch_size)
            ]

            result = collator(examples)

            assert result["input_ids"].shape == (batch_size, seq_len)
            assert result["position_ids"].shape == (batch_size, seq_len + 1)

    def test_collator_label_mask_with_doc_boundaries(self, mock_parallel_context):
        """Test label mask handles document boundaries correctly."""
        from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds

        with patch('nanotron.distributed.get_rank') as mock_get_rank:
            mock_get_rank.return_value = 0

            seq_len = 16

            collator = DataCollatorForCLMWithPositionIds(
                sequence_length=seq_len,
                input_pp_rank=0,
                output_pp_rank=0,
                parallel_context=mock_parallel_context,
                use_doc_masking=True,
            )

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

    def test_collator_without_doc_masking(self, mock_parallel_context):
        """Test collator without document masking."""
        from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds

        with patch('nanotron.distributed.get_rank') as mock_get_rank:
            mock_get_rank.return_value = 0

            seq_len = 16

            collator = DataCollatorForCLMWithPositionIds(
                sequence_length=seq_len,
                input_pp_rank=0,
                output_pp_rank=0,
                parallel_context=mock_parallel_context,
                use_doc_masking=False,
            )

            examples = [
                {
                    "input_ids": np.arange(seq_len + 1),
                    "positions": np.arange(seq_len + 1),
                }
            ]

            result = collator(examples)

            # Without doc masking, all labels should be used
            assert np.all(result["label_mask"])

    def test_collator_non_participating_rank(self, mock_parallel_context):
        """Test collator behavior for non-participating PP ranks."""
        from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
        from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer

        with patch('nanotron.distributed.get_rank') as mock_get_rank:
            mock_get_rank.return_value = 2  # Non-participating rank

            collator = DataCollatorForCLMWithPositionIds(
                sequence_length=16,
                input_pp_rank=0,
                output_pp_rank=1,
                parallel_context=mock_parallel_context,
            )

            # Empty examples for non-participating rank
            examples = [{}, {}]

            result = collator(examples)

            # Should return TensorPointers
            assert isinstance(result["input_ids"], TensorPointer)
            assert isinstance(result["position_ids"], TensorPointer)
            assert isinstance(result["label_ids"], TensorPointer)
            assert isinstance(result["label_mask"], TensorPointer)


class TestHybridPositionalEmbedding:
    """Tests for hybrid positional embedding (absolute + RoPE)."""

    @pytest.fixture
    def mock_process_group(self):
        """Create a mock process group."""
        pg = MagicMock()
        pg.size.return_value = 1
        pg.rank.return_value = 0
        return pg

    def test_embedding_combines_all_components(self, mock_process_group):
        """Test that embedding combines token, discrete, and spatial-temporal embeddings."""
        config = ClimLlamaConfig(
            hidden_size=64,
            vocab_size=100,
            var_vocab_size=5,
            variables=("unk", "t", "q", "u", "v"),
            res_vocab_size=4,
            leadtime_vocab_size=6,
            spatial_temporal_encoding_dim=32,
            use_absolute_position_embeddings=True,
            use_spatial_temporal_encoding=True,
        )

        from nanotron.models.climllama import ClimLlamaEmbedding

        with patch('nanotron.parallel.tensor_parallel.nn.TensorParallelEmbedding') as MockTPEmbed:
            def create_mock_embedding(num_embeddings, embedding_dim, **kwargs):
                embed = torch.nn.Embedding(num_embeddings, embedding_dim)
                # Initialize with different values to verify combination
                torch.nn.init.constant_(embed.weight, 1.0)
                return embed

            MockTPEmbed.side_effect = create_mock_embedding

            embedding_layer = ClimLlamaEmbedding(
                tp_pg=mock_process_group,
                config=config,
                parallel_config=None,
            )

            # Initialize spatial-temporal projections with known values
            torch.nn.init.constant_(embedding_layer.spatial_temporal_proj.weight, 0.1)
            torch.nn.init.constant_(embedding_layer.spatial_temporal_proj.bias, 0.0)
            torch.nn.init.constant_(embedding_layer.spatial_temporal_proj2.weight, 0.1)
            torch.nn.init.constant_(embedding_layer.spatial_temporal_proj2.bias, 0.0)

            batch_size = 1
            seq_len = 4

            input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
            position_ids = torch.arange(seq_len).unsqueeze(0)
            var_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
            res_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
            leadtime_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
            spatial_temporal_features = torch.ones(batch_size, seq_len, 7)

            output = embedding_layer(
                input_ids=input_ids,
                position_ids=position_ids,
                var_idx=var_idx,
                res_idx=res_idx,
                leadtime_idx=leadtime_idx,
                spatial_temporal_features=spatial_temporal_features,
            )

            # Embedding should be the sum of all components
            # token_embed (1.0 * hidden_size) + var_embed (1.0) + res_embed (1.0) + leadtime_embed (1.0) + spatial
            embeds = output["input_embeds"]

            # Verify the embedding is non-zero and has expected shape
            assert embeds.shape == (batch_size * seq_len, config.hidden_size)
            assert not torch.all(embeds == 0)

    def test_position_ids_passed_through(self, mock_process_group):
        """Test that position_ids are passed through for RoPE in attention layers."""
        config = ClimLlamaConfig(
            hidden_size=64,
            vocab_size=100,
            var_vocab_size=5,
            variables=("unk", "t", "q", "u", "v"),
        )

        from nanotron.models.climllama import ClimLlamaEmbedding

        with patch('nanotron.parallel.tensor_parallel.nn.TensorParallelEmbedding') as MockTPEmbed:
            def create_mock_embedding(num_embeddings, embedding_dim, **kwargs):
                return torch.nn.Embedding(num_embeddings, embedding_dim)

            MockTPEmbed.side_effect = create_mock_embedding

            embedding_layer = ClimLlamaEmbedding(
                tp_pg=mock_process_group,
                config=config,
                parallel_config=None,
            )

            batch_size = 2
            seq_len = 8

            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

            output = embedding_layer(
                input_ids=input_ids,
                position_ids=position_ids,
            )

            # Position IDs should be returned unchanged for use in RoPE
            assert torch.equal(output["position_ids"], position_ids)

    def test_different_position_ids_per_batch(self, mock_process_group):
        """Test handling of different position IDs per batch element."""
        config = ClimLlamaConfig(
            hidden_size=64,
            vocab_size=100,
            var_vocab_size=5,
            variables=("unk", "t", "q", "u", "v"),
        )

        from nanotron.models.climllama import ClimLlamaEmbedding

        with patch('nanotron.parallel.tensor_parallel.nn.TensorParallelEmbedding') as MockTPEmbed:
            def create_mock_embedding(num_embeddings, embedding_dim, **kwargs):
                return torch.nn.Embedding(num_embeddings, embedding_dim)

            MockTPEmbed.side_effect = create_mock_embedding

            embedding_layer = ClimLlamaEmbedding(
                tp_pg=mock_process_group,
                config=config,
                parallel_config=None,
            )

            batch_size = 2
            seq_len = 8

            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            # Different position IDs per batch (e.g., different document starts)
            position_ids = torch.stack([
                torch.arange(seq_len),
                torch.cat([torch.arange(4), torch.arange(4)]),  # Two documents
            ])

            output = embedding_layer(
                input_ids=input_ids,
                position_ids=position_ids,
            )

            # Should handle different position patterns per batch element
            assert output["input_embeds"].shape == (batch_size * seq_len, config.hidden_size)
            assert torch.equal(output["position_ids"], position_ids)


class TestClimLlamaIntegration:
    """Integration tests for ClimLlama model."""

    @pytest.fixture
    def small_config(self):
        """Create a small configuration for testing."""
        return ClimLlamaConfig(
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

    def test_config_serialization(self, small_config):
        """Test that config can be serialized and deserialized."""
        import dataclasses
        import json

        # Convert to dict
        config_dict = dataclasses.asdict(small_config)

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

        assert new_config.hidden_size == small_config.hidden_size
        assert new_config.var_vocab_size == small_config.var_vocab_size
        assert new_config.variables == small_config.variables

    def test_embedding_forward_backward(self, small_config):
        """Test forward and backward pass through embedding layer."""
        from nanotron.models.climllama import ClimLlamaEmbedding

        mock_pg = MagicMock()
        mock_pg.size.return_value = 1
        mock_pg.rank.return_value = 0

        with patch('nanotron.parallel.tensor_parallel.nn.TensorParallelEmbedding') as MockTPEmbed:
            def create_mock_embedding(num_embeddings, embedding_dim, **kwargs):
                return torch.nn.Embedding(num_embeddings, embedding_dim)

            MockTPEmbed.side_effect = create_mock_embedding

            embedding_layer = ClimLlamaEmbedding(
                tp_pg=mock_pg,
                config=small_config,
                parallel_config=None,
            )

            batch_size = 2
            seq_len = 8

            input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            var_idx = torch.randint(0, small_config.var_vocab_size, (batch_size, seq_len))
            res_idx = torch.randint(0, small_config.res_vocab_size, (batch_size, seq_len))
            leadtime_idx = torch.randint(0, small_config.leadtime_vocab_size, (batch_size, seq_len))
            spatial_temporal_features = torch.randn(batch_size, seq_len, 7, requires_grad=True)

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


class TestSpatialTemporalFeatures:
    """Tests for spatial-temporal feature handling."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
