"""
Tests for Embedding Models (ColPali and SciNCL).

Covers:
- EmbeddingOutput dataclass creation and validation
- ColPali embedder: embed_text raises NotImplementedError
- SciNCL embedder: embed_image raises NotImplementedError
"""

from __future__ import annotations

import pytest
import torch

from src.embeddings.base_embedder import EmbeddingOutput
from src.embeddings.colpali_embedder import ColPaliEmbedder
from src.embeddings.scincl_embedder import SciNCLEmbedder


# ===========================================================================
# TESTS
# ===========================================================================


class TestEmbeddingOutput:
    """Tests for the EmbeddingOutput dataclass."""

    def test_embedding_output_creation(self) -> None:
        """Test basic creation of an EmbeddingOutput with a valid tensor."""
        vectors = torch.randn(128)
        output = EmbeddingOutput(
            vectors=vectors,
            doc_id="2305.12345",
        )
        assert output.doc_id == "2305.12345"
        assert output.vectors.shape == (128,)
        assert output.page_num is None
        assert output.metadata == {}
        assert output.embedding_time == 0.0

    def test_embedding_output_with_all_fields(self) -> None:
        """Test creation with all fields populated."""
        vectors = torch.randn(1030, 128)  # ColPali-style multi-vector
        output = EmbeddingOutput(
            vectors=vectors,
            doc_id="2305.12345",
            page_num=3,
            metadata={"model_name": "vidore/colpali-v1.2", "type": "colpali"},
            embedding_time=1.52,
        )
        assert output.page_num == 3
        assert output.vectors.shape == (1030, 128)
        assert output.metadata["model_name"] == "vidore/colpali-v1.2"
        assert output.embedding_time == 1.52

    def test_embedding_output_scincl_style(self) -> None:
        """Test creation with SciNCL-style 768-dim dense vector."""
        vectors = torch.randn(768)
        output = EmbeddingOutput(
            vectors=vectors,
            doc_id="2310.12345",
            page_num=1,
            metadata={"model_name": "malteos/scincl", "type": "scincl_dense"},
            embedding_time=0.35,
        )
        assert output.vectors.shape == (768,)
        assert output.metadata["type"] == "scincl_dense"

    def test_embedding_output_empty_doc_id_raises(self) -> None:
        """Test that empty doc_id raises ValueError."""
        vectors = torch.randn(128)
        with pytest.raises(ValueError, match="doc_id must not be empty"):
            EmbeddingOutput(vectors=vectors, doc_id="")

    def test_embedding_output_negative_embedding_time_raises(self) -> None:
        """Test that negative embedding_time raises ValueError."""
        vectors = torch.randn(128)
        with pytest.raises(ValueError, match="embedding_time must be >= 0"):
            EmbeddingOutput(vectors=vectors, doc_id="test", embedding_time=-0.5)

    def test_embedding_output_non_tensor_vectors_raises(self) -> None:
        """Test that non-tensor vectors raises ValueError."""
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            EmbeddingOutput(vectors=[1.0, 2.0, 3.0], doc_id="test")

    def test_embedding_output_numpy_array_vectors_raises(self) -> None:
        """Test that numpy array vectors raises ValueError (must be torch.Tensor)."""
        import numpy as np

        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            EmbeddingOutput(vectors=np.array([1.0, 2.0]), doc_id="test")

    def test_embedding_output_zero_embedding_time_ok(self) -> None:
        """Test that zero embedding_time is valid."""
        vectors = torch.randn(128)
        output = EmbeddingOutput(vectors=vectors, doc_id="test", embedding_time=0.0)
        assert output.embedding_time == 0.0


class TestColPaliEmbedder:
    """Tests for the ColPaliEmbedder interface."""

    def test_colpali_embedder_not_implemented_text(self) -> None:
        """Test that ColPali.embed_text raises NotImplementedError."""
        embedder = ColPaliEmbedder()
        with pytest.raises(NotImplementedError, match="ColPali does not support text"):
            embedder.embed_text("Some query text")

    def test_colpali_embedder_initialization(self) -> None:
        """Test that ColPaliEmbedder initializes with expected defaults."""
        embedder = ColPaliEmbedder()
        assert embedder.model_name == "vidore/colpali-v1.2"
        assert embedder.IMAGE_SIZE == 448
        assert embedder.max_pages_per_batch == 4

    def test_colpali_embedder_custom_params(self) -> None:
        """Test ColPaliEmbedder with custom parameters."""
        embedder = ColPaliEmbedder(
            model_name="custom/colpali",
            device="cpu",
            torch_dtype="float32",
            max_pages_per_batch=8,
        )
        assert embedder.model_name == "custom/colpali"
        assert embedder.device == "cpu"
        assert embedder.max_pages_per_batch == 8

    def test_colpali_embedder_not_loaded_by_default(self) -> None:
        """Test that ColPaliEmbedder is not loaded after initialization."""
        embedder = ColPaliEmbedder()
        assert not embedder.is_loaded()


class TestSciNCLEmbedder:
    """Tests for the SciNCLEmbedder interface."""

    def test_scincl_embedder_not_implemented_image(self) -> None:
        """Test that SciNCL.embed_image raises NotImplementedError."""
        embedder = SciNCLEmbedder()
        with pytest.raises(NotImplementedError, match="SciNCL does not support image"):
            embedder.embed_image(None)

    def test_scincl_embedder_initialization(self) -> None:
        """Test that SciNCLEmbedder initializes with expected defaults."""
        embedder = SciNCLEmbedder()
        assert embedder.model_name == "malteos/scincl"
        assert embedder.OUTPUT_DIM == 768
        assert embedder.max_length == 512

    def test_scincl_embedder_custom_params(self) -> None:
        """Test SciNCLEmbedder with custom parameters."""
        embedder = SciNCLEmbedder(
            model_name="custom/scincl",
            device="cpu",
            max_length=256,
        )
        assert embedder.model_name == "custom/scincl"
        assert embedder.max_length == 256

    def test_scincl_embedder_not_loaded_by_default(self) -> None:
        """Test that SciNCLEmbedder is not loaded after initialization."""
        embedder = SciNCLEmbedder()
        assert not embedder.is_loaded()

    def test_scincl_mean_pool(self) -> None:
        """Test the static _mean_pool helper method."""
        # Create dummy token embeddings and attention mask
        token_embeddings = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],  # batch item 1 (3 tokens, last is padding)
            [[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]],  # batch item 2 (1 real token)
        ])
        attention_mask = torch.tensor([
            [1, 1, 0],
            [1, 0, 0],
        ])

        pooled = SciNCLEmbedder._mean_pool(token_embeddings, attention_mask)

        assert pooled.shape == (2, 2)
        # Item 1: mean of [1,2] and [3,4] = [2, 3]
        assert torch.allclose(pooled[0], torch.tensor([2.0, 3.0]))
        # Item 2: just [5, 6]
        assert torch.allclose(pooled[1], torch.tensor([5.0, 6.0]))
