"""
SciNCL Text Embedder — Dense Scientific Text Embeddings.

Implements :class:`BaseEmbedder` using **SciNCL**
(``malteos/scincl``), a SciBERT-based model fine-tuned with
nearest-neighbor contrastive learning for scientific text.  It
produces a single **768-dimensional** dense vector per input text,
suitable for semantic similarity search in ChromaDB.

Key Design Decisions
--------------------
* **Mean pooling** over the last hidden state is used instead of the
  ``[CLS]`` token, as it generally yields better sentence-level
  representations.
* **L2 normalisation** ensures that dot-product and cosine similarity
  are equivalent, simplifying downstream retrieval.
* ChromaDB is used for persistence because SciNCL produces a **single
  dense vector** per text chunk — unlike ColPali's multi-vector output
  — which is exactly what ChromaDB's ANN index is designed for.

VRAM: ~0.6 GB on GPU (SciBERT-base with float16).

Example:
    >>> from src.embeddings.scincl_embedder import SciNCLEmbedder
    >>> embedder = SciNCLEmbedder()
    >>> embedder.load()
    >>> output = embedder.embed_text("Transformers achieve state-of-the-art…")
    >>> print(output.vectors.shape)  # torch.Size([768])
    >>> embedder.save_to_chromadb(output, "paper_chunks", "./chroma_db")
    >>> embedder.unload()
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from src.embeddings.base_embedder import BaseEmbedder, EmbeddingOutput
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: VRAM logging
# ---------------------------------------------------------------------------

def _log_vram(label: str) -> None:
    """Log current GPU memory usage with a descriptive label.

    Args:
        label: A human-readable tag, e.g. ``"SciNCL after load"``.
    """
    if not torch.cuda.is_available():
        logger.debug("%s | CUDA not available", label)
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    logger.info(
        "%s | VRAM — allocated: %.2f GB, reserved: %.2f GB",
        label,
        allocated,
        reserved,
    )


# ---------------------------------------------------------------------------
# SciNCLEmbedder
# ---------------------------------------------------------------------------

class SciNCLEmbedder(BaseEmbedder):
    """Dense text embedder using the SciNCL model.

    Produces a single 768-dimensional L2-normalised vector per text
    input, suitable for ChromaDB-based ANN retrieval.

    Args:
        model_name: Hugging Face model identifier.  Defaults to
            ``"malteos/scincl"``.
        device: Target device, e.g. ``"cuda"`` or ``"cpu"``.
        max_length: Maximum token length for the tokenizer.  Texts
            longer than this are truncated.

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """

    OUTPUT_DIM: int = 768

    def __init__(
        self,
        model_name: str = "malteos/scincl",
        device: str = "cuda",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        # Internal state — set by load()
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

        logger.info(
            "SciNCLEmbedder initialised — model=%s, device=%s, "
            "max_length=%d",
            self.model_name,
            self.device,
            self.max_length,
        )

    # -----------------------------------------------------------------
    # load
    # -----------------------------------------------------------------

    def load(self) -> None:
        """Load the SciNCL model and tokenizer from Hugging Face.

        Downloads model weights on first use.  The model is moved to
        the configured device and set to eval mode.  Expected VRAM
        usage is ~0.6 GB in float16.

        Raises:
            RuntimeError: If CUDA is not available when requested, or
                if the model cannot be loaded (OOM, download failure).
        """
        logger.info("Loading SciNCL model: %s", self.model_name)

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available — falling back to CPU."
            )
            self.device = "cpu"

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info("Downloading / loading SciNCL weights…")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("SciNCL model loaded and set to eval mode.")

            _log_vram("SciNCL after load (expected ~0.6 GB)")

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA OOM while loading SciNCL.  "
                "Ensure previous models have been unloaded."
            )
            self._model = None
            self._tokenizer = None
            raise RuntimeError(
                "CUDA Out of Memory while loading SciNCL.  "
                "Call unload() on other models first."
            ) from None

        except Exception as exc:
            logger.error("Failed to load SciNCL: %s", exc)
            self._model = None
            self._tokenizer = None
            raise RuntimeError(
                f"Failed to load SciNCL model: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # embed_text
    # -----------------------------------------------------------------

    def embed_text(self, text: str) -> EmbeddingOutput:
        """Embed a single text string into a 768-dim dense vector.

        The text is tokenised (with truncation to ``max_length``),
        passed through the SciNCL encoder, and the last hidden state
        is **mean-pooled** and **L2-normalised** to produce a single
        768-dim vector.

        Args:
            text: The input text to embed.  Must not be empty.

        Returns:
            An :class:`EmbeddingOutput` with ``vectors`` of shape
            ``(768,)``.

        Raises:
            ValueError: If *text* is empty.
            RuntimeError: If the model has not been loaded or if
                embedding fails.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")

        if not self.is_loaded():
            raise RuntimeError(
                "Model not loaded. Call load() before embed_text()."
            )

        t_start = time.time()
        logger.debug("Embedding text — length=%d", len(text))

        try:
            # Tokenise.
            encoded = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Forward pass.
            with torch.no_grad():
                outputs = self._model(**encoded)

            # Mean pooling over token dimension.
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            pooled = self._mean_pool(token_embeddings, attention_mask)

            # L2 normalise.
            normalized = F.normalize(pooled, p=2, dim=-1)

            # Squeeze batch dimension → (768,).
            vector = normalized.squeeze(0).cpu()

            embedding_time = time.time() - t_start
            logger.info(
                "Embedded text — vector shape: %s, time: %.2f s",
                tuple(vector.shape),
                embedding_time,
            )

            return EmbeddingOutput(
                vectors=vector,
                doc_id="unknown",
                page_num=None,
                metadata={
                    "model_name": self.model_name,
                    "embedding_type": "scincl_dense",
                    "max_length": self.max_length,
                    "pooling": "mean",
                    "normalization": "l2",
                },
                embedding_time=embedding_time,
            )

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during embed_text.")
            raise RuntimeError(
                "CUDA Out of Memory during text embedding."
            ) from None

        except Exception as exc:
            logger.error("Failed to embed text: %s", exc)
            raise RuntimeError(f"Text embedding failed: {exc}") from exc

    # -----------------------------------------------------------------
    # embed_image — NOT SUPPORTED
    # -----------------------------------------------------------------

    def embed_image(self, image: Any) -> EmbeddingOutput:
        """Not supported — SciNCL is a text-only embedder.

        SciNCL produces dense text embeddings, not image embeddings.
        For vision embeddings, use
        :class:`~src.embeddings.colpali_embedder.ColPaliEmbedder`.

        Args:
            image: Ignored.

        Raises:
            NotImplementedError: Always raised with a helpful message.
        """
        raise NotImplementedError(
            "SciNCL does not support image embedding.  "
            "Use ColPaliEmbedder for image → multi-vector embeddings."
        )

    # -----------------------------------------------------------------
    # embed_batch
    # -----------------------------------------------------------------

    def embed_batch(
        self,
        items: List[Any],
        item_type: str = "text",
    ) -> List[EmbeddingOutput]:
        """Embed a batch of text strings.

        Texts are tokenised and encoded in a single forward pass for
        efficiency.  If the batch is too large for GPU memory, it is
        automatically split into smaller mini-batches.

        Args:
            items: List of text strings to embed.
            item_type: Must be ``"text"`` (SciNCL only supports text).

        Returns:
            A list of :class:`EmbeddingOutput` objects, one per text,
            in the same order as the input.

        Raises:
            NotImplementedError: If *item_type* is not ``"text"``.
            ValueError: If *items* is empty or contains non-string
                elements.
            RuntimeError: If the model has not been loaded.
        """
        self._validate_item_type(item_type)

        if item_type != "text":
            raise NotImplementedError(
                f"SciNCL only supports item_type='text', got {item_type!r}"
            )

        if not items:
            raise ValueError("items list must not be empty.")

        if not self.is_loaded():
            raise RuntimeError(
                "Model not loaded. Call load() before embed_batch()."
            )

        # Validate all items are strings.
        for i, item in enumerate(items):
            if not isinstance(item, str):
                raise ValueError(
                    f"All items must be strings; item {i} is "
                    f"{type(item).__name__}"
                )

        logger.info("Batch embedding %d texts.", len(items))

        # Process in mini-batches to avoid OOM.
        mini_batch_size = 16
        results: List[EmbeddingOutput] = []

        for start in range(0, len(items), mini_batch_size):
            batch_texts = items[start : start + mini_batch_size]

            try:
                batch_results = self._embed_text_batch(batch_texts)
                results.extend(batch_results)
            except torch.cuda.OutOfMemoryError:
                # Reduce batch size and retry.
                logger.warning(
                    "OOM with mini_batch_size=%d — retrying one by one.",
                    mini_batch_size,
                )
                torch.cuda.empty_cache()
                for text in batch_texts:
                    results.append(self.embed_text(text))

        logger.info(
            "Batch embedding complete — %d outputs produced.", len(results)
        )
        return results

    def _embed_text_batch(self, texts: List[str]) -> List[EmbeddingOutput]:
        """Embed a small batch of texts in a single forward pass.

        Args:
            texts: List of text strings (typically ≤ 16).

        Returns:
            List of :class:`EmbeddingOutput`.
        """
        t_start = time.time()

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)

        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        pooled = self._mean_pool(token_embeddings, attention_mask)
        normalized = F.normalize(pooled, p=2, dim=-1)
        vectors = normalized.cpu()

        batch_time = time.time() - t_start

        results: List[EmbeddingOutput] = []
        for i in range(len(texts)):
            results.append(
                EmbeddingOutput(
                    vectors=vectors[i],
                    doc_id="unknown",
                    page_num=None,
                    metadata={
                        "model_name": self.model_name,
                        "embedding_type": "scincl_dense",
                        "pooling": "mean",
                        "normalization": "l2",
                        "batch_index": i,
                    },
                    embedding_time=batch_time / len(texts),
                )
            )

        return results

    # -----------------------------------------------------------------
    # Mean pooling helper
    # -----------------------------------------------------------------

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mean pooling weighted by the attention mask.

        Computes the average of token embeddings, ignoring padding
        tokens (those with attention_mask == 0).

        Args:
            token_embeddings: Tensor of shape ``(batch, seq_len, dim)``.
            attention_mask: Tensor of shape ``(batch, seq_len)``.

        Returns:
            Pooled tensor of shape ``(batch, dim)``.
        """
        # Expand mask to match embedding dimensions.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        # Weighted sum and average.
        sum_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, dim=1
        )
        sum_mask = torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )
        return sum_embeddings / sum_mask

    # -----------------------------------------------------------------
    # save_vectors (basic .npy)
    # -----------------------------------------------------------------

    def save_vectors(
        self, output: EmbeddingOutput, filepath: Union[str, Path]
    ) -> None:
        """Save a single SciNCL embedding as a ``.npy`` file.

        Note:
            For production use, prefer :meth:`save_to_chromadb` which
            stores embeddings in a ChromaDB collection with full ANN
            search support.

        Args:
            output: The :class:`EmbeddingOutput` to persist.
            filepath: Destination path (``.npy`` extension added if
                missing).

        Raises:
            IOError: If the file cannot be written.
        """
        import numpy as np

        filepath = Path(filepath)
        if filepath.suffix != ".npy":
            filepath = filepath.with_suffix(".npy")

        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            vector_np = output.vectors.numpy()
            np.save(str(filepath), vector_np)
            logger.info(
                "Saved SciNCL vector to %s — shape: %s",
                filepath,
                vector_np.shape,
            )
        except Exception as exc:
            logger.error("Failed to save vector to %s: %s", filepath, exc)
            raise IOError(f"Failed to save vector: {exc}") from exc

    # -----------------------------------------------------------------
    # save_to_chromadb
    # -----------------------------------------------------------------

    def save_to_chromadb(
        self,
        output: EmbeddingOutput,
        collection_name: str,
        persist_dir: str,
    ) -> None:
        """Store a SciNCL embedding in a ChromaDB collection.

        ChromaDB is the recommended persistence backend for SciNCL
        because each text chunk maps to a **single** 768-dim vector,
        which fits ChromaDB's ANN index perfectly (unlike ColPali's
        multi-vector output).

        Args:
            output: The :class:`EmbeddingOutput` to store.
            collection_name: Name of the ChromaDB collection, e.g.
                ``"paper_chunks"``.
            persist_dir: Directory where ChromaDB data is persisted,
                e.g. ``"./chroma_db"``.

        Raises:
            ImportError: If ``chromadb`` is not installed.
            RuntimeError: If the ChromaDB write fails.
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for save_to_chromadb().  "
                "Install it with: pip install chromadb"
            )

        logger.info(
            "Saving embedding to ChromaDB — collection=%s, dir=%s",
            collection_name,
            persist_dir,
        )

        try:
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Generate a unique ID from doc_id and page_num.
            doc_id = output.doc_id or "unknown"
            page_part = f"_p{output.page_num}" if output.page_num else ""
            unique_id = f"{doc_id}{page_part}"

            vector_list = output.vectors.tolist()

            collection.upsert(
                ids=[unique_id],
                embeddings=[vector_list],
                metadatas=[{
                    "doc_id": doc_id,
                    "page_num": output.page_num,
                    "model_name": self.model_name,
                    **output.metadata,
                }],
                documents=[output.metadata.get("text", "")],
            )

            logger.info(
                "Upserted embedding id=%s into collection '%s'.",
                unique_id,
                collection_name,
            )

        except Exception as exc:
            logger.error(
                "Failed to save to ChromaDB collection '%s': %s",
                collection_name,
                exc,
            )
            raise RuntimeError(
                f"ChromaDB save failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # unload
    # -----------------------------------------------------------------

    def unload(self) -> None:
        """Unload the SciNCL model and tokenizer, freeing GPU memory.

        Deletes model and tokenizer references, forces garbage
        collection, and empties the CUDA cache.

        Raises:
            RuntimeError: If unloading encounters an unexpected error.
        """
        logger.info("Unloading SciNCL model…")

        vram_before = (
            torch.cuda.memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        try:
            del self._model
            del self._tokenizer
        except AttributeError:
            logger.warning("Model or tokenizer was already None during unload.")
        finally:
            self._model = None
            self._tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vram_after = (
            torch.cuda.memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        freed = vram_before - vram_after
        logger.info(
            "SciNCL unloaded — VRAM freed: %.2f GB "
            "(before: %.2f GB, after: %.2f GB)",
            freed,
            vram_before,
            vram_after,
        )
        _log_vram("SciNCL after unload")
