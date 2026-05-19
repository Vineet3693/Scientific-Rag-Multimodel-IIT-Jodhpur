"""
ColPali Vision Embedder — Multi-Vector Page Embeddings.

Implements :class:`BaseEmbedder` using **ColPali-v1.2**
(``vidore/colpali-v1.2``), a vision-language model built on Gemma-2B
that produces **multi-vector** embeddings for document pages.

MaxSim Concept
--------------
Unlike traditional single-vector embeddings, ColPali outputs a
**set** of 128-dimensional vectors — one per visual patch token — for
each page image.  At retrieval time, the query is also embedded as a
set of 128-dim vectors, and similarity is computed via **MaxSim**
(maximum-similarity aggregation):

    score(query, page) = Σ_t  max_i  cos(query_t, page_i)

where ``query_t`` iterates over query token vectors and ``page_i``
over page token vectors.  This late-interaction mechanism captures
fine-grained alignment between query terms and visual regions,
outperforming single-vector approaches on scientific document
retrieval benchmarks.

Why .npy Instead of ChromaDB?
------------------------------
ChromaDB (and most vector stores) store **one** vector per document.
ColPali produces **N** vectors per page (one per patch token), which
does not fit the single-vector paradigm.  Therefore, ColPali
embeddings are saved as ``.npy`` files and MaxSim scoring is performed
at query time in the retrieval module.

VRAM: ~2.5 GB on GPU (Gemma-2B backbone with float16).

Example:
    >>> from src.embeddings.colpali_embedder import ColPaliEmbedder
    >>> embedder = ColPaliEmbedder()
    >>> embedder.load()
    >>> output = embedder.embed_image(pil_image)
    >>> print(output.vectors.shape)  # e.g. torch.Size([1030, 128])
    >>> embedder.save_vectors(output, "embeddings/page_3.npy")
    >>> embedder.unload()
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from src.embeddings.base_embedder import BaseEmbedder, EmbeddingOutput
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: VRAM logging
# ---------------------------------------------------------------------------

def _log_vram(label: str) -> None:
    """Log current GPU memory usage with a descriptive label.

    Args:
        label: A human-readable tag, e.g. ``"ColPali after load"``.
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
# ColPaliEmbedder
# ---------------------------------------------------------------------------

class ColPaliEmbedder(BaseEmbedder):
    """Multi-vector vision embedder using ColPali-v1.2.

    Each page image is resized to 448×448 and processed through the
    ColPali model to produce a tensor of shape ``(num_tokens, 128)``
    where ``num_tokens`` varies by image content (typically ~1000).

    Args:
        model_name: Hugging Face model identifier.  Defaults to
            ``"vidore/colpali-v1.2"``.
        device: Target device, e.g. ``"cuda"`` or ``"cpu"``.
        torch_dtype: Floating-point dtype for model weights.  Accepts
            ``"float16"`` or ``"float32"``.
        max_pages_per_batch: Maximum number of pages to embed in a
            single forward pass.  Larger values are faster but consume
            more VRAM.  Default 4 is safe for P100 (16 GB).

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """

    # Default image size expected by ColPali
    IMAGE_SIZE: int = 448

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = "cuda",
        torch_dtype: str = "float16",
        max_pages_per_batch: int = 4,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.max_pages_per_batch = max_pages_per_batch

        # Resolve dtype string → torch dtype
        self._torch_dtype = (
            torch.float16 if torch_dtype == "float16" else torch.float32
        )

        # Internal state — set by load()
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

        logger.info(
            "ColPaliEmbedder initialised — model=%s, device=%s, dtype=%s, "
            "max_batch=%d",
            self.model_name,
            self.device,
            self.torch_dtype_str,
            self.max_pages_per_batch,
        )

    # -----------------------------------------------------------------
    # load
    # -----------------------------------------------------------------

    def load(self) -> None:
        """Load the ColPali model and processor from colpali_engine.

        Downloads model weights from Hugging Face Hub on first use.
        The model is moved to the configured device and set to eval
        mode.  Expected VRAM usage is ~2.5 GB in float16.

        Raises:
            RuntimeError: If CUDA is not available when requested, or
                if the model cannot be loaded (OOM, download failure).
        """
        logger.info("Loading ColPali model: %s", self.model_name)

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available — falling back to CPU.  "
                "Embedding will be significantly slower."
            )
            self.device = "cpu"

        try:
            from colpali_engine.models import ColPali, ColPaliProcessor

            logger.info("Downloading / loading ColPali weights…")
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=self._torch_dtype,
                device_map=self.device,
            )
            self._model.eval()
            logger.info("ColPali model loaded and set to eval mode.")

            self._processor = ColPaliProcessor.from_pretrained(self.model_name)
            logger.info("ColPali processor loaded.")

            _log_vram("ColPali after load (expected ~2.5 GB)")

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA OOM while loading ColPali.  "
                "Ensure previous models have been unloaded."
            )
            self._model = None
            self._processor = None
            raise RuntimeError(
                "CUDA Out of Memory while loading ColPali.  "
                "Call unload() on other models first."
            ) from None

        except Exception as exc:
            logger.error("Failed to load ColPali: %s", exc)
            self._model = None
            self._processor = None
            raise RuntimeError(
                f"Failed to load ColPali model: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # embed_image
    # -----------------------------------------------------------------

    def embed_image(self, image: Any) -> EmbeddingOutput:
        """Embed a single page image into multi-vector representation.

        The image is resized to 448×448 (ColPali's expected input
        resolution) and processed through the model.  The output is
        a tensor of shape ``(num_tokens, 128)`` where each row is a
        128-dim vector for one visual patch token.

        Args:
            image: A ``PIL.Image.Image`` instance (typically a
                rendered PDF page).

        Returns:
            An :class:`EmbeddingOutput` with ``vectors`` of shape
            ``(num_tokens, 128)``.

        Raises:
            RuntimeError: If the model has not been loaded or if an
                OOM error occurs.
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Model not loaded. Call load() before embed_image()."
            )

        t_start = time.time()
        logger.debug("Embedding single image — mode=%s, size=%s", image.mode, image.size)

        try:
            # Resize to ColPali's expected input size.
            resized = image.convert("RGB").resize(
                (self.IMAGE_SIZE, self.IMAGE_SIZE)
            )

            # Process through ColPali.
            inputs = self._processor(images=[resized]).to(self._model.device)

            with torch.no_grad():
                embeddings = self._model(**inputs)

            # embeddings is a list of tensors, one per image.
            # Each tensor has shape (num_tokens, 128).
            vectors = embeddings[0].cpu()

            embedding_time = time.time() - t_start
            logger.info(
                "Embedded image — vectors shape: %s, time: %.2f s",
                tuple(vectors.shape),
                embedding_time,
            )

            return EmbeddingOutput(
                vectors=vectors,
                doc_id="unknown",
                page_num=None,
                metadata={
                    "model_name": self.model_name,
                    "embedding_type": "colpali_multi_vector",
                    "image_size": self.IMAGE_SIZE,
                },
                embedding_time=embedding_time,
            )

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during embed_image.")
            raise RuntimeError(
                "CUDA Out of Memory during image embedding.  "
                "Try reducing max_pages_per_batch."
            ) from None

        except Exception as exc:
            logger.error("Failed to embed image: %s", exc)
            raise RuntimeError(f"Image embedding failed: {exc}") from exc

    # -----------------------------------------------------------------
    # embed_text — NOT SUPPORTED
    # -----------------------------------------------------------------

    def embed_text(self, text: str) -> EmbeddingOutput:
        """Not supported — ColPali is a vision-only embedder.

        ColPali produces multi-vector embeddings from page images,
        not from raw text.  For text embeddings, use
        :class:`~src.embeddings.scincl_embedder.SciNCLEmbedder` instead.

        Args:
            text: Ignored.

        Raises:
            NotImplementedError: Always raised with a helpful message.
        """
        raise NotImplementedError(
            "ColPali does not support text embedding.  "
            "Use SciNCLEmbedder for text → dense vector embeddings."
        )

    # -----------------------------------------------------------------
    # embed_batch
    # -----------------------------------------------------------------

    def embed_batch(
        self,
        items: List[Any],
        item_type: str = "image",
    ) -> List[EmbeddingOutput]:
        """Embed a batch of page images with OOM auto-recovery.

        Processes images in mini-batches of ``max_pages_per_batch``.
        If an OOM error occurs, the batch size is automatically
        reduced (8 → 4 → 2 → 1 → CPU fallback) and the batch is
        retried.

        Args:
            items: List of ``PIL.Image.Image`` instances.
            item_type: Must be ``"image"`` (ColPali only supports
                image embedding).

        Returns:
            A list of :class:`EmbeddingOutput` objects, one per item,
            in the same order as the input.

        Raises:
            NotImplementedError: If *item_type* is not ``"image"``.
            RuntimeError: If the model has not been loaded or if
                embedding fails even at batch_size=1 on CPU.
            ValueError: If *items* is empty.
        """
        self._validate_item_type(item_type)

        if item_type != "image":
            raise NotImplementedError(
                f"ColPali only supports item_type='image', got {item_type!r}"
            )

        if not items:
            raise ValueError("items list must not be empty.")

        if not self.is_loaded():
            raise RuntimeError(
                "Model not loaded. Call load() before embed_batch()."
            )

        logger.info("Batch embedding %d images.", len(items))

        # Pre-process: resize all images.
        processed_images = []
        for img in items:
            resized = img.convert("RGB").resize(
                (self.IMAGE_SIZE, self.IMAGE_SIZE)
            )
            processed_images.append(resized)

        # OOM recovery: progressively reduce batch size.
        batch_sizes = [self.max_pages_per_batch, 4, 2, 1]
        results: List[EmbeddingOutput] = []
        idx = 0

        while idx < len(processed_images):
            remaining = len(processed_images) - idx
            success = False

            for batch_size in batch_sizes:
                actual_batch = min(batch_size, remaining)
                batch = processed_images[idx : idx + actual_batch]

                try:
                    logger.debug(
                        "Embedding mini-batch [%d:%d] (batch_size=%d)",
                        idx,
                        idx + actual_batch,
                        actual_batch,
                    )

                    t_start = time.time()
                    inputs = self._processor(images=batch).to(
                        self._model.device
                    )

                    with torch.no_grad():
                        embeddings = self._model(**inputs)

                    batch_time = time.time() - t_start

                    for i, emb in enumerate(embeddings):
                        results.append(
                            EmbeddingOutput(
                                vectors=emb.cpu(),
                                doc_id="unknown",
                                page_num=None,
                                metadata={
                                    "model_name": self.model_name,
                                    "embedding_type": "colpali_multi_vector",
                                    "image_size": self.IMAGE_SIZE,
                                    "batch_index": idx + i,
                                },
                                embedding_time=batch_time / actual_batch,
                            )
                        )

                    idx += actual_batch
                    success = True
                    break  # batch succeeded, move on

                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "OOM with batch_size=%d — reducing.",
                        actual_batch,
                    )
                    torch.cuda.empty_cache()
                    continue

            if not success:
                # Final fallback: CPU.
                logger.warning(
                    "All GPU batch sizes failed — falling back to CPU "
                    "for remaining items [%d:%d].",
                    idx,
                    idx + remaining,
                )
                cpu_results = self._embed_batch_on_cpu(
                    processed_images[idx:]
                )
                results.extend(cpu_results)
                break

        logger.info(
            "Batch embedding complete — %d outputs produced.", len(results)
        )
        return results

    # -----------------------------------------------------------------
    # CPU fallback for batch
    # -----------------------------------------------------------------

    def _embed_batch_on_cpu(self, images: List[Any]) -> List[EmbeddingOutput]:
        """Embed images on CPU as a last-resort OOM recovery.

        Moves the model to CPU temporarily, embeds the images, then
        moves back to the original device.

        Args:
            images: List of pre-processed PIL images.

        Returns:
            List of :class:`EmbeddingOutput`.

        Raises:
            RuntimeError: If CPU embedding also fails.
        """
        logger.info("Moving ColPali model to CPU for fallback embedding.")
        original_device = self._model.device
        self._model = self._model.to("cpu")

        results: List[EmbeddingOutput] = []
        try:
            for i, img in enumerate(images):
                t_start = time.time()
                inputs = self._processor(images=[img]).to("cpu")

                with torch.no_grad():
                    embeddings = self._model(**inputs)

                emb_time = time.time() - t_start
                results.append(
                    EmbeddingOutput(
                        vectors=embeddings[0].cpu(),
                        doc_id="unknown",
                        page_num=None,
                        metadata={
                            "model_name": self.model_name,
                            "embedding_type": "colpali_multi_vector",
                            "fallback": "cpu",
                            "batch_index": i,
                        },
                        embedding_time=emb_time,
                    )
                )
        except Exception as exc:
            logger.error("CPU fallback embedding failed: %s", exc)
            raise RuntimeError(
                f"CPU fallback embedding failed: {exc}"
            ) from exc
        finally:
            # Restore device.
            if "cuda" in str(original_device) and torch.cuda.is_available():
                logger.info("Restoring ColPali model to %s.", original_device)
                self._model = self._model.to(original_device)

        return results

    # -----------------------------------------------------------------
    # save_vectors
    # -----------------------------------------------------------------

    def save_vectors(
        self, output: EmbeddingOutput, filepath: Union[str, Path]
    ) -> None:
        """Save ColPali multi-vector embeddings as a ``.npy`` file.

        The vectors tensor is converted to a NumPy array and saved via
        :func:`numpy.save`.  A companion ``.meta.npy`` file is also
        written containing the metadata dictionary, so that embeddings
        can be loaded back with full provenance.

        Args:
            output: The :class:`EmbeddingOutput` to persist.
            filepath: Destination path.  The ``.npy`` extension is
                appended automatically if not present.

        Raises:
            IOError: If the file cannot be written.
            ValueError: If *output* or *filepath* is invalid.
        """
        filepath = Path(filepath)
        if filepath.suffix != ".npy":
            filepath = filepath.with_suffix(".npy")

        # Ensure parent directory exists.
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            vectors_np = output.vectors.numpy()
            np.save(str(filepath), vectors_np)
            logger.info(
                "Saved ColPali vectors to %s — shape: %s",
                filepath,
                vectors_np.shape,
            )

            # Save metadata alongside.
            meta_path = filepath.with_suffix(".meta.npy")
            np.save(str(meta_path), output.metadata)
            logger.debug("Saved metadata to %s", meta_path)

        except Exception as exc:
            logger.error("Failed to save vectors to %s: %s", filepath, exc)
            raise IOError(f"Failed to save vectors: {exc}") from exc

    # -----------------------------------------------------------------
    # unload
    # -----------------------------------------------------------------

    def unload(self) -> None:
        """Unload the ColPali model and processor, freeing GPU memory.

        Deletes model and processor references, forces garbage
        collection, and empties the CUDA cache.  This is essential
        before loading the next model in the staggered-loading
        sequence.

        Raises:
            RuntimeError: If unloading encounters an unexpected error.
        """
        logger.info("Unloading ColPali model…")

        vram_before = (
            torch.cuda.memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        try:
            del self._model
            del self._processor
        except AttributeError:
            logger.warning("Model or processor was already None during unload.")
        finally:
            self._model = None
            self._processor = None

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
            "ColPali unloaded — VRAM freed: %.2f GB "
            "(before: %.2f GB, after: %.2f GB)",
            freed,
            vram_before,
            vram_after,
        )
        _log_vram("ColPali after unload")
