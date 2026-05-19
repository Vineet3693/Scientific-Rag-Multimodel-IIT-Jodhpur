"""
ColPali Retriever — Multi-Vector Page Retrieval via MaxSim Scoring.

Implements :class:`BaseRetriever` for the ColPali vision-embedding
backend.  ColPali produces **multi-vector** embeddings (one 128-dim
vector per visual patch token) for each document page, stored as
individual ``.npy`` files.  At query time, the query is also embedded
as a set of 128-dim vectors, and similarity is computed via the
**MaxSim** (maximum-similarity aggregation) algorithm.

MaxSim Algorithm
----------------
Given a query with *Q* token vectors and a page with *P* token vectors,
both of dimension 128:

1. Compute the dot-product similarity matrix:
   ``S = query @ page.T``  →  shape ``(Q, P)``

2. For each query token, find the maximum similarity across all
   page tokens: ``max_per_query = S.max(dim=1).values``  →  shape ``(Q,)``

3. Sum these maxima to get the final score:
   ``score = max_per_query.sum()``

This late-interaction mechanism captures fine-grained alignment between
query terms and visual regions, outperforming single-vector approaches
on scientific document retrieval benchmarks.

Why .npy Instead of ChromaDB?
------------------------------
ChromaDB (and most vector stores) store **one** vector per document.
ColPali produces **N** vectors per page (one per patch token), which
does not fit the single-vector paradigm.  Therefore, ColPali
embeddings are saved as ``.npy`` files and MaxSim scoring is performed
at query time in this module.

Example:
    >>> import torch
    >>> from src.retrieval.colpali_retriever import ColPaliRetriever
    >>> retriever = ColPaliRetriever(npy_dir="data/indices/multivectors/")
    >>> retriever.load_index("data/indices/multivectors/")
    >>> query_emb = torch.randn(30, 128)  # 30 query tokens, 128 dims
    >>> results = retriever.retrieve(query_emb, top_k=5)
    >>> for doc in results:
    ...     print(doc.doc_id, doc.page_num, f"{doc.score:.4f}")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import PIL.Image
import torch

from src.retrieval.base_retriever import BaseRetriever, RetrievedDocument, SourceCitation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ColPaliRetriever
# ---------------------------------------------------------------------------

class ColPaliRetriever(BaseRetriever):
    """Multi-vector page retriever using ColPali embeddings and MaxSim.

    Loads pre-computed ColPali page embeddings from a directory of
    ``.npy`` files and retrieves the most relevant pages for a given
    query using the MaxSim scoring algorithm.

    The expected ``.npy`` filename convention is::

        <doc_id>_page_<N>.npy

    For example, ``2305.12345_page_3.npy`` contains the multi-vector
    embedding for page 3 of arXiv paper 2305.12345.  Each ``.npy``
    file stores a NumPy array of shape ``(num_tokens, 128)``.

    Args:
        npy_dir: Path to the directory containing ``.npy`` embedding
            files.  Used as the default index path.
        top_k: Default number of results to return.  Can be overridden
            per query.

    Raises:
        FileNotFoundError: If *npy_dir* does not exist at load time.
    """

    # Regex to parse doc_id and page number from filenames like
    # "2305.12345_page_3.npy" or "my_paper_page_12.npy"
    _FILENAME_PATTERN = re.compile(r"^(.+)_page_(\d+)\.npy$")

    def __init__(self, npy_dir: str, top_k: int = 5) -> None:
        self.npy_dir = npy_dir
        self.top_k = top_k

        # Internal state — populated by load_index()
        self._index: Dict[str, np.ndarray] = {}
        self._loaded: bool = False

        logger.info(
            "ColPaliRetriever initialised — npy_dir=%s, top_k=%d",
            self.npy_dir,
            self.top_k,
        )

    # -----------------------------------------------------------------
    # load_index
    # -----------------------------------------------------------------

    def load_index(self, index_path: str) -> None:
        """Load all ``.npy`` embedding files from a directory.

        Scans *index_path* for files matching the naming convention
        ``<doc_id>_page_<N>.npy`` and loads each into memory as a
        NumPy array of shape ``(num_tokens, 128)``.  The key in the
        internal dictionary is ``"<doc_id>_page_<N>"`` (without the
        ``.npy`` extension).

        Args:
            index_path: Path to the directory containing ``.npy``
                embedding files.

        Raises:
            FileNotFoundError: If *index_path* does not exist or is
                not a directory.
            RuntimeError: If no valid ``.npy`` files are found.
        """
        dir_path = Path(index_path)

        if not dir_path.exists():
            raise FileNotFoundError(
                f"Index directory not found: {dir_path}"
            )

        if not dir_path.is_dir():
            raise FileNotFoundError(
                f"Index path is not a directory: {dir_path}"
            )

        logger.info("Loading ColPali index from: %s", dir_path)

        self._index.clear()
        loaded_count = 0
        skipped_count = 0

        for npy_file in sorted(dir_path.glob("*.npy")):
            # Skip metadata files (e.g. .meta.npy saved by ColPaliEmbedder)
            if ".meta.npy" in npy_file.name:
                logger.debug("Skipping metadata file: %s", npy_file.name)
                skipped_count += 1
                continue

            # Parse filename to extract key.
            match = self._FILENAME_PATTERN.match(npy_file.name)
            if match:
                key = f"{match.group(1)}_page_{match.group(2)}"
            else:
                # Fallback: use stem as key (without .npy)
                key = npy_file.stem
                logger.warning(
                    "Filename %s does not match expected pattern "
                    "'<doc_id>_page_<N>.npy' — using stem as key: %s",
                    npy_file.name,
                    key,
                )

            try:
                vectors = np.load(str(npy_file))
                self._index[key] = vectors
                loaded_count += 1
                logger.debug(
                    "Loaded %s — shape: %s", npy_file.name, vectors.shape
                )
            except Exception as exc:
                logger.error(
                    "Failed to load %s: %s — skipping.", npy_file.name, exc
                )
                skipped_count += 1

        if loaded_count == 0:
            raise RuntimeError(
                f"No valid .npy files found in {dir_path}.  "
                f"Skipped {skipped_count} files."
            )

        self._loaded = True
        logger.info(
            "ColPali index loaded — %d pages indexed, %d files skipped.",
            loaded_count,
            skipped_count,
        )

    # -----------------------------------------------------------------
    # retrieve
    # -----------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        """Retrieve the top-k pages using MaxSim scoring.

        For each indexed page, computes the MaxSim score against the
        query embedding and returns the top-k results sorted by
        descending score.

        Args:
            query_embedding: Query multi-vector tensor of shape
                ``(num_query_tokens, 128)``.  This is the output of
                the ColPali query embedder.
            top_k: Number of results to return.  Defaults to the value
                set at initialisation.

        Returns:
            A list of :class:`RetrievedDocument` objects sorted by
            descending MaxSim score.

        Raises:
            RuntimeError: If the index has not been loaded via
                :meth:`load_index`.
            ValueError: If *top_k* is not a positive integer.
        """
        if not self._loaded:
            raise RuntimeError(
                "Index not loaded. Call load_index() before retrieve()."
            )

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        logger.info(
            "ColPali retrieval — query shape: %s, top_k: %d, "
            "index size: %d pages",
            tuple(query_embedding.shape),
            top_k,
            len(self._index),
        )

        # Compute MaxSim score for every indexed page.
        scored_pages: List[tuple] = []  # (key, score)

        for key, page_vectors_np in self._index.items():
            # Convert numpy array to torch tensor for matrix operations.
            page_vectors = torch.from_numpy(page_vectors_np).float()

            score = self.compute_maxsim(query_embedding, page_vectors)
            scored_pages.append((key, float(score)))

        # Sort by descending score.
        scored_pages.sort(key=lambda x: x[1], reverse=True)

        # Take top_k.
        top_results = scored_pages[:top_k]

        # Build RetrievedDocument objects.
        results: List[RetrievedDocument] = []
        for key, score in top_results:
            doc_id, page_num = self._parse_key(key)

            citation = SourceCitation(
                paper_title=doc_id,  # Will be enriched later by metadata
                paper_id=doc_id,
                arxiv_url=f"https://arxiv.org/abs/{doc_id}",
                page_numbers=[page_num],
                relevance_score=score,
            )

            results.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    page_num=page_num,
                    score=score,
                    image=None,  # Image loading is deferred to context builder
                    text=None,   # Text extraction is deferred to context builder
                    source_citation=citation,
                    retrieval_method="colpali",
                )
            )

        logger.info(
            "ColPali retrieval complete — returned %d results "
            "(scores: %.4f … %.4f).",
            len(results),
            results[0].score if results else 0.0,
            results[-1].score if results else 0.0,
        )

        return results

    # -----------------------------------------------------------------
    # compute_maxsim
    # -----------------------------------------------------------------

    @staticmethod
    def compute_maxsim(
        query_vectors: torch.Tensor,
        page_vectors: torch.Tensor,
    ) -> float:
        """Compute the MaxSim score between query and page embeddings.

        The MaxSim algorithm is the core of ColPali's late-interaction
        retrieval mechanism.  For each query token vector, it finds the
        most similar page token vector (by dot product), then sums
        these maxima to produce a single relevance score.

        Algorithm:
            1. Compute similarity matrix:
               ``S = query_vectors @ page_vectors.T``
               Shape: ``(Q, P)`` where Q = num query tokens,
               P = num page tokens.

            2. For each query token, take the maximum similarity across
               all page tokens:
               ``max_per_query = S.max(dim=1).values``
               Shape: ``(Q,)``

            3. Sum the maxima:
               ``score = max_per_query.sum()``

        This approach is superior to single-vector retrieval because
        it preserves fine-grained alignment: each query term can
        independently "find" its best match among the page's visual
        patches, capturing layout, figures, and table structures that
        a single pooled vector would lose.

        Example:
            >>> query = torch.randn(30, 128)   # 30 query tokens
            >>> page  = torch.randn(1030, 128)  # 1030 page tokens
            >>> score = ColPaliRetriever.compute_maxsim(query, page)
            >>> print(f"MaxSim score: {score:.4f}")

        Args:
            query_vectors: Tensor of shape ``(Q, 128)`` where Q is the
                number of query token vectors.
            page_vectors: Tensor of shape ``(P, 128)`` where P is the
                number of page token vectors.

        Returns:
            The MaxSim score as a float.  Higher values indicate
            greater relevance.

        Raises:
            ValueError: If either input is not a 2-D tensor or if the
                embedding dimensions do not match.
        """
        if query_vectors.dim() != 2:
            raise ValueError(
                f"query_vectors must be 2-D, got {query_vectors.dim()}-D "
                f"with shape {tuple(query_vectors.shape)}"
            )
        if page_vectors.dim() != 2:
            raise ValueError(
                f"page_vectors must be 2-D, got {page_vectors.dim()}-D "
                f"with shape {tuple(page_vectors.shape)}"
            )
        if query_vectors.shape[1] != page_vectors.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: "
                f"query has {query_vectors.shape[1]}, "
                f"page has {page_vectors.shape[1]}"
            )

        # Step 1: Dot-product similarity matrix [Q, P]
        similarity_matrix = query_vectors @ page_vectors.T

        # Step 2: Max similarity per query token [Q]
        max_per_query = similarity_matrix.max(dim=1).values

        # Step 3: Sum the maxima → final score
        score = max_per_query.sum()

        return float(score.item())

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_key(key: str) -> tuple:
        """Parse an index key into (doc_id, page_num).

        The key format is ``"<doc_id>_page_<N>"``.  For example,
        ``"2305.12345_page_3"`` → ``("2305.12345", 3)``.

        Args:
            key: The index key string.

        Returns:
            A tuple ``(doc_id: str, page_num: int)``.
        """
        # Split from the right on "_page_" to handle doc_ids that may
        # contain underscores.
        parts = key.rsplit("_page_", 1)
        if len(parts) == 2:
            return parts[0], int(parts[1])
        # Fallback: treat the entire key as doc_id with page 1.
        logger.warning(
            "Could not parse key '%s' — defaulting to page 1.", key
        )
        return key, 1
