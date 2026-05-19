"""
Text Retriever — Dense Text Retrieval via ChromaDB.

Implements :class:`BaseRetriever` for the SciNCL text-embedding
backend.  SciNCL produces a single **768-dimensional** dense vector per
text chunk, stored in a ChromaDB collection for approximate nearest
neighbour (ANN) search.

At query time, the query is embedded as a 768-dim vector and ChromaDB's
HNSW index is used to retrieve the top-k most similar text chunks
using cosine similarity.  Scores are then min-max normalised to the
``[0, 1]`` range so they can be fused with ColPali scores in
:class:`~src.retrieval.fusion_retriever.FusionRetriever`.

Why ChromaDB?
-------------
ChromaDB provides efficient ANN search over single-vector embeddings,
which is exactly what SciNCL produces.  Unlike ColPali's multi-vector
output (one vector per patch token), SciNCL outputs a single 768-dim
vector per text chunk — a perfect fit for ChromaDB's HNSW index with
cosine distance.

Example:
    >>> from src.retrieval.text_retriever import TextRetriever
    >>> retriever = TextRetriever(chroma_dir="data/indices/chroma_index/")
    >>> retriever.load_index("data/indices/chroma_index/")
    >>> query_vec = [0.1, -0.2, ...]  # 768-dim list
    >>> results = retriever.retrieve(query_vec, top_k=5)
    >>> for doc in results:
    ...     print(doc.doc_id, doc.page_num, f"{doc.score:.4f}")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import PIL.Image

from src.retrieval.base_retriever import BaseRetriever, RetrievedDocument, SourceCitation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TextRetriever
# ---------------------------------------------------------------------------

class TextRetriever(BaseRetriever):
    """Dense text retriever using ChromaDB and SciNCL embeddings.

    Loads a ChromaDB persistent collection and retrieves the most
    relevant text chunks for a given query embedding using cosine
    similarity.  Results are min-max normalised to ``[0, 1]`` before
    being returned.

    Args:
        chroma_dir: Path to the ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection to query.
            Defaults to ``"sci_text"``.
        top_k: Default number of results to return.

    Raises:
        ImportError: If ``chromadb`` is not installed.
        FileNotFoundError: If *chroma_dir* does not exist at load time.
    """

    def __init__(
        self,
        chroma_dir: str,
        collection_name: str = "sci_text",
        top_k: int = 5,
    ) -> None:
        # Verify chromadb is available at init time so the user gets
        # an early error rather than at query time.
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ImportError(
                "chromadb is required for TextRetriever.  "
                "Install it with: pip install chromadb"
            )

        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.top_k = top_k

        # Internal state — populated by load_index()
        self._client: Optional[object] = None
        self._collection: Optional[object] = None
        self._loaded: bool = False

        logger.info(
            "TextRetriever initialised — chroma_dir=%s, "
            "collection=%s, top_k=%d",
            self.chroma_dir,
            self.collection_name,
            self.top_k,
        )

    # -----------------------------------------------------------------
    # load_index
    # -----------------------------------------------------------------

    def load_index(self, index_path: str) -> None:
        """Load a ChromaDB persistent client and collection.

        Connects to the ChromaDB database at *index_path* and
        retrieves the collection named ``self.collection_name``.
        The collection must already exist (created during the
        embedding/indexing pipeline stage).

        Args:
            index_path: Path to the ChromaDB persistence directory.

        Raises:
            FileNotFoundError: If *index_path* does not exist.
            RuntimeError: If the collection does not exist or cannot
                be loaded.
        """
        import chromadb

        dir_path = Path(index_path)

        if not dir_path.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {dir_path}"
            )

        logger.info(
            "Loading ChromaDB index from: %s, collection: %s",
            dir_path,
            self.collection_name,
        )

        try:
            self._client = chromadb.PersistentClient(path=str(dir_path))

            # List available collections for debugging.
            collections = self._client.list_collections()
            collection_names = [c.name for c in collections]
            logger.debug(
                "Available ChromaDB collections: %s", collection_names
            )

            self._collection = self._client.get_collection(
                name=self.collection_name,
            )

            count = self._collection.count()
            self._loaded = True

            logger.info(
                "ChromaDB collection '%s' loaded — %d documents indexed.",
                self.collection_name,
                count,
            )

        except Exception as exc:
            self._client = None
            self._collection = None
            logger.error(
                "Failed to load ChromaDB collection '%s': %s",
                self.collection_name,
                exc,
            )
            raise RuntimeError(
                f"Failed to load ChromaDB collection "
                f"'{self.collection_name}': {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # retrieve
    # -----------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: object,
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        """Retrieve the top-k text chunks using ChromaDB ANN search.

        Queries the ChromaDB collection with the provided query
        embedding using cosine similarity.  The raw distance scores
        from ChromaDB are then min-max normalised to ``[0, 1]``.

        Args:
            query_embedding: The query embedding as a list of floats
                (length 768 for SciNCL).  Can also be a 1-D NumPy
                array or torch tensor.
            top_k: Number of results to return.

        Returns:
            A list of :class:`RetrievedDocument` objects sorted by
            descending score (after normalisation).  Each result
            contains the text chunk and metadata stored in ChromaDB.

        Raises:
            RuntimeError: If the index has not been loaded.
            ValueError: If *top_k* is not a positive integer.
        """
        if not self._loaded or self._collection is None:
            raise RuntimeError(
                "Index not loaded. Call load_index() before retrieve()."
            )

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Convert query embedding to a list for ChromaDB's API.
        query_list = self._to_list(query_embedding)

        logger.info(
            "Text retrieval — query dim: %d, top_k: %d",
            len(query_list),
            top_k,
        )

        try:
            # Query ChromaDB with cosine distance.
            results = self._collection.query(
                query_embeddings=[query_list],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            raise RuntimeError(
                f"ChromaDB query failed: {exc}"
            ) from exc

        # ChromaDB returns nested lists: results["ids"][0], etc.
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not ids:
            logger.warning("ChromaDB returned no results.")
            return []

        logger.debug(
            "ChromaDB returned %d raw results.", len(ids)
        )

        # Build RetrievedDocument objects with raw scores.
        raw_results: List[RetrievedDocument] = []

        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            text_content = documents[i] if i < len(documents) else None
            distance = distances[i] if i < len(distances) else 0.0

            # ChromaDB returns cosine *distance* (1 - similarity).
            # Convert to similarity score: score = 1 - distance.
            similarity = 1.0 - distance

            page_num = int(metadata.get("page_num", 1))
            paper_id = metadata.get("doc_id", doc_id)
            paper_title = metadata.get("paper_title", paper_id)

            citation = SourceCitation(
                paper_title=paper_title,
                paper_id=paper_id,
                arxiv_url=f"https://arxiv.org/abs/{paper_id}",
                page_numbers=[page_num],
                relevance_score=similarity,
                text_snippet=text_content[:200] if text_content else "",
            )

            raw_results.append(
                RetrievedDocument(
                    doc_id=paper_id,
                    page_num=page_num,
                    score=similarity,
                    image=None,
                    text=text_content,
                    source_citation=citation,
                    retrieval_method="scincl",
                )
            )

        # Normalise scores to [0, 1].
        normalised_results = self.normalize_scores(raw_results)

        logger.info(
            "Text retrieval complete — returned %d results "
            "(scores: %.4f … %.4f).",
            len(normalised_results),
            normalised_results[0].score if normalised_results else 0.0,
            normalised_results[-1].score if normalised_results else 0.0,
        )

        return normalised_results

    # -----------------------------------------------------------------
    # normalize_scores
    # -----------------------------------------------------------------

    @staticmethod
    def normalize_scores(
        results: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Min-max normalise retrieval scores to the ``[0, 1]`` range.

        Applies min-max normalisation so that the lowest score maps to
        0.0 and the highest maps to 1.0.  This is necessary before
        fusing scores from different retrieval backends (ColPali vs.
        SciNCL), as their raw score scales differ significantly:

        * ColPali MaxSim scores can range from 0 to ~200+ (sum of
          dot products across all query tokens).
        * SciNCL cosine similarity scores range from 0 to 1.

        Without normalisation, ColPali scores would dominate the fusion.

        If all scores are identical (zero variance), every score is
        set to 1.0 (all equally relevant).

        Args:
            results: List of :class:`RetrievedDocument` objects with
                raw scores.

        Returns:
            A new list of :class:`RetrievedDocument` objects with
            scores normalised to ``[0, 1]``.  The original list is
            not modified.
        """
        if not results:
            return []

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        normalised: List[RetrievedDocument] = []

        for r in results:
            if score_range < 1e-9:
                # All scores are identical — assign uniform score.
                norm_score = 1.0
            else:
                norm_score = (r.score - min_score) / score_range

            # Build a new RetrievedDocument with the normalised score.
            # We create a new SourceCitation with the normalised score
            # as well for consistency.
            new_citation = SourceCitation(
                paper_title=r.source_citation.paper_title,
                paper_id=r.source_citation.paper_id,
                arxiv_url=r.source_citation.arxiv_url,
                page_numbers=r.source_citation.page_numbers,
                relevance_score=norm_score,
                page_images=r.source_citation.page_images,
                text_snippet=r.source_citation.text_snippet,
            )

            normalised.append(
                RetrievedDocument(
                    doc_id=r.doc_id,
                    page_num=r.page_num,
                    score=norm_score,
                    image=r.image,
                    text=r.text,
                    source_citation=new_citation,
                    retrieval_method=r.retrieval_method,
                )
            )

        return normalised

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _to_list(embedding: object) -> List[float]:
        """Convert various embedding formats to a plain list of floats.

        Supports Python lists, NumPy arrays, and PyTorch tensors.

        Args:
            embedding: The query embedding in any supported format.

        Returns:
            A list of floats.

        Raises:
            TypeError: If the embedding format is not supported.
            ValueError: If the resulting list is empty.
        """
        import numpy as np
        import torch

        if isinstance(embedding, list):
            result = embedding
        elif isinstance(embedding, np.ndarray):
            result = embedding.flatten().tolist()
        elif isinstance(embedding, torch.Tensor):
            result = embedding.flatten().tolist()
        else:
            raise TypeError(
                f"Unsupported embedding type: {type(embedding).__name__}.  "
                f"Expected list, numpy.ndarray, or torch.Tensor."
            )

        if not result:
            raise ValueError("Query embedding is empty.")

        return result
