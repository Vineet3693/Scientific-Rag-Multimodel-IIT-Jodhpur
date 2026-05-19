"""
Fusion Retriever — Weighted Score Fusion of ColPali and SciNCL.

Combines retrieval results from the vision-based ColPali retriever and
the text-based SciNCL retriever using weighted score fusion.  This
hybrid approach leverages the complementary strengths of both backends:

* **ColPali (vision)** excels at capturing document layout, figures,
  tables, and visual structure — information that pure text retrieval
  misses entirely.
* **SciNCL (text)** excels at semantic text matching, especially for
  keyword-rich queries where the relevant content is in the prose.

Weight Rationale: 0.7 / 0.3 Split
-----------------------------------
The 70/30 weight split in favour of ColPali is deliberate and reflects
the unique requirements of scientific document RAG:

1. **Layout matters**: Scientific papers convey critical information
   through figures, tables, equations, and captions.  ColPali's
   vision embeddings capture these visual elements natively, while
   text retrieval only sees OCR/extracted text (which often loses
   spatial relationships and table structure).

2. **Multi-modal queries**: Users frequently ask about visual content
   ("What does the architecture diagram look like?", "What are the
   results in Table 3?").  ColPali can match these queries to the
   correct pages even when the text alone would be ambiguous.

3. **Text as supplement**: SciNCL provides a safety net for queries
   that are purely textual and where the relevant information is in
   the body text, not in figures.  The 30% weight ensures that text
   relevance still influences ranking without overpowering the visual
   signal.

Both score sets are independently min-max normalised to ``[0, 1]``
before fusion, ensuring that the different raw score scales (ColPali
MaxSim ≈ 0–200+ vs. SciNCL cosine ≈ 0–1) do not bias the result.

Example:
    >>> from src.retrieval import ColPaliRetriever, TextRetriever, FusionRetriever
    >>> colpali = ColPaliRetriever(npy_dir="data/indices/multivectors/")
    >>> text = TextRetriever(chroma_dir="data/indices/chroma_index/")
    >>> fusion = FusionRetriever(colpali, text)
    >>> results = fusion.retrieve(query_colpali, query_scincl, top_k=5)
    >>> for doc in results:
    ...     print(doc.doc_id, doc.page_num, f"{doc.score:.4f}", doc.retrieval_method)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.retrieval.base_retriever import RetrievedDocument, SourceCitation
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.text_retriever import TextRetriever
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FusionRetriever
# ---------------------------------------------------------------------------

class FusionRetriever:
    """Weighted score fusion of ColPali and SciNCL retrieval results.

    Retrieves results from both backends, normalises their scores to a
    common ``[0, 1]`` range, and combines them with configurable
    weights.  Pages that appear in both result sets receive the
    weighted sum of their normalised scores; pages that appear in only
    one set receive the weighted score from that set alone.

    Args:
        colpali_retriever: A :class:`ColPaliRetriever` instance with
            index already loaded.
        text_retriever: A :class:`TextRetriever` instance with index
            already loaded.
        colpali_weight: Weight for ColPali scores in the fusion.
            Defaults to 0.7 (70%).
        scincl_weight: Weight for SciNCL scores in the fusion.
            Defaults to 0.3 (30%).
        top_k: Default number of final results to return.

    Raises:
        ValueError: If weights do not sum to approximately 1.0.
    """

    def __init__(
        self,
        colpali_retriever: ColPaliRetriever,
        text_retriever: TextRetriever,
        colpali_weight: float = 0.7,
        scincl_weight: float = 0.3,
        top_k: int = 5,
    ) -> None:
        if abs(colpali_weight + scincl_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got "
                f"{colpali_weight} + {scincl_weight} = "
                f"{colpali_weight + scincl_weight}"
            )

        self.colpali_retriever = colpali_retriever
        self.text_retriever = text_retriever
        self.colpali_weight = colpali_weight
        self.scincl_weight = scincl_weight
        self.top_k = top_k

        logger.info(
            "FusionRetriever initialised — colpali_weight=%.2f, "
            "scincl_weight=%.2f, top_k=%d",
            self.colpali_weight,
            self.scincl_weight,
            self.top_k,
        )

    # -----------------------------------------------------------------
    # retrieve
    # -----------------------------------------------------------------

    def retrieve(
        self,
        query_embedding_colpali: object,
        query_embedding_scincl: object,
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        """Retrieve and fuse results from both backends.

        The fusion algorithm proceeds as follows:

        1. Retrieve ``top_k * 2`` results from each backend to ensure
           sufficient overlap coverage.
        2. Min-max normalise each result set independently to ``[0, 1]``.
        3. For pages that appear in **both** result sets, the fused
           score is::

               fused = colpali_weight * colpali_score
                     + scincl_weight * scincl_score

        4. For pages that appear in **only one** result set, the fused
           score is the weighted score from that set::

               colpali_only: fused = colpali_weight * colpali_score
               scincl_only:  fused = scincl_weight * scincl_score

        5. Sort all pages by descending fused score and return the
           top_k.

        This approach ensures that:
        * Pages confirmed by both modalities are boosted.
        * Pages found by only one modality are not penalised to zero
          but receive a proportionally reduced score.
        * The 0.7/0.3 weighting gives ColPali (vision) the dominant
          influence while still allowing SciNCL (text) to shift
          rankings when text relevance is strong.

        Args:
            query_embedding_colpali: Query embedding for ColPali —
                a ``torch.Tensor`` of shape ``(num_tokens, 128)``.
            query_embedding_scincl: Query embedding for SciNCL —
                a list of 768 floats (or compatible format).
            top_k: Number of final fused results to return.

        Returns:
            A list of :class:`RetrievedDocument` objects sorted by
            descending fused score, with ``retrieval_method="fusion"``.

        Raises:
            ValueError: If *top_k* is not a positive integer.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Expand search space to improve overlap.
        expanded_k = top_k * 2

        logger.info(
            "Fusion retrieval — expanded_k=%d, final top_k=%d",
            expanded_k,
            top_k,
        )

        # Step 1: Retrieve from both backends.
        logger.debug("Retrieving from ColPali (top_k=%d)…", expanded_k)
        colpali_results = self.colpali_retriever.retrieve(
            query_embedding_colpali, top_k=expanded_k
        )

        logger.debug("Retrieving from SciNCL (top_k=%d)…", expanded_k)
        text_results = self.text_retriever.retrieve(
            query_embedding_scincl, top_k=expanded_k
        )

        logger.info(
            "Raw results — ColPali: %d, SciNCL: %d",
            len(colpali_results),
            len(text_results),
        )

        # Step 2: Normalise both result sets independently.
        colpali_normalised = self.normalize_scores(colpali_results)
        text_normalised = self.normalize_scores(text_results)

        # Step 3: Build lookup maps keyed by (doc_id, page_num).
        colpali_map: Dict[Tuple[str, int], RetrievedDocument] = {}
        for doc in colpali_normalised:
            colpali_map[(doc.doc_id, doc.page_num)] = doc

        text_map: Dict[Tuple[str, int], RetrievedDocument] = {}
        for doc in text_normalised:
            text_map[(doc.doc_id, doc.page_num)] = doc

        # Step 4: Compute fused scores.
        all_keys = set(colpali_map.keys()) | set(text_map.keys())

        fused_results: List[RetrievedDocument] = []

        for key in all_keys:
            doc_id, page_num = key
            colpali_doc = colpali_map.get(key)
            text_doc = text_map.get(key)

            in_colpali = colpali_doc is not None
            in_text = text_doc is not None

            if in_colpali and in_text:
                # Page found by both backends: weighted sum.
                fused_score = (
                    self.colpali_weight * colpali_doc.score
                    + self.scincl_weight * text_doc.score
                )
                # Prefer the ColPali document for image data, but
                # fall back to text document's text content.
                base_doc = colpali_doc
                combined_text = (
                    text_doc.text
                    if text_doc.text
                    else base_doc.text
                )
                combined_image = base_doc.image or text_doc.image

                logger.debug(
                    "Fused (both) — key=%s, colpali=%.4f, text=%.4f, "
                    "fused=%.4f",
                    key,
                    colpali_doc.score,
                    text_doc.score,
                    fused_score,
                )
            elif in_colpali:
                # Page found only by ColPali.
                fused_score = self.colpali_weight * colpali_doc.score
                base_doc = colpali_doc
                combined_text = base_doc.text
                combined_image = base_doc.image

                logger.debug(
                    "Fused (colpali only) — key=%s, score=%.4f, "
                    "fused=%.4f",
                    key,
                    colpali_doc.score,
                    fused_score,
                )
            else:
                # Page found only by SciNCL.
                fused_score = self.scincl_weight * text_doc.score
                base_doc = text_doc
                combined_text = base_doc.text
                combined_image = base_doc.image

                logger.debug(
                    "Fused (text only) — key=%s, score=%.4f, "
                    "fused=%.4f",
                    key,
                    text_doc.score,
                    fused_score,
                )

            # Build a new SourceCitation with the fused score.
            new_citation = SourceCitation(
                paper_title=base_doc.source_citation.paper_title,
                paper_id=base_doc.source_citation.paper_id,
                arxiv_url=base_doc.source_citation.arxiv_url,
                page_numbers=base_doc.source_citation.page_numbers,
                relevance_score=fused_score,
                page_images=base_doc.source_citation.page_images,
                text_snippet=base_doc.source_citation.text_snippet,
            )

            fused_results.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    page_num=page_num,
                    score=fused_score,
                    image=combined_image,
                    text=combined_text,
                    source_citation=new_citation,
                    retrieval_method="fusion",
                )
            )

        # Step 5: Sort by descending fused score and return top_k.
        fused_results.sort(key=lambda r: r.score, reverse=True)
        final_results = fused_results[:top_k]

        logger.info(
            "Fusion retrieval complete — %d candidates, returned %d "
            "(scores: %.4f … %.4f).",
            len(fused_results),
            len(final_results),
            final_results[0].score if final_results else 0.0,
            final_results[-1].score if final_results else 0.0,
        )

        return final_results

    # -----------------------------------------------------------------
    # normalize_scores
    # -----------------------------------------------------------------

    @staticmethod
    def normalize_scores(
        results: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Min-max normalise retrieval scores to the ``[0, 1]`` range.

        This is a shared utility that performs the same normalisation
        as :meth:`TextRetriever.normalize_scores`.  It is duplicated
        here so that :class:`FusionRetriever` is self-contained and
        does not depend on a specific retriever's implementation.

        If all scores are identical (zero variance), every score is
        set to 1.0.

        Args:
            results: List of :class:`RetrievedDocument` objects with
                raw (or previously normalised) scores.

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
                norm_score = 1.0
            else:
                norm_score = (r.score - min_score) / score_range

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
