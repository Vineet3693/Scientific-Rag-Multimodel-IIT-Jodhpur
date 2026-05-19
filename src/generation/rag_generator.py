"""
RAG Generator — End-to-End Retrieval-Augmented Generation Pipeline.

Implements the full RAG pipeline for scientific document question-answering
with staggered model loading, hybrid retrieval, score fusion, answer
generation via Qwen2-VL, and three-level self-check verification.

Pipeline Overview
-----------------
The :meth:`RAGGenerator.query` method executes the following steps:

1. **Encode query with ColPali** → multi-vector embedding → **UNLOAD**
2. **Encode query with SciNCL** → dense 768-dim embedding → **UNLOAD**
3. **ColPali MaxSim retrieval** — score query against ``.npy`` index
4. **SciNCL ChromaDB retrieval** — ANN search in ChromaDB
5. **Score fusion** — weighted combination (0.7 ColPali + 0.3 SciNCL)
6. **Build context** — assemble system/user prompts, page images, text
7. **Generate answer with Qwen2-VL** → **UNLOAD**
8. **Self-check** — attribution, faithfulness, confidence
9. **Retry loop** — if confidence < 0.6, retry up to 2 more times

Error Handling / Fallback Chains
---------------------------------
The pipeline degrades gracefully when individual components fail:

* ColPali fails → SciNCL only (weight = 1.0)
* SciNCL fails → ColPali only (weight = 1.0)
* Both fail → keyword TF-IDF search over stored text
* Qwen2-VL fails → return retrieved sources only (no answer)

Timing Estimates (Kaggle P100)
------------------------------
* ColPali query encoding: ~1-2 s
* SciNCL query encoding: ~0.3-0.5 s
* ColPali MaxSim retrieval: ~0.5 s (for ~100 pages)
* SciNCL ChromaDB retrieval: ~0.1 s
* Context building: ~0.05 s
* Qwen2-VL generation: ~3-8 s
* Self-check: ~0.01 s
* **Total per query (no retries): ~5-12 s**

Example:
    >>> from src.generation.rag_generator import RAGGenerator
    >>> config = {
    ...     "models": {"colpali": {"model_name": "vidore/colpali-v1.2"}},
    ...     "retrieval": {"top_k": 5, "colpali_weight": 0.7, "scincl_weight": 0.3},
    ... }
    >>> generator = RAGGenerator(config)
    >>> result = generator.query("What is the attention mechanism?")
    >>> print(result.answer)
    >>> print(result.check_result.passed)
"""

from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.generation.self_check import CheckResult, SelfChecker
from src.retrieval.base_retriever import RetrievedDocument, SourceCitation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RAGResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Complete result of a RAG query, including answer and metadata.

    Bundles the generated answer, all intermediate data (sources,
    scores, check results), and timing information for end-to-end
    observability.

    Attributes:
        answer: The generated answer text.  May be empty if VLM
            generation failed (fallback to sources-only mode).
        confidence: The model's self-reported confidence score for
            the final answer, in the range ``[0, 1]``.
        sources: A list of :class:`SourceCitation` objects for all
            retrieved documents used to construct the answer.
        retrieval_scores: A dictionary mapping retrieval method names
            (e.g. ``"colpali"``, ``"scincl"``, ``"fusion"``) to their
            top scores or aggregated score statistics.
        check_result: The :class:`CheckResult` from the three-level
            self-check pipeline.
        total_time: Total wall-clock time for the entire query
            pipeline in seconds.
        retries: Number of retry attempts that were needed (0 if the
            first attempt passed all checks).
    """

    answer: str
    confidence: float
    sources: List[SourceCitation]
    retrieval_scores: Dict[str, float]
    check_result: CheckResult
    total_time: float
    retries: int


# ---------------------------------------------------------------------------
# RAGGenerator class
# ---------------------------------------------------------------------------

class RAGGenerator:
    """End-to-end RAG pipeline with staggered model loading.

    Orchestrates query encoding, hybrid retrieval, score fusion,
    context building, VLM generation, and self-check verification.
    Models are loaded one at a time (load → use → unload) to stay
    within VRAM constraints on GPU-constrained environments.

    Args:
        config: Configuration dictionary, typically loaded from
            ``configs/model_config.yaml``, ``configs/retrieval_config.yaml``,
            and ``configs/data_config.yaml``.  Expected top-level keys:

            * ``models`` — Model parameters (colpali, scincl, qwen2vl).
            * ``retrieval`` — Retrieval parameters (top_k, weights, paths).
            * ``paths`` — Data directory paths.

    Example:
        >>> from src.utils.config_loader import load_config
        >>> config = load_config("model_config")
        >>> generator = RAGGenerator(config)
        >>> result = generator.query("What is the attention mechanism?")
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # Extract sub-configs with safe defaults
        self._models_cfg = config.get("models", {})
        self._retrieval_cfg = config.get("retrieval", {})
        self._paths_cfg = config.get("paths", {})

        # Retrieval parameters
        self._top_k = self._retrieval_cfg.get("top_k", 5)
        self._colpali_weight = self._retrieval_cfg.get("colpali_weight", 0.7)
        self._scincl_weight = self._retrieval_cfg.get("scincl_weight", 0.3)

        # Qwen2-VL parameters
        qwen_cfg = self._models_cfg.get("qwen2vl", {})
        self._qwen_model_name = qwen_cfg.get(
            "model_name", "Qwen/Qwen2-VL-2B-Instruct"
        )
        self._qwen_max_new_tokens = qwen_cfg.get("max_new_tokens", 512)
        self._qwen_temperature = qwen_cfg.get("temperature", 0.3)
        self._confidence_threshold = qwen_cfg.get("confidence_threshold", 0.6)
        self._max_retries = qwen_cfg.get("max_retries", 2)

        # Self-checker
        self._checker = SelfChecker(
            confidence_threshold=self._confidence_threshold,
            faithfulness_threshold=0.3,
        )

        # Lazy-loaded components (initialised on first use)
        self._colpali_embedder: Optional[Any] = None
        self._scincl_embedder: Optional[Any] = None
        self._colpali_retriever: Optional[Any] = None
        self._text_retriever: Optional[Any] = None
        self._fusion_retriever: Optional[Any] = None
        self._context_builder: Optional[Any] = None

        logger.info(
            "RAGGenerator initialised — top_k=%d, "
            "colpali_weight=%.2f, scincl_weight=%.2f, "
            "confidence_threshold=%.2f, max_retries=%d",
            self._top_k,
            self._colpali_weight,
            self._scincl_weight,
            self._confidence_threshold,
            self._max_retries,
        )

    # -----------------------------------------------------------------
    # query (main entry point)
    # -----------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """Execute the full RAG pipeline for a user question.

        Runs the complete pipeline: encode → retrieve → fuse → build
        context → generate → self-check, with automatic retries if
        the confidence check fails.

        **Timing estimate**: 5-12 seconds per query on a Kaggle P100
        (no retries).  With retries, up to 3x that time.

        Args:
            question: The user's natural-language question.

        Returns:
            An :class:`RAGResult` containing the answer, sources,
            scores, check result, timing, and retry count.

        Raises:
            ValueError: If *question* is empty or too short.
        """
        t_start = time.time()

        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        logger.info("RAG query started: '%s'", question[:80])

        # ----------------------------------------------------------
        # Step 1 & 2: Encode query with ColPali and SciNCL
        # ----------------------------------------------------------
        query_embedding_colpali = None
        query_embedding_scincl = None
        colpali_ok = False
        scincl_ok = False

        # --- ColPali encoding ---
        try:
            query_embedding_colpali = self._encode_colpali(question)
            colpali_ok = True
            logger.info("ColPali query encoding succeeded.")
        except Exception as exc:
            logger.error("ColPali query encoding failed: %s", exc)

        # --- SciNCL encoding ---
        try:
            query_embedding_scincl = self._encode_scincl(question)
            scincl_ok = True
            logger.info("SciNCL query encoding succeeded.")
        except Exception as exc:
            logger.error("SciNCL query encoding failed: %s", exc)

        # ----------------------------------------------------------
        # Step 3 & 4: Retrieval with fallback chains
        # ----------------------------------------------------------
        colpali_results: List[RetrievedDocument] = []
        scincl_results: List[RetrievedDocument] = []

        if colpali_ok:
            try:
                colpali_results = self._retrieve_colpali(query_embedding_colpali)
            except Exception as exc:
                logger.error("ColPali retrieval failed: %s", exc)
                colpali_ok = False

        if scincl_ok:
            try:
                scincl_results = self._retrieve_scincl(query_embedding_scincl)
            except Exception as exc:
                logger.error("SciNCL retrieval failed: %s", exc)
                scincl_ok = False

        # ----------------------------------------------------------
        # Step 5: Score fusion with fallback weights
        # ----------------------------------------------------------
        fused_results = self._fuse_results(
            colpali_results, colpali_ok,
            scincl_results, scincl_ok,
        )

        # Fallback: TF-IDF keyword search if both retrievers failed
        if not fused_results:
            logger.warning(
                "Both ColPali and SciNCL retrieval failed — "
                "falling back to keyword TF-IDF search."
            )
            fused_results = self._keyword_search(question)

        # ----------------------------------------------------------
        # Step 6: Build context
        # ----------------------------------------------------------
        context_text = self._build_context_text(fused_results)
        page_images = self._collect_page_images(fused_results)
        sources = self._extract_sources(fused_results)
        retrieval_scores = self._extract_scores(
            colpali_results, scincl_results, fused_results,
        )

        # ----------------------------------------------------------
        # Step 7, 8, 9: Generate → self-check → retry loop
        # ----------------------------------------------------------
        best_result: Optional[RAGResult] = None
        retries = 0

        for attempt in range(self._max_retries + 1):
            logger.info(
                "Generation attempt %d/%d",
                attempt + 1,
                self._max_retries + 1,
            )

            # Step 7: Generate answer with Qwen2-VL
            answer, confidence = self._generate_answer(
                question, context_text, page_images
            )

            # Step 8: Self-check
            check_result = self._checker.check(
                answer=answer,
                context=context_text,
                confidence=confidence,
            )

            # Build partial result
            total_time = time.time() - t_start
            current_result = RAGResult(
                answer=answer,
                confidence=confidence,
                sources=sources,
                retrieval_scores=retrieval_scores,
                check_result=check_result,
                total_time=total_time,
                retries=attempt,
            )

            # Track best result
            if best_result is None or confidence > best_result.confidence:
                best_result = current_result

            # Step 9: Check if retry is needed
            if confidence >= self._confidence_threshold:
                logger.info(
                    "Confidence %.2f >= threshold %.2f — accepting answer.",
                    confidence,
                    self._confidence_threshold,
                )
                best_result = current_result
                break

            if attempt < self._max_retries:
                logger.warning(
                    "Confidence %.2f < threshold %.2f — retrying "
                    "(attempt %d/%d).",
                    confidence,
                    self._confidence_threshold,
                    attempt + 1,
                    self._max_retries,
                )
                retries = attempt + 1

        total_time = time.time() - t_start

        # Update timing on best result
        if best_result is not None:
            best_result.total_time = total_time
            best_result.retries = retries

        logger.info(
            "RAG query complete — time: %.2f s, retries: %d, "
            "confidence: %.2f, check passed: %s",
            total_time,
            retries,
            best_result.confidence if best_result else 0.0,
            best_result.check_result.passed if best_result else False,
        )

        # If we never got a result (very unlikely), return a minimal one
        if best_result is None:
            best_result = RAGResult(
                answer="",
                confidence=0.0,
                sources=sources,
                retrieval_scores=retrieval_scores,
                check_result=CheckResult(
                    passed=False,
                    attribution_passed=False,
                    faithfulness_passed=False,
                    confidence=0.0,
                    confidence_passed=False,
                    details="All generation attempts failed.",
                ),
                total_time=total_time,
                retries=retries,
            )

        return best_result

    # -----------------------------------------------------------------
    # Step 1: ColPali query encoding
    # -----------------------------------------------------------------

    def _encode_colpali(self, question: str) -> Any:
        """Encode the query using ColPali and immediately unload.

        Loads the ColPali embedder, encodes the query as a multi-vector
        embedding, then unloads to free VRAM (~2.5 GB).

        Args:
            question: The user's query text.

        Returns:
            A ``torch.Tensor`` of shape ``(num_tokens, 128)``.

        Raises:
            RuntimeError: If ColPali encoding fails.
        """
        from src.embeddings.colpali_embedder import ColPaliEmbedder

        colpali_cfg = self._models_cfg.get("colpali", {})
        embedder = ColPaliEmbedder(
            model_name=colpali_cfg.get("model_name", "vidore/colpali-v1.2"),
            device=colpali_cfg.get("device", "cuda"),
            torch_dtype=colpali_cfg.get("torch_dtype", "float16"),
        )

        try:
            embedder.load()
            # ColPali processes text as an image of rendered text,
            # but we use the processor directly for text queries.
            # The ColPali processor handles text encoding.
            from colpali_engine.models import ColPaliProcessor

            processor = ColPaliProcessor.from_pretrained(
                colpali_cfg.get("model_name", "vidore/colpali-v1.2")
            )

            inputs = processor(text=[question]).to(embedder._model.device)

            with torch.no_grad():
                embeddings = embedder._model(**inputs)

            # embeddings[0] shape: (num_query_tokens, 128)
            query_embedding = embeddings[0].cpu()

            logger.info(
                "ColPali query encoded — shape: %s",
                tuple(query_embedding.shape),
            )
            return query_embedding

        finally:
            embedder.unload()
            logger.debug("ColPali embedder unloaded after query encoding.")

    # -----------------------------------------------------------------
    # Step 2: SciNCL query encoding
    # -----------------------------------------------------------------

    def _encode_scincl(self, question: str) -> Any:
        """Encode the query using SciNCL and immediately unload.

        Loads the SciNCL embedder, encodes the query as a 768-dim
        dense vector, then unloads to free VRAM (~0.6 GB).

        Args:
            question: The user's query text.

        Returns:
            A list of 768 floats (the query embedding).

        Raises:
            RuntimeError: If SciNCL encoding fails.
        """
        from src.embeddings.scincl_embedder import SciNCLEmbedder

        scincl_cfg = self._models_cfg.get("scincl", {})
        embedder = SciNCLEmbedder(
            model_name=scincl_cfg.get("model_name", "malteos/scincl"),
            device=scincl_cfg.get("device", "cuda"),
            max_length=scincl_cfg.get("max_length", 512),
        )

        try:
            embedder.load()
            output = embedder.embed_text(question)
            query_embedding = output.vectors.tolist()

            logger.info(
                "SciNCL query encoded — dim: %d",
                len(query_embedding),
            )
            return query_embedding

        finally:
            embedder.unload()
            logger.debug("SciNCL embedder unloaded after query encoding.")

    # -----------------------------------------------------------------
    # Step 3: ColPali MaxSim retrieval
    # -----------------------------------------------------------------

    def _retrieve_colpali(
        self, query_embedding: Any
    ) -> List[RetrievedDocument]:
        """Retrieve pages using ColPali MaxSim scoring.

        Args:
            query_embedding: Multi-vector query tensor of shape
                ``(num_tokens, 128)``.

        Returns:
            A list of :class:`RetrievedDocument` from ColPali.

        Raises:
            RuntimeError: If the ColPali index cannot be loaded or
                queried.
        """
        from src.retrieval.colpali_retriever import ColPaliRetriever

        npy_dir = self._paths_cfg.get(
            "multivectors",
            self._retrieval_cfg.get("npy_dir", "data/indices/multivectors/"),
        )

        retriever = ColPaliRetriever(npy_dir=npy_dir, top_k=self._top_k)

        try:
            retriever.load_index(npy_dir)
        except FileNotFoundError:
            # Try resolving path
            alt_dir = str(Path("data/indices/multivectors/"))
            retriever.load_index(alt_dir)

        results = retriever.retrieve(query_embedding, top_k=self._top_k)
        logger.info("ColPali retrieval returned %d results.", len(results))
        return results

    # -----------------------------------------------------------------
    # Step 4: SciNCL ChromaDB retrieval
    # -----------------------------------------------------------------

    def _retrieve_scincl(
        self, query_embedding: Any
    ) -> List[RetrievedDocument]:
        """Retrieve text chunks using SciNCL + ChromaDB.

        Args:
            query_embedding: Dense query vector as a list of 768 floats.

        Returns:
            A list of :class:`RetrievedDocument` from SciNCL.

        Raises:
            RuntimeError: If the ChromaDB index cannot be loaded or
                queried.
        """
        from src.retrieval.text_retriever import TextRetriever

        chroma_dir = self._paths_cfg.get(
            "chroma_index",
            self._retrieval_cfg.get("chroma_persist_dir", "data/indices/chroma_index/"),
        )
        collection_name = self._retrieval_cfg.get("chroma_collection", "sci_text")

        retriever = TextRetriever(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            top_k=self._top_k,
        )

        try:
            retriever.load_index(chroma_dir)
        except FileNotFoundError:
            alt_dir = str(Path("data/indices/chroma_index/"))
            retriever.load_index(alt_dir)

        results = retriever.retrieve(query_embedding, top_k=self._top_k)
        logger.info("SciNCL retrieval returned %d results.", len(results))
        return results

    # -----------------------------------------------------------------
    # Step 5: Score fusion with fallback weights
    # -----------------------------------------------------------------

    def _fuse_results(
        self,
        colpali_results: List[RetrievedDocument],
        colpali_ok: bool,
        scincl_results: List[RetrievedDocument],
        scincl_ok: bool,
    ) -> List[RetrievedDocument]:
        """Fuse ColPali and SciNCL retrieval results.

        Uses the :class:`~src.retrieval.fusion_retriever.FusionRetriever`
        when both backends succeed.  Falls back to a single-backend
        result set (with weight = 1.0) when one fails.

        Args:
            colpali_results: Results from ColPali retrieval.
            colpali_ok: Whether ColPali retrieval succeeded.
            scincl_results: Results from SciNCL retrieval.
            scincl_ok: Whether SciNCL retrieval succeeded.

        Returns:
            A list of fused :class:`RetrievedDocument` objects,
            or a single-backend result list if one failed.
        """
        if colpali_ok and scincl_ok:
            # Both backends succeeded — use full fusion.
            from src.retrieval.fusion_retriever import FusionRetriever
            from src.retrieval.colpali_retriever import ColPaliRetriever
            from src.retrieval.text_retriever import TextRetriever

            # Create dummy retrievers for the fusion interface.
            # We'll use the static normalize method and manual fusion
            # instead, since the retrievers need loaded indices.
            fused = self._manual_fusion(
                colpali_results, scincl_results,
                self._colpali_weight, self._scincl_weight,
            )
            return fused

        elif colpali_ok:
            # ColPali only — use weight 1.0
            logger.warning(
                "SciNCL retrieval failed — using ColPali only (weight=1.0)."
            )
            return colpali_results

        elif scincl_ok:
            # SciNCL only — use weight 1.0
            logger.warning(
                "ColPali retrieval failed — using SciNCL only (weight=1.0)."
            )
            return scincl_results

        else:
            logger.error("Both ColPali and SciNCL retrieval failed.")
            return []

    def _manual_fusion(
        self,
        colpali_results: List[RetrievedDocument],
        scincl_results: List[RetrievedDocument],
        colpali_weight: float,
        scincl_weight: float,
    ) -> List[RetrievedDocument]:
        """Manually fuse two result sets with weighted scores.

        Normalises scores from both result sets to ``[0, 1]``, then
        combines them with the specified weights.  Pages appearing in
        both sets receive the weighted sum; pages in only one set
        receive the weighted score from that set.

        Args:
            colpali_results: ColPali retrieval results.
            scincl_results: SciNCL retrieval results.
            colpali_weight: Weight for ColPali scores.
            scincl_weight: Weight for SciNCL scores.

        Returns:
            Fused results sorted by descending score.
        """
        # Normalise each result set
        norm_colpali = self._normalize_scores(colpali_results)
        norm_scincl = self._normalize_scores(scincl_results)

        # Build lookup maps
        colpali_map: Dict[tuple, RetrievedDocument] = {}
        for doc in norm_colpali:
            colpali_map[(doc.doc_id, doc.page_num)] = doc

        scincl_map: Dict[tuple, RetrievedDocument] = {}
        for doc in norm_scincl:
            scincl_map[(doc.doc_id, doc.page_num)] = doc

        # Compute fused scores
        all_keys = set(colpali_map.keys()) | set(scincl_map.keys())
        fused: List[RetrievedDocument] = []

        for key in all_keys:
            c_doc = colpali_map.get(key)
            s_doc = scincl_map.get(key)

            if c_doc and s_doc:
                fused_score = colpali_weight * c_doc.score + scincl_weight * s_doc.score
                base = c_doc
                combined_text = s_doc.text if s_doc.text else base.text
            elif c_doc:
                fused_score = colpali_weight * c_doc.score
                base = c_doc
                combined_text = base.text
            else:
                fused_score = scincl_weight * s_doc.score
                base = s_doc
                combined_text = base.text

            new_citation = SourceCitation(
                paper_title=base.source_citation.paper_title,
                paper_id=base.source_citation.paper_id,
                arxiv_url=base.source_citation.arxiv_url,
                page_numbers=base.source_citation.page_numbers,
                relevance_score=fused_score,
                page_images=base.source_citation.page_images,
                text_snippet=base.source_citation.text_snippet,
            )

            fused.append(
                RetrievedDocument(
                    doc_id=base.doc_id,
                    page_num=base.page_num,
                    score=fused_score,
                    image=base.image,
                    text=combined_text,
                    source_citation=new_citation,
                    retrieval_method="fusion",
                )
            )

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:self._top_k]

    @staticmethod
    def _normalize_scores(
        results: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Min-max normalise retrieval scores to ``[0, 1]``.

        Args:
            results: List of retrieved documents with raw scores.

        Returns:
            New list with normalised scores.  Original list is not
            modified.
        """
        if not results:
            return []

        scores = [r.score for r in results]
        min_s = min(scores)
        max_s = max(scores)
        score_range = max_s - min_s

        normalised: List[RetrievedDocument] = []
        for r in results:
            norm_score = (
                (r.score - min_s) / score_range
                if score_range > 1e-9
                else 1.0
            )
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
    # Fallback: Keyword TF-IDF search
    # -----------------------------------------------------------------

    def _keyword_search(self, question: str) -> List[RetrievedDocument]:
        """Fallback keyword search using simple TF-IDF scoring.

        Used when both ColPali and SciNCL retrieval fail.  Scans the
        stored text chunks for keyword matches with the query and
        returns the most relevant documents.

        Args:
            question: The user's query text.

        Returns:
            A list of :class:`RetrievedDocument` from keyword matching.
        """
        logger.info("Performing keyword TF-IDF fallback search.")

        # Try to load text from ChromaDB for keyword matching
        try:
            import chromadb

            chroma_dir = self._paths_cfg.get(
                "chroma_index",
                self._retrieval_cfg.get(
                    "chroma_persist_dir", "data/indices/chroma_index/"
                ),
            )
            collection_name = self._retrieval_cfg.get("chroma_collection", "sci_text")

            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_collection(name=collection_name)

            # Use ChromaDB's built-in keyword search
            query_words = re.findall(r"\w+", question.lower())
            query_text = " ".join(query_words)

            results = collection.query(
                query_texts=[query_text],
                n_results=self._top_k,
                include=["documents", "metadatas", "distances"],
            )

            keyword_results: List[RetrievedDocument] = []
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                text_content = documents[i] if i < len(documents) else None
                distance = distances[i] if i < len(distances) else 1.0
                similarity = max(0.0, 1.0 - distance)

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

                keyword_results.append(
                    RetrievedDocument(
                        doc_id=paper_id,
                        page_num=page_num,
                        score=similarity,
                        image=None,
                        text=text_content,
                        source_citation=citation,
                        retrieval_method="keyword_tfidf",
                    )
                )

            logger.info(
                "Keyword TF-IDF search returned %d results.",
                len(keyword_results),
            )
            return keyword_results

        except Exception as exc:
            logger.error("Keyword TF-IDF search also failed: %s", exc)
            return []

    # -----------------------------------------------------------------
    # Step 6: Build context
    # -----------------------------------------------------------------

    def _build_context_text(self, results: List[RetrievedDocument]) -> str:
        """Build text context from retrieved documents.

        Concatenates text from all retrieved documents with citation
        markers for the VLM.

        Args:
            results: Fused retrieval results.

        Returns:
            Assembled text context string.
        """
        if not results:
            return ""

        parts: List[str] = []
        for doc in results:
            citation = f"[Source: {doc.source_citation.paper_title}, Page {doc.page_num}]"
            if doc.text and doc.text.strip():
                parts.append(f"{citation}\n{doc.text.strip()}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def _collect_page_images(self, results: List[RetrievedDocument]) -> List[Any]:
        """Collect non-None page images from retrieval results.

        Args:
            results: Fused retrieval results.

        Returns:
            A list of PIL images.
        """
        images = []
        for doc in results:
            if doc.image is not None:
                images.append(doc.image)
        return images

    def _extract_sources(self, results: List[RetrievedDocument]) -> List[SourceCitation]:
        """Extract deduplicated source citations.

        Args:
            results: Fused retrieval results.

        Returns:
            A deduplicated list of :class:`SourceCitation`.
        """
        seen: set = set()
        sources: List[SourceCitation] = []
        for doc in results:
            key = (doc.source_citation.paper_id, doc.page_num)
            if key not in seen:
                seen.add(key)
                sources.append(doc.source_citation)
        return sources

    def _extract_scores(
        self,
        colpali_results: List[RetrievedDocument],
        scincl_results: List[RetrievedDocument],
        fused_results: List[RetrievedDocument],
    ) -> Dict[str, float]:
        """Extract top scores from each retrieval method.

        Args:
            colpali_results: ColPali results.
            scincl_results: SciNCL results.
            fused_results: Fused results.

        Returns:
            A dictionary mapping method names to their top scores.
        """
        scores: Dict[str, float] = {}
        if colpali_results:
            scores["colpali_top"] = max(r.score for r in colpali_results)
        if scincl_results:
            scores["scincl_top"] = max(r.score for r in scincl_results)
        if fused_results:
            scores["fusion_top"] = max(r.score for r in fused_results)
        return scores

    # -----------------------------------------------------------------
    # Step 7: Generate answer with Qwen2-VL
    # -----------------------------------------------------------------

    def _generate_answer(
        self,
        question: str,
        context_text: str,
        page_images: List[Any],
    ) -> tuple[str, float]:
        """Generate an answer using Qwen2-VL and immediately unload.

        Loads the Qwen2-VL model, generates an answer from the
        context, extracts a confidence score, then unloads to free
        VRAM (~1.5 GB with 4-bit quantization).

        If Qwen2-VL fails, returns a sources-only fallback.

        Args:
            question: The user's question.
            context_text: Assembled text context.
            page_images: Page images for multi-modal input.

        Returns:
            A tuple ``(answer, confidence)`` where *answer* is the
            generated text and *confidence* is a float in ``[0, 1]``.
        """
        try:
            from transformers import (
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
                BitsAndBytesConfig,
            )

            logger.info("Loading Qwen2-VL model: %s", self._qwen_model_name)

            # Configure 4-bit quantization for VRAM efficiency.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._qwen_model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
            model.eval()

            processor = AutoProcessor.from_pretrained(self._qwen_model_name)

            logger.info("Qwen2-VL loaded — generating answer.")

            # Build messages for the chat template.
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific research assistant. "
                        "Answer based strictly on the provided context. "
                        "Cite sources using [Source: Paper Title, Page N]. "
                        "Do not fabricate information."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"## Query\n{question}"},
                        {"type": "text", "text": f"## Context\n{context_text}"},
                    ],
                },
            ]

            # Add page images as multi-modal input.
            for img in page_images[:3]:  # Limit to 3 images for VRAM
                messages[1]["content"].append(
                    {"type": "image"}
                )

            # Prepare inputs
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Collect images for the processor
            image_inputs = page_images[:3]

            inputs = processor(
                text=[text_prompt],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
            ).to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self._qwen_max_new_tokens,
                    temperature=self._qwen_temperature,
                    do_sample=self._qwen_temperature > 0,
                )

            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]
            answer = processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()

            # Extract confidence from the answer.
            confidence = self._extract_confidence(answer)

            logger.info(
                "Qwen2-VL generated answer — length: %d chars, "
                "confidence: %.2f",
                len(answer),
                confidence,
            )

            return answer, confidence

        except Exception as exc:
            logger.error(
                "Qwen2-VL generation failed: %s — returning sources only.",
                exc,
            )
            # Fallback: return sources only, no answer
            return "", 0.0

        finally:
            # Unload Qwen2-VL to free VRAM.
            try:
                if "model" in dir():
                    del model
                if "processor" in dir():
                    del processor
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Qwen2-VL unloaded after generation.")

    # -----------------------------------------------------------------
    # Confidence extraction
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_confidence(answer: str) -> float:
        """Extract a confidence score from the generated answer.

        Looks for explicit confidence indicators in the answer text.
        If none are found, estimates confidence based on the presence
        of citation markers and answer length.

        Args:
            answer: The generated answer text.

        Returns:
            A confidence float in the range ``[0, 1]``.
        """
        if not answer or not answer.strip():
            return 0.0

        # Look for explicit confidence patterns like "confidence: 0.8"
        conf_match = re.search(
            r"confidence[:\s]+([0-9]*\.?[0-9]+)", answer, re.IGNORECASE
        )
        if conf_match:
            try:
                conf = float(conf_match.group(1))
                return min(1.0, max(0.0, conf))
            except ValueError:
                pass

        # Look for "confidence: X/5" or "confidence: X out of 5"
        conf_match = re.search(
            r"confidence[:\s]+([1-5])(?:\s*/\s*5|\s+out\s+of\s+5)?",
            answer,
            re.IGNORECASE,
        )
        if conf_match:
            try:
                conf = int(conf_match.group(1)) / 5.0
                return min(1.0, max(0.0, conf))
            except ValueError:
                pass

        # Heuristic: estimate confidence from citation count and length
        num_citations = len(re.findall(r"\[Source:\s*[^]]+\]", answer, re.IGNORECASE))
        has_citations = num_citations > 0
        answer_length = len(answer.strip())

        # Base confidence from citations
        if has_citations:
            base_conf = 0.7 + min(0.2, num_citations * 0.05)
        else:
            base_conf = 0.3

        # Adjust for answer length (very short = uncertain, very long = detailed)
        if answer_length < 50:
            base_conf *= 0.7
        elif answer_length < 100:
            base_conf *= 0.85

        return min(1.0, max(0.0, base_conf))
