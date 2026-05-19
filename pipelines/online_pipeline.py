"""
Online Pipeline — Query-Time Retrieval-Augmented Generation.

Implements the online (query-time) pipeline for the Scientific Multimodal
RAG system.  Each query goes through the full encode → retrieve → fuse →
generate → verify cycle, with staggered model loading to stay within
GPU memory constraints.

Pipeline Steps (per query)
--------------------------
1. **Validate query** — Check length, minimum words, and non-empty.
2. **ColPali encode query** — Multi-vector query embedding → **UNLOAD**
3. **SciNCL encode query** — Dense 768-dim query embedding → **UNLOAD**
4. **Fusion retrieval** — Combine ColPali MaxSim + SciNCL ChromaDB
   results with weighted score fusion (0.7 / 0.3).
5. **Context building** — Assemble system/user prompts, page images,
   and text context from retrieved documents.
6. **Qwen2-VL generate** — Produce answer with citations → **UNLOAD**
7. **Self-check** — Attribution, faithfulness, confidence verification.
8. **Return RAGResult** — Bundle answer, sources, scores, timing.

Timing Estimates (Kaggle P100)
------------------------------
* ColPali query encoding: ~1-2 s
* SciNCL query encoding: ~0.3-0.5 s
* Retrieval + fusion: ~0.5 s
* Context building: ~0.05 s
* Qwen2-VL generation: ~3-8 s
* Self-check: ~0.01 s
* **Total per query: ~5-15 s** (depending on retries)

Example:
    >>> from pipelines.online_pipeline import OnlinePipeline
    >>> pipeline = OnlinePipeline(config_path="configs/pipeline_config.yaml")
    >>> result = pipeline.query("What is the Vision Transformer architecture?")
    >>> print(result.answer)
    >>> print(f"Confidence: {result.confidence:.1%}")
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.generation.rag_generator import RAGGenerator, RAGResult
from src.retrieval.base_retriever import RetrievedDocument, SourceCitation
from src.utils.config_loader import load_config, resolve_paths
from src.utils.device import free_vram, get_device, print_gpu_info
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# OnlinePipeline
# ---------------------------------------------------------------------------

class OnlinePipeline:
    """Query-time RAG pipeline with staggered model loading.

    Orchestrates the full online pipeline for answering questions
    about scientific papers.  Models are loaded one at a time to
    respect GPU memory constraints, then immediately unloaded after
    use.

    **Timing estimate**: ~5-15 seconds per query on Kaggle P100.

    Args:
        config_path: Path to the pipeline configuration YAML file.
            Defaults to ``"configs/pipeline_config.yaml"``.

    Example:
        >>> pipeline = OnlinePipeline()
        >>> result = pipeline.query("How does self-attention work?")
        >>> print(result.answer)
        >>> print(f"Sources: {len(result.sources)}")
    """

    def __init__(
        self, config_path: str = "configs/pipeline_config.yaml"
    ) -> None:
        self.config_path = config_path
        self._load_all_configs()
        self._rag_generator: Optional[RAGGenerator] = None

        logger.info("OnlinePipeline initialised.")

    # -----------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------

    def _load_all_configs(self) -> None:
        """Load and merge all configuration files."""
        logger.info("Loading configurations from: %s", self.config_path)

        self.data_config = load_config("data_config")
        self.model_config = load_config("model_config")
        self.pipeline_config = load_config("pipeline_config")
        self.retrieval_config = load_config("retrieval_config")

        # Merge into a single config for RAGGenerator
        self.merged_config = {
            "data": self.data_config.get("data", {}),
            "paths": self.data_config.get("paths", {}),
            "parsing": self.data_config.get("parsing", {}),
            "models": self.model_config.get("models", {}),
            "pipeline": self.pipeline_config.get("pipeline", {}),
            "retrieval": self.retrieval_config.get("retrieval", {}),
        }

        # Resolve paths for the current environment
        self.merged_config = resolve_paths(self.merged_config)

        # Extract commonly used settings
        self._paths = self.merged_config.get("paths", {})
        self._pipeline_cfg = self.merged_config.get("pipeline", {})
        self._online_cfg = self._pipeline_cfg.get("online", {})
        self._models_cfg = self.merged_config.get("models", {})
        self._retrieval_cfg = self.merged_config.get("retrieval", {})

        # Online pipeline settings
        self._max_query_length = self._online_cfg.get("max_query_length", 200)
        self._min_query_words = self._online_cfg.get("min_query_words", 3)

        logger.info("All configurations loaded and paths resolved.")

    # -----------------------------------------------------------------
    # Query validation
    # -----------------------------------------------------------------

    def _validate_query(self, question: str) -> None:
        """Validate a user query before processing.

        Checks that the query is non-empty, meets the minimum word
        count, and does not exceed the maximum length.

        Args:
            question: The user's question string.

        Raises:
            ValueError: If the query is empty, too short, or too long.
        """
        if not question or not question.strip():
            raise ValueError(
                "Query must not be empty. Please provide a question."
            )

        stripped = question.strip()
        word_count = len(stripped.split())

        if word_count < self._min_query_words:
            raise ValueError(
                f"Query must contain at least {self._min_query_words} "
                f"words, got {word_count}. Please be more specific."
            )

        if len(stripped) > self._max_query_length:
            raise ValueError(
                f"Query must not exceed {self._max_query_length} "
                f"characters, got {len(stripped)}. Please shorten your "
                f"question."
            )

        logger.debug(
            "Query validated — %d words, %d chars.",
            word_count, len(stripped),
        )

    # -----------------------------------------------------------------
    # _get_rag_generator — lazy init
    # -----------------------------------------------------------------

    def _get_rag_generator(self) -> RAGGenerator:
        """Get or create the RAGGenerator instance (lazy initialization).

        The generator is created once and reused for subsequent queries
        within the same session.

        Returns:
            An initialized :class:`RAGGenerator` instance.
        """
        if self._rag_generator is None:
            logger.info("Initializing RAGGenerator (lazy init)…")
            self._rag_generator = RAGGenerator(self.merged_config)
        return self._rag_generator

    # -----------------------------------------------------------------
    # query — main entry point
    # -----------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """Execute the online RAG pipeline for a single question.

        Runs the complete pipeline: validate → encode → retrieve →
        fuse → build context → generate → self-check.

        **Timing estimate**: ~5-15 seconds per query on Kaggle P100,
        depending on retrieval size and generation retries.

        Args:
            question: The user's natural-language question.

        Returns:
            An :class:`RAGResult` containing the generated answer,
            source citations, retrieval scores, check result, timing,
            and retry count.

        Raises:
            ValueError: If the query fails validation (empty, too short,
                or too long).

        Example:
            >>> result = pipeline.query("What is the Vision Transformer?")
            >>> print(result.answer)
            >>> print(f"Confidence: {result.confidence:.1%}")
            >>> for src in result.sources:
            ...     print(f"  - {src.paper_title} (Page {src.page_numbers})")
        """
        t_start = time.time()

        # Step 1: Validate query
        logger.info("Online pipeline — query: '%s'", question[:80])
        self._validate_query(question)

        # Step 2-8: Delegate to RAGGenerator which handles the full
        # encode → retrieve → fuse → generate → self-check pipeline
        # with staggered model loading and unloading.
        generator = self._get_rag_generator()
        result = generator.query(question)

        total_time = time.time() - t_start
        logger.info(
            "Online pipeline complete — time: %.2f s, "
            "confidence: %.2f, retries: %d",
            total_time,
            result.confidence,
            result.retries,
        )

        return result

    # -----------------------------------------------------------------
    # batch_query
    # -----------------------------------------------------------------

    def batch_query(self, questions: List[str]) -> List[RAGResult]:
        """Process a batch of questions sequentially.

        Runs the online pipeline for each question in order, collecting
        results.  If a query fails validation or processing, the error
        is logged and a fallback :class:`RAGResult` is returned for
        that question (with empty answer and zero confidence).

        **Timing estimate**: ~5-15 seconds per question, so a batch
        of 10 questions takes ~1-3 minutes.

        Args:
            questions: List of question strings to process.

        Returns:
            A list of :class:`RAGResult` objects, one per question,
            in the same order as the input.

        Example:
            >>> questions = [
            ...     "What is the Vision Transformer?",
            ...     "How does self-attention work?",
            ...     "What datasets were used for evaluation?",
            ... ]
            >>> results = pipeline.batch_query(questions)
            >>> for q, r in zip(questions, results):
            ...     print(f"Q: {q}")
            ...     print(f"A: {r.answer[:100]}…")
            ...     print(f"Confidence: {r.confidence:.1%}")
            ...     print()
        """
        logger.info("Batch query — %d questions.", len(questions))

        results: List[RAGResult] = []

        for i, question in enumerate(questions):
            logger.info(
                "Processing question %d/%d: '%s'",
                i + 1, len(questions), question[:60],
            )

            try:
                result = self.query(question)
                results.append(result)
            except ValueError as exc:
                logger.error(
                    "Query validation failed for question %d: %s",
                    i + 1, exc,
                )
                # Return a fallback RAGResult
                from src.generation.self_check import CheckResult

                results.append(
                    RAGResult(
                        answer=f"Error: {exc}",
                        confidence=0.0,
                        sources=[],
                        retrieval_scores={},
                        check_result=CheckResult(
                            passed=False,
                            attribution_passed=False,
                            faithfulness_passed=False,
                            confidence=0.0,
                            confidence_passed=False,
                            details=f"Query validation failed: {exc}",
                        ),
                        total_time=0.0,
                        retries=0,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Unexpected error on question %d: %s",
                    i + 1, exc,
                )
                from src.generation.self_check import CheckResult

                results.append(
                    RAGResult(
                        answer="",
                        confidence=0.0,
                        sources=[],
                        retrieval_scores={},
                        check_result=CheckResult(
                            passed=False,
                            attribution_passed=False,
                            faithfulness_passed=False,
                            confidence=0.0,
                            confidence_passed=False,
                            details=f"Unexpected error: {exc}",
                        ),
                        total_time=0.0,
                        retries=0,
                    )
                )

        logger.info(
            "Batch query complete — %d results out of %d questions.",
            len(results), len(questions),
        )

        return results
