"""
Online Pipeline — Query Facade
================================

Thin wrapper around src.generation.rag_generator.RAGGenerator
that loads config, validates the index, and exposes a clean
``query(question)`` interface for the Streamlit app.
"""

from __future__ import annotations

import os
from typing import Optional


class OnlinePipeline:
    """Online RAG query pipeline.

    Initialises the RAGGenerator with the project config and
    exposes a single ``query()`` method for the Streamlit app.

    Args:
        cfg: Loaded YAML config dict.

    Example::

        pipeline = OnlinePipeline(cfg)
        result   = pipeline.query("How does patch embedding work in ViT?")
        print(result.answer)
        for source in result.sources:
            print(source.paper_title, source.page_numbers)
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._generator = None          # lazy-initialised on first query

    # ─────────────────────────────────────────────────────────────────────────
    # Public: query
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        status_callback: Optional[callable] = None,
    ):
        """Run the full RAG pipeline for a user question.

        Lazily initialises the RAGGenerator on the first call,
        then delegates to it for all subsequent calls.

        Args:
            question: The user's natural-language question.
            status_callback: Optional ``(step_name, message, pct)``
                callback for progress display in the Streamlit UI.

        Returns:
            A :class:`src.generation.rag_generator.RAGResult` with::

                result.answer           — generated answer text
                result.confidence       — float [0, 1]
                result.sources          — list of SourceCitation
                result.check_result     — attribution + faithfulness flags
                result.total_time       — seconds
                result.retries          — int
        """
        generator = self._get_generator()
        return generator.query(question, status_callback=status_callback)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: lazy generator init
    # ─────────────────────────────────────────────────────────────────────────

    def _get_generator(self):
        """Initialise RAGGenerator on first call (lazy)."""
        if self._generator is not None:
            return self._generator

        from src.generation.rag_generator import RAGGenerator

        paths = self.cfg.get("paths", {})

        # Build config dict expected by RAGGenerator
        gen_cfg = {
            "models":    self.cfg.get("models", {}),
            "retrieval": {
                **self.cfg.get("retrieval", {}),
                "npy_dir":           paths.get("multivectors", "data/indices/multivectors"),
                "chroma_persist_dir":paths.get("chroma_index",  "data/indices/chroma_index"),
            },
            "paths": paths,
        }

        self._generator = RAGGenerator(gen_cfg)
        return self._generator
