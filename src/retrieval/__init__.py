"""
Retrieval Package — Vision & Text Retrieval Backends with Score Fusion.

Re-exports the core classes so that downstream modules can import
from the package root::

    from src.retrieval import (
        BaseRetriever, RetrievedDocument, SourceCitation,
        ColPaliRetriever, TextRetriever, FusionRetriever,
    )
"""

from src.retrieval.base_retriever import (
    BaseRetriever,
    RetrievedDocument,
    SourceCitation,
)
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.fusion_retriever import FusionRetriever
from src.retrieval.text_retriever import TextRetriever

__all__ = [
    "BaseRetriever",
    "RetrievedDocument",
    "SourceCitation",
    "ColPaliRetriever",
    "TextRetriever",
    "FusionRetriever",
]
