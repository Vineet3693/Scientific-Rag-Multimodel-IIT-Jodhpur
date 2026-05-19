"""
Generation Package — RAG Generation and Answer Verification.

Re-exports the core classes so that downstream modules can import
from the package root::

    from src.generation import RAGResult, RAGGenerator, CheckResult, SelfChecker
"""

from src.generation.self_check import CheckResult, SelfChecker
from src.generation.rag_generator import RAGResult, RAGGenerator

__all__ = [
    "RAGResult",
    "RAGGenerator",
    "CheckResult",
    "SelfChecker",
]
