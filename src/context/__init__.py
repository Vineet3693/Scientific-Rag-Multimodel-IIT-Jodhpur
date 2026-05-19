"""
Context Package — Prompt Assembly and Template Management.

Re-exports the core classes so that downstream modules can import
from the package root::

    from src.context import ContextObject, ContextBuilder, PromptTemplates
"""

from src.context.context_builder import ContextBuilder, ContextObject
from src.context.prompt_templates import PromptTemplates

__all__ = [
    "ContextObject",
    "ContextBuilder",
    "PromptTemplates",
]
