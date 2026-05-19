"""
Scientific Multimodal RAG — Source Package

A vision + text hybrid retrieval-augmented generation system
for scientific research papers.

Modules:
    data: PDF download, parsing, and preprocessing
    models: VLM loading and generation (Qwen2-VL-2B)
    embeddings: ColPali (vision) and SciNCL (text) embedders
    retrieval: MaxSim, ChromaDB, and score fusion retrievers
    context: Prompt assembly and context building
    generation: Full RAG pipeline and self-check
    utils: Shared utilities (logging, device, metrics, etc.)
"""

__version__ = "0.1.0"
__author__ = "Vineet"
