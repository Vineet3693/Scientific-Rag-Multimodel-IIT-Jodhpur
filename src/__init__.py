"""
Scientific Multimodal RAG — Source Package
==========================================
A vision + text hybrid retrieval-augmented generation system
for scientific research papers.
"""

__version__ = "0.1.0"
__author__ = "Vineet"

# Expose key modular interfaces
from src.utils.helpers import clean_vram, ensure_directories, extract_zip_archive, create_zip_archive
from src.models.loader import load_colpali, load_scincl, load_qwen2vl
from src.context.pdf_parser import PDFParser
from src.embeddings.colpali_embedder import ColPaliEmbedder
from src.embeddings.scincl_embedder import SciNCLEmbedder
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.text_retriever import TextRetriever
from src.retrieval.fusion_retriever import FusionRetriever
from src.generation.self_check import DomainGuard
from src.generation.rag_generator import RAGGenerator
