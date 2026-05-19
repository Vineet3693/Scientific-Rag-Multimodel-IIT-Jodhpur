"""
setup.py — Package installer for Scientific Multimodal RAG

Usage:
    pip install -e .

This makes the src/ directory importable as a module,
so you can do: from src.data import ArxivDataset
"""

from setuptools import setup, find_packages

setup(
    name="scientific-multimodal-rag",
    version="0.1.0",
    description="Vision + Text Hybrid RAG for Scientific Papers",
    author="Vineet",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "colpali-engine>=0.3.0",
        "chromadb>=0.4.22",
        "Pillow>=9.0.0",
        "pyyaml>=6.0.1",
    ],
)
