"""
Offline Pipeline — Download, Parse, Embed, and Index Scientific Papers.

Implements the complete offline indexing pipeline for the Scientific
Multimodal RAG system.  This pipeline runs once (or incrementally via
checkpointing) to build the retrieval index from arXiv PDFs.

Pipeline Steps
--------------
1. **Load configs** — Read data_config, model_config, pipeline_config
   from YAML files.
2. **Download PDFs** — Fetch arXiv papers via the ArxivDataset helper.
3. **Parse PDFs** — Convert each PDF into page images + markdown text
   using the DualPDFParser helper.
4. **ColPali embed pages** — Produce multi-vector embeddings for every
   page image and save as ``.npy`` files, then **UNLOAD** the model.
5. **SciNCL embed text** — Produce dense 768-dim vectors for each text
   chunk and store in ChromaDB, then **UNLOAD** the model.
6. **Save metadata** — Write ``doc_mapping.json`` and
   ``page_metadata.json`` for cross-referencing.
7. **Checkpoint** — Update ``checkpoint.json`` after each PDF so the
   pipeline can be safely resumed.
8. **Print summary** — Display a human-readable report of results.

Timing Estimates (Kaggle P100)
------------------------------
* Download 10 PDFs: ~2-3 min
* Parse 10 PDFs (~100 pages): ~5-8 min
* ColPali embed ~100 pages: ~10-15 min
* SciNCL embed ~100 pages: ~3-5 min
* **Total for 10 PDFs: ~30-40 min**

Example:
    >>> from pipelines.offline_pipeline import OfflinePipeline
    >>> pipeline = OfflinePipeline(config_path="configs/pipeline_config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Processed {result['papers_processed']} papers in {result['total_time']:.1f}s")
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.utils.config_loader import load_config, resolve_paths, validate_config
from src.utils.device import free_vram, get_device, get_vram_usage, print_gpu_info
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ArxivDataset — arXiv paper downloader
# ---------------------------------------------------------------------------

class ArxivDataset:
    """Download arXiv papers matching a search query.

    Uses the ``arxiv`` Python package to search for papers and download
    their PDFs.  Papers are saved to a specified output directory with
    filenames based on their arXiv IDs.

    Args:
        query: Search query string, e.g. ``"vision transformer"``.
        category: arXiv category filter, e.g. ``"cs.CV"``.
        max_results: Maximum number of search results to return.
        keep_best: Number of papers to actually download (the top-N
            from the search results sorted by relevance).
        date_range: Tuple of ``(start_year, end_year)`` for filtering.
        output_dir: Directory where PDFs will be saved.

    Example:
        >>> dataset = ArxivDataset(
        ...     query="vision transformer",
        ...     category="cs.CV",
        ...     max_results=20,
        ...     keep_best=10,
        ...     output_dir="data/raw/",
        ... )
        >>> results = dataset.download()
        >>> print(f"Downloaded {len(results)} papers")
    """

    def __init__(
        self,
        query: str = "vision transformer",
        category: str = "cs.CV",
        max_results: int = 20,
        keep_best: int = 10,
        date_range: tuple = (2021, 2024),
        output_dir: str = "data/raw/",
    ) -> None:
        self.query = query
        self.category = category
        self.max_results = max_results
        self.keep_best = keep_best
        self.date_range = date_range
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ArxivDataset initialised — query='%s', category=%s, "
            "max_results=%d, keep_best=%d, date_range=%s",
            self.query, self.category, self.max_results,
            self.keep_best, self.date_range,
        )

    def download(self) -> List[Dict[str, Any]]:
        """Search arXiv and download PDFs for the top papers.

        Returns:
            A list of dictionaries, one per downloaded paper, with keys:
                - ``arxiv_id``: The arXiv identifier (e.g. ``"2305.12345"``)
                - ``title``: Paper title
                - ``pdf_path``: Local filesystem path to the downloaded PDF
                - ``url``: arXiv abstract URL
                - ``status``: ``"success"`` or ``"failed"``
        """
        try:
            import arxiv
        except ImportError:
            raise ImportError(
                "The 'arxiv' package is required for downloading papers.  "
                "Install it with: pip install arxiv"
            )

        logger.info(
            "Searching arXiv for '%s' in category %s (max %d results)…",
            self.query, self.category, self.max_results,
        )

        # Build the search query with category filter
        search_query = f"all:{self.query}"
        if self.category:
            search_query = f"cat:{self.category} AND all:{self.query}"

        search = arxiv.Search(
            query=search_query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        # Collect results
        papers: List[Dict[str, Any]] = []
        for result in search.results():
            arxiv_id = result.entry_id.split("/")[-1]
            published_year = result.published.year

            # Filter by date range
            if published_year < self.date_range[0] or published_year > self.date_range[1]:
                logger.debug(
                    "Skipping %s (year %d outside range %s)",
                    arxiv_id, published_year, self.date_range,
                )
                continue

            papers.append({
                "arxiv_id": arxiv_id,
                "title": result.title,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "published": str(result.published),
                "authors": ", ".join(a.name for a in result.authors[:3]),
                "summary": result.summary[:200],
            })

        # Keep only the best N
        papers = papers[:self.keep_best]
        logger.info("Found %d papers matching criteria.", len(papers))

        # Download PDFs
        results: List[Dict[str, Any]] = []
        for i, paper in enumerate(papers):
            arxiv_id = paper["arxiv_id"]

            # Create a clean, filesystem-friendly version of the paper title
            import re
            title_clean = re.sub(r'[^a-zA-Z0-9\s_-]', '', paper["title"])
            title_slug = re.sub(r'\s+', '_', title_clean).strip('_')[:50]

            pdf_filename = f"{arxiv_id.replace('/', '_')}_{title_slug}.pdf"
            pdf_path = self.output_dir / pdf_filename

            # Check if old format filename exists and rename it
            old_filename = f"{arxiv_id.replace('/', '_')}.pdf"
            old_pdf_path = self.output_dir / old_filename

            if old_pdf_path.exists() and not pdf_path.exists():
                try:
                    old_pdf_path.rename(pdf_path)
                    logger.info("Renamed existing PDF: %s -> %s", old_filename, pdf_filename)
                except Exception as rename_exc:
                    logger.warning("Could not rename existing PDF: %s", rename_exc)

            if pdf_path.exists():
                logger.info(
                    "[%d/%d] %s already downloaded — skipping.",
                    i + 1, len(papers), arxiv_id,
                )
                paper["pdf_path"] = str(pdf_path)
                paper["status"] = "success"
                results.append(paper)
                continue

            logger.info(
                "[%d/%d] Downloading %s…",
                i + 1, len(papers), arxiv_id,
            )

            try:
                # Add a sleep before fetching to avoid rate limits
                time.sleep(3)

                # Direct download with urllib, bypassing extra API search call
                # which causes SSL errors on some systems and triggers rate limits
                import urllib.request
                import ssl

                ssl_context = ssl._create_unverified_context()
                req = urllib.request.Request(
                    paper["pdf_url"],
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAG-Bot/1.0"}
                )

                with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                    with open(pdf_path, "wb") as out_file:
                        out_file.write(response.read())

                paper["pdf_path"] = str(pdf_path)
                paper["status"] = "success"
                logger.info("Downloaded: %s → %s", arxiv_id, pdf_path)

            except Exception as exc:
                logger.warning("Direct download failed for %s: %s. Attempting fallback...", arxiv_id, exc)
                try:
                    # Use arxiv package as fallback
                    search_single = arxiv.Search(id_list=[arxiv_id])
                    for res in search_single.results():
                        res.download_pdf(
                            dirpath=str(self.output_dir),
                            filename=pdf_filename,
                        )
                        break

                    paper["pdf_path"] = str(pdf_path)
                    paper["status"] = "success"
                    logger.info("Downloaded via fallback: %s → %s", arxiv_id, pdf_path)

                except Exception as fallback_exc:
                    logger.error("Failed to download %s (all methods failed): %s", arxiv_id, fallback_exc)
                    paper["pdf_path"] = ""
                    paper["status"] = "failed"

            results.append(paper)

        succeeded = sum(1 for r in results if r["status"] == "success")
        logger.info(
            "Download complete — %d/%d papers downloaded successfully.",
            succeeded, len(results),
        )

        return results


# ---------------------------------------------------------------------------
# DualPDFParser — PDF → images + markdown
# ---------------------------------------------------------------------------

class DualPDFParser:
    """Parse PDFs into page images and markdown text.

    Uses ``pdf2image`` for high-resolution page rendering and
    ``marker-pdf`` (or a fallback) for markdown extraction.  Each
    page is saved as a separate PNG image, and the full text is
    extracted as markdown.

    Args:
        output_pages_dir: Directory for page images.
        output_markdown_dir: Directory for markdown files.
        dpi: Resolution for page rendering (default 200).
        image_format: Output image format (default ``"PNG"``).

    Example:
        >>> parser = DualPDFParser(
        ...     output_pages_dir="data/parsed/pages/",
        ...     output_markdown_dir="data/parsed/markdown/",
        ... )
        >>> pages, text = parser.parse("data/raw/2305.12345.pdf")
        >>> print(f"Parsed {len(pages)} pages")
    """

    def __init__(
        self,
        output_pages_dir: str = "data/parsed/pages/",
        output_markdown_dir: str = "data/parsed/markdown/",
        dpi: int = 200,
        image_format: str = "PNG",
    ) -> None:
        self.output_pages_dir = Path(output_pages_dir)
        self.output_markdown_dir = Path(output_markdown_dir)
        self.dpi = dpi
        self.image_format = image_format

        self.output_pages_dir.mkdir(parents=True, exist_ok=True)
        self.output_markdown_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DualPDFParser initialised — pages_dir=%s, markdown_dir=%s, "
            "dpi=%d, format=%s",
            self.output_pages_dir, self.output_markdown_dir,
            self.dpi, self.image_format,
        )

    def parse(
        self, pdf_path: str, doc_id: str = ""
    ) -> Dict[str, Any]:
        """Parse a single PDF into page images and markdown text.

        Args:
            pdf_path: Path to the input PDF file.
            doc_id: Document identifier used for naming output files.
                Defaults to the PDF filename stem.

        Returns:
            A dictionary with keys:
                - ``doc_id``: The document identifier.
                - ``num_pages``: Total number of pages.
                - ``page_images``: List of paths to rendered page images.
                - ``markdown_path``: Path to the extracted markdown file.
                - ``page_texts``: List of text strings, one per page.
                - ``status``: ``"success"`` or ``"failed"``.
        """
        pdf_path = Path(pdf_path)
        if not doc_id:
            doc_id = pdf_path.stem

        doc_pages_dir = self.output_pages_dir / doc_id
        doc_pages_dir.mkdir(parents=True, exist_ok=True)

        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "num_pages": 0,
            "page_images": [],
            "markdown_path": "",
            "page_texts": [],
            "status": "failed",
        }

        # Step 1: Render page images with pdf2image
        try:
            from pdf2image import convert_from_path

            logger.info("Rendering pages for %s (dpi=%d)…", doc_id, self.dpi)
            images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt=self.image_format.lower(),
            )

            page_image_paths: List[str] = []
            for i, img in enumerate(images):
                page_num = i + 1
                image_filename = f"{doc_id}_page_{page_num}.{self.image_format.lower()}"
                image_path = doc_pages_dir / image_filename
                img.save(str(image_path), self.image_format)
                page_image_paths.append(str(image_path))

            result["num_pages"] = len(images)
            result["page_images"] = page_image_paths
            logger.info(
                "Rendered %d pages for %s.", len(images), doc_id,
            )

        except ImportError:
            logger.warning(
                "pdf2image not available — attempting PyMuPDF fallback."
            )
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(str(pdf_path))
                page_image_paths: List[str] = []
                for i in range(len(doc)):
                    page = doc[i]
                    pix = page.get_pixmap(dpi=self.dpi)
                    page_num = i + 1
                    image_filename = f"{doc_id}_page_{page_num}.png"
                    image_path = doc_pages_dir / image_filename
                    pix.save(str(image_path))
                    page_image_paths.append(str(image_path))

                result["num_pages"] = len(doc)
                result["page_images"] = page_image_paths
                doc.close()
                logger.info(
                    "Rendered %d pages via PyMuPDF for %s.",
                    result["num_pages"], doc_id,
                )

            except ImportError:
                logger.error(
                    "Neither pdf2image nor PyMuPDF available — "
                    "cannot render page images."
                )
                return result

        except Exception as exc:
            logger.error("Failed to render pages for %s: %s", doc_id, exc)
            return result

        # Step 2: Extract text / markdown
        try:
            markdown_text = self._extract_markdown(pdf_path, doc_id)
            markdown_path = self.output_markdown_dir / f"{doc_id}.md"

            with open(str(markdown_path), "w", encoding="utf-8") as f:
                f.write(markdown_text)

            result["markdown_path"] = str(markdown_path)

            # Split markdown into per-page text chunks
            result["page_texts"] = self._split_text_by_pages(
                markdown_text, result["num_pages"]
            )

            logger.info(
                "Extracted text for %s — %d chars, saved to %s",
                doc_id, len(markdown_text), markdown_path,
            )

        except Exception as exc:
            logger.error(
                "Failed to extract text for %s: %s — using empty text.",
                doc_id, exc,
            )
            result["page_texts"] = [""] * result["num_pages"]

        result["status"] = "success"
        return result

    def _extract_markdown(self, pdf_path: Path, doc_id: str) -> str:
        """Extract markdown text from a PDF.

        Tries marker-pdf first, then falls back to PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.
            doc_id: Document identifier for logging.

        Returns:
            Extracted markdown text string.
        """
        # Try marker-pdf
        try:
            from marker.converters.pdf import PdfConverter
            from marker.config.parser import ConfigParser
            from marker.output import text_from_rendered

            converter = PdfConverter(
                config=ConfigParser({}).generate_config()
            )
            rendered = converter(str(pdf_path))
            text, _, _ = text_from_rendered(rendered)
            logger.info("Extracted markdown via marker-pdf for %s.", doc_id)
            return text

        except ImportError:
            logger.debug("marker-pdf not available — trying PyMuPDF.")
        except Exception as exc:
            logger.warning(
                "marker-pdf failed for %s: %s — trying PyMuPDF.",
                doc_id, exc,
            )

        # Fallback: PyMuPDF
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            pages_text: List[str] = []
            for page in doc:
                pages_text.append(page.get_text())
            doc.close()

            text = "\n\n---\n\n".join(pages_text)
            logger.info(
                "Extracted text via PyMuPDF for %s — %d chars.",
                doc_id, len(text),
            )
            return text

        except ImportError:
            logger.warning(
                "Neither marker-pdf nor PyMuPDF available — "
                "returning empty text."
            )
            return ""

    @staticmethod
    def _split_text_by_pages(text: str, num_pages: int) -> List[str]:
        """Split extracted text into approximate per-page chunks.

        If the text contains ``---`` page separators (from PyMuPDF
        fallback), splits on those.  Otherwise, distributes text evenly
        across pages.

        Args:
            text: Full extracted text.
            num_pages: Number of pages in the document.

        Returns:
            List of text strings, one per page.
        """
        if not text or num_pages == 0:
            return []

        # Try splitting on page separator
        if "---" in text:
            parts = text.split("---")
            # Filter out empty parts
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) == num_pages:
                return parts

        # Even distribution fallback
        if num_pages <= 0:
            return [text]

        chunk_size = max(1, len(text) // num_pages)
        chunks: List[str] = []
        for i in range(num_pages):
            start = i * chunk_size
            end = start + chunk_size if i < num_pages - 1 else len(text)
            chunks.append(text[start:end].strip())

        return chunks


# ---------------------------------------------------------------------------
# OfflinePipeline
# ---------------------------------------------------------------------------

class OfflinePipeline:
    """Complete offline indexing pipeline for Scientific Multimodal RAG.

    Orchestrates PDF download, parsing, embedding, and index building.
    Supports checkpoint-based resume so that the pipeline can be safely
    interrupted and continued from the last completed PDF.

    Timing Estimate (Kaggle P100, 10 PDFs, ~100 pages total):
        - Download: 2-3 min
        - Parse: 5-8 min
        - ColPali embed: 10-15 min
        - SciNCL embed: 3-5 min
        - **Total: ~30-40 min**

    Args:
        config_path: Path to the pipeline configuration YAML file.
            Defaults to ``"configs/pipeline_config.yaml"``.

    Example:
        >>> pipeline = OfflinePipeline()
        >>> result = pipeline.run()
        >>> print(f"Papers: {result['papers_processed']}, "
        ...       f"Pages: {result['pages_embedded']}, "
        ...       f"Time: {result['total_time']:.1f}s")
    """

    def __init__(
        self, config_path: str = "configs/pipeline_config.yaml"
    ) -> None:
        self.config_path = config_path
        self._load_all_configs()

        logger.info("OfflinePipeline initialised.")

    # -----------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------

    def _load_all_configs(self) -> None:
        """Load and validate all YAML configuration files.

        Loads data_config, model_config, and pipeline_config, then
        resolves paths for the current environment (Kaggle vs. local).
        """
        logger.info("Loading configurations from: %s", self.config_path)

        # Load individual configs
        self.data_config = load_config("data_config")
        self.model_config = load_config("model_config")
        self.pipeline_config = load_config("pipeline_config")

        # Also load retrieval config for embedding paths
        self.retrieval_config = load_config("retrieval_config")

        # Merge into a single config dict for downstream components
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
        self._data_cfg = self.merged_config.get("data", {})
        self._models_cfg = self.merged_config.get("models", {})
        self._retrieval_cfg = self.merged_config.get("retrieval", {})

        logger.info("All configurations loaded and paths resolved.")

    # -----------------------------------------------------------------
    # Checkpoint management
    # -----------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        """Return the path to the checkpoint file."""
        base = self._paths.get("raw_pdfs", "data/raw/")
        return Path(base).parent / "indices" / "checkpoint.json"

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load the checkpoint file if it exists.

        Returns:
            Checkpoint dictionary, or an empty dict with default values
            if no checkpoint exists.
        """
        ckpt_path = self._checkpoint_path()
        if ckpt_path.exists():
            with open(str(ckpt_path), "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            logger.info(
                "Loaded checkpoint — %d papers already processed.",
                len(checkpoint.get("completed_papers", [])),
            )
            return checkpoint

        default: Dict[str, Any] = {
            "completed_papers": [],
            "failed_papers": [],
            "pages_embedded": 0,
            "last_updated": "",
        }
        return default

    def _save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save the checkpoint file.

        Args:
            checkpoint: The checkpoint dictionary to persist.
        """
        ckpt_path = self._checkpoint_path()
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(str(ckpt_path), "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        logger.debug("Checkpoint saved: %s", ckpt_path)

    # -----------------------------------------------------------------
    # Metadata management
    # -----------------------------------------------------------------

    def _save_doc_mapping(
        self, doc_mapping: Dict[str, Any]
    ) -> None:
        """Save the document mapping JSON file.

        Args:
            doc_mapping: Dictionary mapping arXiv IDs to paper metadata.
        """
        mapping_path = self._paths.get(
            "doc_mapping", "data/indices/doc_mapping.json"
        )
        Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)

        with open(str(mapping_path), "w", encoding="utf-8") as f:
            json.dump(doc_mapping, f, indent=2, ensure_ascii=False)

        logger.info("Saved doc_mapping.json — %d entries.", len(doc_mapping))

    def _save_page_metadata(
        self, page_metadata: Dict[str, Any]
    ) -> None:
        """Save the page metadata JSON file.

        Args:
            page_metadata: Dictionary mapping page keys to page metadata.
        """
        meta_path = self._paths.get(
            "page_metadata", "data/indices/page_metadata.json"
        )
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(page_metadata, f, indent=2, ensure_ascii=False)

        logger.info("Saved page_metadata.json — %d entries.", len(page_metadata))

    # -----------------------------------------------------------------
    # run — main entry point
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the complete offline pipeline.

        Runs all pipeline steps: download → parse → embed → index →
        save metadata → checkpoint.

        **Timing estimate**: 30-40 minutes for 10 PDFs (~100 pages)
        on a Kaggle P100 GPU.

        Returns:
            A summary dictionary with keys:
                - ``papers_processed``: Number of successfully processed
                  papers.
                - ``pages_embedded``: Total number of pages embedded.
                - ``failed_downloads``: Number of papers that failed to
                  download.
                - ``failed_parses``: Number of papers that failed to
                  parse.
                - ``colpali_time``: Time spent on ColPali embedding.
                - ``scincl_time``: Time spent on SciNCL embedding.
                - ``total_time``: Total wall-clock time in seconds.
        """
        t_start = time.time()

        print_gpu_info()

        # Initialize result counters
        summary: Dict[str, Any] = {
            "papers_processed": 0,
            "pages_embedded": 0,
            "failed_downloads": 0,
            "failed_parses": 0,
            "colpali_time": 0.0,
            "scincl_time": 0.0,
            "total_time": 0.0,
        }

        # Load checkpoint
        checkpoint = self._load_checkpoint()
        completed = set(checkpoint.get("completed_papers", []))

        # ─── Step 1: Download arXiv PDFs ───
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading arXiv PDFs")
        logger.info("=" * 60)

        dataset = ArxivDataset(
            query=self._data_cfg.get("query", "vision transformer"),
            category=self._data_cfg.get("category", "cs.CV"),
            max_results=self._data_cfg.get("max_results", 20),
            keep_best=self._data_cfg.get("keep_best", 10),
            date_range=tuple(self._data_cfg.get("date_range", [2021, 2024])),
            output_dir=self._paths.get("raw_pdfs", "data/raw/"),
        )

        download_results = dataset.download()
        summary["failed_downloads"] = sum(
            1 for r in download_results if r.get("status") != "success"
        )

        # ─── Step 2: Parse PDFs ───
        logger.info("=" * 60)
        logger.info("STEP 2: Parsing PDFs into images + markdown")
        logger.info("=" * 60)

        parser = DualPDFParser(
            output_pages_dir=self._paths.get("parsed_pages", "data/parsed/pages/"),
            output_markdown_dir=self._paths.get("parsed_markdown", "data/parsed/markdown/"),
            dpi=self.data_config.get("parsing", {}).get("dpi", 200),
            image_format=self.data_config.get("parsing", {}).get("image_format", "PNG"),
        )

        parsed_papers: List[Dict[str, Any]] = []
        doc_mapping: Dict[str, Any] = {}
        page_metadata: Dict[str, Any] = {}

        for dl in download_results:
            if dl.get("status") != "success":
                continue

            arxiv_id = dl["arxiv_id"]

            # Skip if already in checkpoint
            if arxiv_id in completed:
                logger.info(
                    "Skipping %s — already processed (from checkpoint).",
                    arxiv_id,
                )
                # Still add to doc_mapping from existing data
                doc_mapping[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "title": dl.get("title", arxiv_id),
                    "url": dl.get("url", ""),
                    "pdf_path": dl.get("pdf_path", ""),
                    "status": "cached",
                }
                continue

            logger.info("Parsing %s…", arxiv_id)

            parse_result = parser.parse(
                pdf_path=dl["pdf_path"],
                doc_id=arxiv_id,
            )

            if parse_result["status"] != "success":
                logger.error("Failed to parse %s — skipping.", arxiv_id)
                summary["failed_parses"] += 1
                checkpoint["failed_papers"].append(arxiv_id)
                continue

            parsed_papers.append(parse_result)

            # Build doc_mapping entry
            doc_mapping[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": dl.get("title", arxiv_id),
                "url": dl.get("url", ""),
                "pdf_path": dl.get("pdf_path", ""),
                "num_pages": parse_result["num_pages"],
                "page_images": parse_result["page_images"],
                "markdown_path": parse_result.get("markdown_path", ""),
                "authors": dl.get("authors", ""),
                "published": dl.get("published", ""),
                "status": "success",
            }

            # Build page_metadata entries
            for i, img_path in enumerate(parse_result["page_images"]):
                page_num = i + 1
                page_key = f"{arxiv_id}_page_{page_num}"
                page_text = (
                    parse_result["page_texts"][i]
                    if i < len(parse_result["page_texts"])
                    else ""
                )
                page_metadata[page_key] = {
                    "doc_id": arxiv_id,
                    "page_num": page_num,
                    "image_path": img_path,
                    "text": page_text,
                    "paper_title": dl.get("title", arxiv_id),
                }

        # ─── Step 3: ColPali embed pages ───
        logger.info("=" * 60)
        logger.info("STEP 3: ColPali embedding (multi-vector page embeddings)")
        logger.info("=" * 60)

        colpali_start = time.time()

        try:
            from src.embeddings.colpali_embedder import ColPaliEmbedder

            colpali_cfg = self._models_cfg.get("colpali", {})
            colpali = ColPaliEmbedder(
                model_name=colpali_cfg.get("model_name", "vidore/colpali-v1.2"),
                device=colpali_cfg.get("device", "cuda"),
                torch_dtype=colpali_cfg.get("torch_dtype", "float16"),
                max_pages_per_batch=colpali_cfg.get("max_pages_per_batch", 4),
            )

            colpali.load()

            npy_dir = self._paths.get(
                "multivectors",
                self._retrieval_cfg.get("npy_dir", "data/indices/multivectors/"),
            )
            Path(npy_dir).mkdir(parents=True, exist_ok=True)

            from PIL import Image

            for paper in parsed_papers:
                arxiv_id = paper["doc_id"]
                logger.info(
                    "ColPali embedding %s (%d pages)…",
                    arxiv_id, paper["num_pages"],
                )

                # Load page images
                page_images: List[Image.Image] = []
                for img_path in paper["page_images"]:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        page_images.append(img)
                    except Exception as exc:
                        logger.error(
                            "Failed to open image %s: %s", img_path, exc
                        )

                if not page_images:
                    logger.warning(
                        "No images loaded for %s — skipping.", arxiv_id
                    )
                    continue

                # Embed in batches
                try:
                    outputs = colpali.embed_batch(page_images, item_type="image")
                except Exception as exc:
                    logger.error(
                        "ColPali batch embedding failed for %s: %s — "
                        "trying one by one.",
                        arxiv_id, exc,
                    )
                    outputs = []
                    for img in page_images:
                        try:
                            out = colpali.embed_image(img)
                            outputs.append(out)
                        except Exception as e2:
                            logger.error("Single image embed failed: %s", e2)

                # Save each page embedding
                for i, output in enumerate(outputs):
                    page_num = i + 1
                    npy_filename = f"{arxiv_id}_page_{page_num}.npy"
                    npy_path = Path(npy_dir) / npy_filename
                    output.doc_id = arxiv_id
                    output.page_num = page_num
                    colpali.save_vectors(output, str(npy_path))
                    summary["pages_embedded"] += 1

                logger.info(
                    "ColPali embedded %d pages for %s.",
                    len(outputs), arxiv_id,
                )

                # Update checkpoint after each PDF
                checkpoint["completed_papers"].append(arxiv_id)
                checkpoint["pages_embedded"] = summary["pages_embedded"]
                self._save_checkpoint(checkpoint)

            # Unload ColPali
            colpali.unload()
            logger.info("ColPali model unloaded.")

        except Exception as exc:
            logger.error("ColPali embedding step failed: %s", exc)
        finally:
            free_vram()

        colpali_time = time.time() - colpali_start
        summary["colpali_time"] = colpali_time

        # ─── Step 4: SciNCL embed text ───
        logger.info("=" * 60)
        logger.info("STEP 4: SciNCL embedding (dense text embeddings)")
        logger.info("=" * 60)

        scincl_start = time.time()

        try:
            from src.embeddings.scincl_embedder import SciNCLEmbedder

            scincl_cfg = self._models_cfg.get("scincl", {})
            scincl = SciNCLEmbedder(
                model_name=scincl_cfg.get("model_name", "malteos/scincl"),
                device=scincl_cfg.get("device", "cuda"),
                max_length=scincl_cfg.get("max_length", 512),
            )

            scincl.load()

            chroma_dir = self._paths.get(
                "chroma_index",
                self._retrieval_cfg.get(
                    "chroma_persist_dir", "data/indices/chroma_index/"
                ),
            )
            collection_name = self._retrieval_cfg.get(
                "chroma_collection", "sci_text"
            )

            for page_key, meta in page_metadata.items():
                text = meta.get("text", "")
                if not text or not text.strip():
                    logger.debug(
                        "Skipping empty text for %s.", page_key
                    )
                    continue

                try:
                    output = scincl.embed_text(text)
                    output.doc_id = meta["doc_id"]
                    output.page_num = meta["page_num"]
                    output.metadata["paper_title"] = meta.get(
                        "paper_title", meta["doc_id"]
                    )
                    output.metadata["text"] = text[:500]  # Store snippet

                    scincl.save_to_chromadb(
                        output,
                        collection_name=collection_name,
                        persist_dir=str(chroma_dir),
                    )

                except Exception as exc:
                    logger.error(
                        "Failed to embed %s: %s", page_key, exc
                    )

            # Unload SciNCL
            scincl.unload()
            logger.info("SciNCL model unloaded.")

        except Exception as exc:
            logger.error("SciNCL embedding step failed: %s", exc)
        finally:
            free_vram()

        scincl_time = time.time() - scincl_start
        summary["scincl_time"] = scincl_time

        # ─── Step 5: Save metadata ───
        logger.info("=" * 60)
        logger.info("STEP 5: Saving doc_mapping and page_metadata")
        logger.info("=" * 60)

        self._save_doc_mapping(doc_mapping)
        self._save_page_metadata(page_metadata)

        # ─── Step 6: Final checkpoint ───
        checkpoint["pages_embedded"] = summary["pages_embedded"]
        self._save_checkpoint(checkpoint)

        # ─── Summary ───
        total_time = time.time() - t_start
        summary["total_time"] = total_time
        summary["papers_processed"] = len(
            checkpoint.get("completed_papers", [])
        )

        self._print_summary(summary)

        return summary

    # -----------------------------------------------------------------
    # resume
    # -----------------------------------------------------------------

    def resume(self) -> Dict[str, Any]:
        """Resume the offline pipeline from the last checkpoint.

        Reads the checkpoint file to determine which papers have already
        been processed, then runs the pipeline again, skipping papers
        that appear in the ``completed_papers`` list.

        This is useful if the pipeline was interrupted (e.g., Kaggle
        session timeout) and needs to be continued.

        **Timing**: Depends on how many papers remain.  If only a few
        papers remain, expect 5-10 minutes.

        Returns:
            Same summary dictionary as :meth:`run`.
        """
        checkpoint = self._load_checkpoint()
        already_done = len(checkpoint.get("completed_papers", []))

        logger.info(
            "Resuming offline pipeline — %d papers already completed.",
            already_done,
        )

        # Re-run the full pipeline; the run() method checks the
        # checkpoint internally and skips completed papers.
        return self.run()

    # -----------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------

    @staticmethod
    def _print_summary(summary: Dict[str, Any]) -> None:
        """Print a human-readable summary report.

        Args:
            summary: The result dictionary from :meth:`run`.
        """
        print("\n" + "=" * 60)
        print("  OFFLINE PIPELINE — SUMMARY REPORT")
        print("=" * 60)
        print(f"  Papers processed : {summary['papers_processed']}")
        print(f"  Pages embedded   : {summary['pages_embedded']}")
        print(f"  Failed downloads : {summary['failed_downloads']}")
        print(f"  Failed parses    : {summary['failed_parses']}")
        print(f"  ColPali time     : {summary['colpali_time']:.1f}s")
        print(f"  SciNCL time      : {summary['scincl_time']:.1f}s")
        print(f"  Total time       : {summary['total_time']:.1f}s")
        print("=" * 60 + "\n")

        logger.info("Offline pipeline complete — summary printed.")
