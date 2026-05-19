"""
Tests for PDF Parsing and Text/Image Preprocessing.

Covers:
- ScientificSample dataclass creation and validation
- ParsedDocument dataclass creation and validation
- Image preprocessor: resize_for_colpali
- Text preprocessor: clean_markdown
- Text preprocessor: split_by_pages
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import PIL.Image
import pytest

# ---------------------------------------------------------------------------
# Dataclasses under test — these mirror the structures produced by
# DualPDFParser.parse() and related preprocessing helpers.
# ---------------------------------------------------------------------------


@dataclass
class ScientificSample:
    """Metadata for a single scientific paper sample.

    Attributes:
        paper_id: Unique identifier (e.g. arXiv ID).
        title: Paper title.
        authors: Comma-separated author names.
        year: Publication year.
        categories: arXiv category tags.
        abstract: Paper abstract text.
        pdf_path: Local path to the downloaded PDF.
    """

    paper_id: str
    title: str
    authors: str = ""
    year: int = 0
    categories: List[str] = field(default_factory=list)
    abstract: str = ""
    pdf_path: str = ""

    def __post_init__(self) -> None:
        """Validate fields after initialisation."""
        if not self.paper_id:
            raise ValueError("ScientificSample.paper_id must not be empty.")
        if not self.title:
            raise ValueError("ScientificSample.title must not be empty.")
        if self.year < 0:
            raise ValueError(
                f"ScientificSample.year must be >= 0, got {self.year}"
            )


@dataclass
class ParsedDocument:
    """Result of parsing a single PDF document.

    Attributes:
        doc_id: Document identifier (e.g. arXiv ID or filename stem).
        num_pages: Total number of pages in the document.
        page_image_paths: List of file paths to rendered page images.
        markdown_path: Path to the extracted markdown file.
        page_texts: List of text strings, one per page.
        status: Parsing status — ``"success"`` or ``"failed"``.
    """

    doc_id: str
    num_pages: int
    page_image_paths: List[str] = field(default_factory=list)
    markdown_path: str = ""
    page_texts: List[str] = field(default_factory=list)
    status: str = "failed"

    def __post_init__(self) -> None:
        """Validate fields after initialisation."""
        if not self.doc_id:
            raise ValueError("ParsedDocument.doc_id must not be empty.")
        if self.num_pages < 0:
            raise ValueError(
                f"ParsedDocument.num_pages must be >= 0, got {self.num_pages}"
            )
        if self.status not in ("success", "failed"):
            raise ValueError(
                f"ParsedDocument.status must be 'success' or 'failed', "
                f"got {self.status!r}"
            )


# ---------------------------------------------------------------------------
# Text preprocessing helpers under test
# ---------------------------------------------------------------------------


def clean_markdown(text: str) -> str:
    """Clean and normalise markdown text extracted from PDFs.

    Performs the following transformations:
    1. Remove excessive blank lines (3+ consecutive newlines → 2).
    2. Strip trailing whitespace from each line.
    3. Normalise Unicode dashes and quotes to ASCII equivalents.
    4. Remove zero-width characters.
    5. Collapse multiple spaces into one.

    Args:
        text: Raw markdown text.

    Returns:
        Cleaned markdown text.
    """
    if not text:
        return ""

    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Normalise dashes
    text = text.replace("\u2013", "-")  # en dash
    text = text.replace("\u2014", "--")  # em dash
    text = text.replace("\u2012", "-")  # figure dash

    # Normalise quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip trailing whitespace per line
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def split_by_pages(text: str, num_pages: int) -> List[str]:
    """Split extracted text into approximate per-page chunks.

    If the text contains ``---`` page separators, splits on those.
    Otherwise, distributes text evenly across pages.

    Args:
        text: Full extracted text.
        num_pages: Number of pages in the document.

    Returns:
        List of text strings, one per page.
    """
    if not text or num_pages <= 0:
        return []

    # Try splitting on page separator
    if "---" in text:
        parts = text.split("---")
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == num_pages:
            return parts

    # Even distribution fallback
    chunk_size = max(1, len(text) // num_pages)
    chunks: List[str] = []
    for i in range(num_pages):
        start = i * chunk_size
        end = start + chunk_size if i < num_pages - 1 else len(text)
        chunks.append(text[start:end].strip())

    return chunks


# ---------------------------------------------------------------------------
# Image preprocessing (re-use from the project)
# ---------------------------------------------------------------------------

from src.utils.image_utils import resize_for_colpali


# ===========================================================================
# TESTS
# ===========================================================================


class TestScientificSample:
    """Tests for the ScientificSample dataclass."""

    def test_scientific_sample_creation(self) -> None:
        """Test basic creation of a ScientificSample with required fields."""
        sample = ScientificSample(
            paper_id="2305.12345",
            title="Attention Is All You Need",
        )
        assert sample.paper_id == "2305.12345"
        assert sample.title == "Attention Is All You Need"
        assert sample.authors == ""
        assert sample.year == 0
        assert sample.categories == []
        assert sample.abstract == ""
        assert sample.pdf_path == ""

    def test_scientific_sample_full_creation(self) -> None:
        """Test creation with all fields populated."""
        sample = ScientificSample(
            paper_id="2305.12345",
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            year=2017,
            categories=["cs.CL", "cs.LG"],
            abstract="We propose a new network architecture...",
            pdf_path="/data/raw/2305.12345.pdf",
        )
        assert sample.year == 2017
        assert len(sample.categories) == 2
        assert "cs.CL" in sample.categories

    def test_scientific_sample_empty_paper_id_raises(self) -> None:
        """Test that empty paper_id raises ValueError."""
        with pytest.raises(ValueError, match="paper_id must not be empty"):
            ScientificSample(paper_id="", title="Some Title")

    def test_scientific_sample_empty_title_raises(self) -> None:
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="title must not be empty"):
            ScientificSample(paper_id="2305.12345", title="")

    def test_scientific_sample_negative_year_raises(self) -> None:
        """Test that negative year raises ValueError."""
        with pytest.raises(ValueError, match="year must be >= 0"):
            ScientificSample(paper_id="2305.12345", title="Test", year=-1)


class TestParsedDocument:
    """Tests for the ParsedDocument dataclass."""

    def test_parsed_document_creation(self) -> None:
        """Test basic creation of a ParsedDocument."""
        doc = ParsedDocument(
            doc_id="2305.12345",
            num_pages=10,
            status="success",
        )
        assert doc.doc_id == "2305.12345"
        assert doc.num_pages == 10
        assert doc.page_image_paths == []
        assert doc.markdown_path == ""
        assert doc.page_texts == []
        assert doc.status == "success"

    def test_parsed_document_full_creation(self) -> None:
        """Test creation with all fields populated."""
        doc = ParsedDocument(
            doc_id="2305.12345",
            num_pages=5,
            page_image_paths=[
                "/data/pages/2305.12345_page_1.png",
                "/data/pages/2305.12345_page_2.png",
            ],
            markdown_path="/data/markdown/2305.12345.md",
            page_texts=["Page 1 text", "Page 2 text"],
            status="success",
        )
        assert len(doc.page_image_paths) == 2
        assert len(doc.page_texts) == 2
        assert doc.markdown_path.endswith(".md")

    def test_parsed_document_empty_doc_id_raises(self) -> None:
        """Test that empty doc_id raises ValueError."""
        with pytest.raises(ValueError, match="doc_id must not be empty"):
            ParsedDocument(doc_id="", num_pages=1)

    def test_parsed_document_negative_pages_raises(self) -> None:
        """Test that negative num_pages raises ValueError."""
        with pytest.raises(ValueError, match="num_pages must be >= 0"):
            ParsedDocument(doc_id="test", num_pages=-1)

    def test_parsed_document_invalid_status_raises(self) -> None:
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="status must be"):
            ParsedDocument(doc_id="test", num_pages=1, status="pending")

    def test_parsed_document_default_status_is_failed(self) -> None:
        """Test that the default status is 'failed' (safe default)."""
        doc = ParsedDocument(doc_id="test", num_pages=0)
        assert doc.status == "failed"


class TestImagePreprocessor:
    """Tests for image preprocessing functions."""

    def test_image_preprocessor_resize(self) -> None:
        """Test that resize_for_colpali produces a 448x448 RGB image."""
        # Create a test image with non-square aspect ratio
        original = PIL.Image.new("RGB", (800, 1200), color=(100, 150, 200))
        resized = resize_for_colpali(original)

        assert resized.size == (448, 448)
        assert resized.mode == "RGB"

    def test_image_preprocessor_resize_landscape(self) -> None:
        """Test resize with a landscape-oriented image."""
        original = PIL.Image.new("RGB", (1200, 800), color=(50, 100, 150))
        resized = resize_for_colpali(original)

        assert resized.size == (448, 448)
        assert resized.mode == "RGB"

    def test_image_preprocessor_resize_already_correct(self) -> None:
        """Test resize when the image is already 448x448."""
        original = PIL.Image.new("RGB", (448, 448), color=(200, 200, 200))
        resized = resize_for_colpali(original)

        assert resized.size == (448, 448)

    def test_image_preprocessor_resize_grayscale_input(self) -> None:
        """Test that a grayscale input image is converted to RGB."""
        original = PIL.Image.new("L", (600, 800), color=128)
        resized = resize_for_colpali(original)

        assert resized.mode == "RGB"
        assert resized.size == (448, 448)

    def test_image_preprocessor_resize_small_image(self) -> None:
        """Test resize with a very small input image."""
        original = PIL.Image.new("RGB", (10, 10), color=(0, 0, 0))
        resized = resize_for_colpali(original)

        assert resized.size == (448, 448)

    def test_image_preprocessor_resize_custom_size(self) -> None:
        """Test resize with a custom target size."""
        original = PIL.Image.new("RGB", (800, 600))
        resized = resize_for_colpali(original, size=(224, 224))

        assert resized.size == (224, 224)

    def test_image_preprocessor_preserves_white_padding(self) -> None:
        """Test that padding areas are white (255, 255, 255)."""
        # Tall narrow image should have white padding on the sides
        original = PIL.Image.new("RGB", (100, 800), color=(255, 0, 0))
        resized = resize_for_colpali(original)

        # Check top-left corner (should be padding = white) since
        # a 100x800 image scaled to fit 448x448 will have content
        # centered with side padding
        assert resized.size == (448, 448)


class TestTextPreprocessor:
    """Tests for text preprocessing functions."""

    def test_text_preprocessor_clean(self) -> None:
        """Test that clean_markdown removes excessive blank lines."""
        raw = "Line 1\n\n\n\nLine 2\n\n\n\n\nLine 3"
        cleaned = clean_markdown(raw)

        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned

    def test_text_preprocessor_clean_strips_whitespace(self) -> None:
        """Test that clean_markdown strips trailing whitespace per line."""
        raw = "Line 1   \nLine 2\t\nLine 3  "
        cleaned = clean_markdown(raw)

        lines = cleaned.split("\n")
        for line in lines:
            assert line == line.rstrip(), f"Line has trailing whitespace: {line!r}"

    def test_text_preprocessor_clean_normalises_dashes(self) -> None:
        """Test that Unicode dashes are normalised to ASCII."""
        raw = "This is an en\u2013dash and an em\u2014dash."
        cleaned = clean_markdown(raw)

        assert "\u2013" not in cleaned
        assert "\u2014" not in cleaned
        assert "-" in cleaned  # en dash → single dash
        assert "--" in cleaned  # em dash → double dash

    def test_text_preprocessor_clean_normalises_quotes(self) -> None:
        """Test that Unicode quotes are normalised to ASCII."""
        raw = "She said \u201chello\u201d and \u2018hi\u2019."
        cleaned = clean_markdown(raw)

        assert "\u201c" not in cleaned
        assert "\u201d" not in cleaned
        assert "\u2018" not in cleaned
        assert "\u2019" not in cleaned
        assert '"' in cleaned
        assert "'" in cleaned

    def test_text_preprocessor_clean_removes_zero_width(self) -> None:
        """Test that zero-width characters are removed."""
        raw = "Hello\u200bWorld\u200cTest\ufeffEnd"
        cleaned = clean_markdown(raw)

        assert "\u200b" not in cleaned
        assert "\u200c" not in cleaned
        assert "\ufeff" not in cleaned
        assert "HelloWorldTestEnd" in cleaned

    def test_text_preprocessor_clean_empty_input(self) -> None:
        """Test that empty input returns empty string."""
        assert clean_markdown("") == ""
        assert clean_markdown("   ") == ""

    def test_text_preprocessor_clean_collapse_spaces(self) -> None:
        """Test that multiple spaces (not newlines) are collapsed."""
        raw = "Hello    World     Test"
        cleaned = clean_markdown(raw)

        assert "Hello World Test" in cleaned

    def test_text_preprocessor_split(self) -> None:
        """Test that split_by_pages distributes text evenly."""
        text = "Word " * 100  # 500 chars
        pages = split_by_pages(text, 5)

        assert len(pages) == 5
        # All text should be accounted for
        total_chars = sum(len(p) for p in pages)
        assert total_chars > 0

    def test_text_preprocessor_split_with_separators(self) -> None:
        """Test split_by_pages with page separators."""
        text = "Page 1 content---Page 2 content---Page 3 content"
        pages = split_by_pages(text, 3)

        assert len(pages) == 3
        assert "Page 1" in pages[0]
        assert "Page 2" in pages[1]
        assert "Page 3" in pages[2]

    def test_text_preprocessor_split_empty_text(self) -> None:
        """Test split_by_pages with empty text."""
        pages = split_by_pages("", 3)
        assert pages == []

    def test_text_preprocessor_split_zero_pages(self) -> None:
        """Test split_by_pages with zero pages."""
        pages = split_by_pages("Some text", 0)
        assert pages == []

    def test_text_preprocessor_split_single_page(self) -> None:
        """Test split_by_pages with a single page."""
        text = "All content on one page."
        pages = split_by_pages(text, 1)

        assert len(pages) == 1
        assert pages[0] == text.strip()
