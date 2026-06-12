"""
PDF Parser
==========
Extracts text using PyMuPDF and renders page images using pdf2image.
"""

import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path

class PDFParser:
    """PDF text and image extraction helper."""

    @staticmethod
    def parse_text(pdf_path: str) -> list[str]:
        """Extracts text page-by-page from the PDF."""
        page_texts = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text("text")
                page_texts.append(text)
            doc.close()
        except Exception as e:
            print(f"            ❌ Text extraction failed for {pdf_path}: {e}")
        return page_texts

    @staticmethod
    def render_images(pdf_path: str, dpi: int, output_dir: str, prefix: str) -> list[str]:
        """Renders page images from the PDF and returns list of paths."""
        page_images = []
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            for i, img in enumerate(images):
                img_path = os.path.join(output_dir, f"{prefix}_page_{i+1}.png")
                img.save(img_path, "PNG")
                page_images.append(img_path)
        except Exception as e:
            print(f"            ⚠️ Image rendering failed for {pdf_path}: {e}")
        return page_images

    @classmethod
    def build_metadata(cls, pdf_path: str, arxiv_id: str, title: str, dpi: int, parsed_dirs: dict) -> tuple[dict, dict]:
        """Parses the PDF, renders images, saves markdown, and returns metadata."""
        pages_dir = parsed_dirs.get("pages", "data/parsed/pages")
        markdown_dir = parsed_dirs.get("markdown", "data/parsed/markdown")

        # 1. Extract text
        page_texts = cls.parse_text(pdf_path)
        num_pages = len(page_texts)

        # 2. Render images
        page_images = cls.render_images(pdf_path, dpi, pages_dir, arxiv_id)

        # 3. Save markdown file
        md_path = os.path.join(markdown_dir, f"{arxiv_id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"arXiv ID: {arxiv_id}\n\n")
            for i, text in enumerate(page_texts):
                f.write(f"## Page {i+1}\n\n{text}\n\n---\n\n")

        # 4. Build doc_mapping info
        doc_info = {
            "arxiv_id": arxiv_id,
            "title": title,
            "num_pages": num_pages,
            "page_images": page_images,
            "markdown_path": md_path,
            "status": "success"
        }

        # 5. Build page metadata chunk
        page_chunk = {}
        for i in range(num_pages):
            page_key = f"{arxiv_id}_page_{i+1}"
            page_chunk[page_key] = {
                "doc_id": arxiv_id,
                "page_num": i + 1,
                "image_path": page_images[i] if i < len(page_images) else "",
                "text": page_texts[i] if i < len(page_texts) else "",
                "paper_title": title
            }

        return doc_info, page_chunk
