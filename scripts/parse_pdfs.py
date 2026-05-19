#!/usr/bin/env python3
"""
Parse PDFs — CLI Script.

Renders PDF pages as images and extracts markdown/text using the
DualPDFParser.  Each page is saved as a separate image file, and
the full text is extracted and saved as markdown.

Usage:
    python scripts/parse_pdfs.py
    python scripts/parse_pdfs.py --input data/raw/ --output data/parsed/
    python scripts/parse_pdfs.py --input data/raw/ --dpi 300

Arguments:
    --input  : Directory containing PDF files (default: from config)
    --output : Base directory for parsed output (default: from config)
    --dpi    : Resolution for page rendering (default: 200)
    --format : Image format, PNG or JPEG (default: PNG)

Example:
    $ python scripts/parse_pdfs.py --input data/raw/ --output data/parsed/
    Parsing PDFs in data/raw/ → data/parsed/…
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.config_loader import load_config, resolve_paths
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Parse PDF files into page images and markdown text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/parse_pdfs.py\n"
            "  python scripts/parse_pdfs.py --input data/raw/ --output data/parsed/\n"
            "  python scripts/parse_pdfs.py --dpi 300 --format JPEG\n"
        ),
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory containing PDF files (default: from data_config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory for parsed files (default: from config)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Resolution for page rendering (default: 200)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["PNG", "JPEG", "png", "jpeg"],
        default=None,
        help="Image output format (default: PNG)",
    )

    return parser.parse_args()


def find_pdfs(input_dir: Path) -> List[Path]:
    """Find all PDF files in a directory.

    Args:
        input_dir: Directory to scan for PDF files.

    Returns:
        Sorted list of Path objects for each PDF found.
    """
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        # Also check subdirectories
        pdfs = sorted(input_dir.rglob("*.pdf"))
    return pdfs


def main() -> None:
    """Main entry point for the parse_pdfs script.

    Loads configuration, discovers PDF files, parses each one, and
    prints a summary of results.
    """
    args = parse_args()

    # Load configuration
    try:
        data_config = load_config("data_config")
    except Exception as exc:
        logger.error("Failed to load data_config: %s", exc)
        print(f"ERROR: Could not load config: {exc}")
        sys.exit(1)

    resolved_config = resolve_paths(data_config)
    paths_cfg = resolved_config.get("paths", {})
    parsing_cfg = data_config.get("parsing", {})

    # Apply CLI overrides
    input_dir = args.input or paths_cfg.get("raw_pdfs", "data/raw/")
    output_base = args.output or str(
        Path(paths_cfg.get("parsed_pages", "data/parsed/pages/")).parent
    )
    dpi = args.dpi or parsing_cfg.get("dpi", 200)
    image_format = (args.format or parsing_cfg.get("image_format", "PNG")).upper()

    pages_dir = str(Path(output_base) / "pages")
    markdown_dir = str(Path(output_base) / "markdown")

    print(f"Input    : {input_dir}")
    print(f"Pages    : {pages_dir}")
    print(f"Markdown : {markdown_dir}")
    print(f"DPI      : {dpi}")
    print(f"Format   : {image_format}")
    print()

    # Find PDFs
    pdf_files = find_pdfs(Path(input_dir))

    if not pdf_files:
        print(f"ERROR: No PDF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) to parse.\n")

    # Import and initialize parser
    from pipelines.offline_pipeline import DualPDFParser

    parser = DualPDFParser(
        output_pages_dir=pages_dir,
        output_markdown_dir=markdown_dir,
        dpi=dpi,
        image_format=image_format,
    )

    # Parse each PDF
    t_start = time.time()
    results = []
    total_pages = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files):
        doc_id = pdf_path.stem
        print(f"[{i + 1}/{len(pdf_files)}] Parsing {doc_id}…")

        try:
            result = parser.parse(str(pdf_path), doc_id=doc_id)
            results.append(result)

            if result["status"] == "success":
                total_pages += result["num_pages"]
                print(
                    f"  ✓ {result['num_pages']} pages, "
                    f"{len(result.get('page_texts', []))} text chunks"
                )
            else:
                failed += 1
                print(f"  ✗ Parse failed")

        except Exception as exc:
            logger.error("Failed to parse %s: %s", doc_id, exc)
            failed += 1
            print(f"  ✗ Error: {exc}")

    total_time = time.time() - t_start

    # Print summary
    print()
    print("=" * 50)
    print("  PARSE SUMMARY")
    print("=" * 50)
    print(f"  PDFs parsed : {len(results)}")
    print(f"  Total pages : {total_pages}")
    print(f"  Failed      : {failed}")
    print(f"  Time        : {total_time:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
