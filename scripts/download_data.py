#!/usr/bin/env python3
"""
Download arXiv Papers — CLI Script.

Searches arXiv for papers matching a query and category, then downloads
the top PDFs to a specified output directory.  Uses the ArxivDataset
helper from the offline pipeline module.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --query "vision transformer" --category "cs.CV" --max 10
    python scripts/download_data.py --query "attention mechanism" --category "cs.LG" --max 5 --output data/raw/

Arguments:
    --query   : Search query string (default: "vision transformer")
    --category: arXiv category filter (default: "cs.CV")
    --max     : Maximum number of papers to download (default: 10)
    --output  : Output directory for PDFs (default: from config)

Example:
    $ python scripts/download_data.py --query "BERT" --category "cs.CL" --max 5
    Downloading 5 arXiv papers matching 'BERT' in cs.CL…
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so src/ is importable
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
        description="Download arXiv papers for the Scientific Multimodal RAG project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_data.py\n"
            "  python scripts/download_data.py --query 'BERT' --category cs.CL --max 5\n"
            "  python scripts/download_data.py --max 3 --output /tmp/papers/\n"
        ),
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="arXiv search query (default: from data_config.yaml)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="arXiv category filter, e.g. cs.CV, cs.LG (default: from config)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of papers to download (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for PDFs (default: from config)",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=None,
        help="Start year filter (default: from config)",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=None,
        help="End year filter (default: from config)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the download_data script.

    Loads configuration, applies CLI overrides, downloads papers, and
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
    data_cfg = resolved_config.get("data", {})
    paths_cfg = resolved_config.get("paths", {})

    # Apply CLI overrides
    query = args.query or data_cfg.get("query", "vision transformer")
    category = args.category or data_cfg.get("category", "cs.CV")
    max_results = args.max or data_cfg.get("keep_best", 10)
    output_dir = args.output or paths_cfg.get("raw_pdfs", "data/raw/")

    date_range = data_cfg.get("date_range", [2021, 2024])
    if args.year_start is not None:
        date_range[0] = args.year_start
    if args.year_end is not None:
        date_range[1] = args.year_end

    print(f"Query   : {query}")
    print(f"Category: {category}")
    print(f"Max     : {max_results}")
    print(f"Output  : {output_dir}")
    print(f"Years   : {date_range[0]}–{date_range[1]}")
    print()

    # Import ArxivDataset from the offline pipeline
    from pipelines.offline_pipeline import ArxivDataset

    dataset = ArxivDataset(
        query=query,
        category=category,
        max_results=max_results * 2,  # Search wider, then filter
        keep_best=max_results,
        date_range=tuple(date_range),
        output_dir=output_dir,
    )

    results = dataset.download()

    # Print summary
    succeeded = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    print()
    print("=" * 50)
    print("  DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"  Downloaded : {len(succeeded)}")
    print(f"  Failed     : {len(failed)}")

    if succeeded:
        print()
        print("  Downloaded papers:")
        for paper in succeeded:
            print(f"    - {paper['arxiv_id']}: {paper['title'][:60]}…")

    if failed:
        print()
        print("  Failed papers:")
        for paper in failed:
            print(f"    - {paper['arxiv_id']}")

    # Save download manifest
    manifest_path = Path(output_dir) / "download_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Manifest saved: {manifest_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
