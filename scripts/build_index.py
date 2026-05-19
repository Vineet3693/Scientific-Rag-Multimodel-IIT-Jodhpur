#!/usr/bin/env python3
"""
Build Index — CLI Script.

Runs the complete offline pipeline to build the retrieval index:
download → parse → embed → store.  Supports checkpoint-based resume
so that interrupted runs can be continued without reprocessing.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --config configs/pipeline_config.yaml
    python scripts/build_index.py --resume

Arguments:
    --config : Path to pipeline config YAML (default: configs/pipeline_config.yaml)
    --resume : Resume from checkpoint instead of starting fresh

Example:
    $ python scripts/build_index.py
    Building index from configs/pipeline_config.yaml…
    
    $ python scripts/build_index.py --resume
    Resuming from checkpoint…
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the retrieval index for Scientific Multimodal RAG. "
            "Runs the full offline pipeline: download → parse → embed → store."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/build_index.py\n"
            "  python scripts/build_index.py --config my_config.yaml\n"
            "  python scripts/build_index.py --resume\n"
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline config YAML (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint instead of starting fresh",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the build_index script.

    Loads the offline pipeline and either runs it from scratch or
    resumes from a checkpoint.  Prints the summary report on completion.
    """
    args = parse_args()

    config_path = args.config
    resume_mode = args.resume

    print("=" * 60)
    if resume_mode:
        print("  BUILD INDEX — RESUME MODE")
    else:
        print("  BUILD INDEX — FULL RUN")
    print("=" * 60)
    print(f"  Config : {config_path}")
    print(f"  Mode   : {'Resume from checkpoint' if resume_mode else 'Fresh run'}")
    print()

    # Import the pipeline (may take a moment for torch init)
    from pipelines.offline_pipeline import OfflinePipeline

    try:
        pipeline = OfflinePipeline(config_path=config_path)
    except Exception as exc:
        logger.error("Failed to initialize pipeline: %s", exc)
        print(f"ERROR: Could not initialize pipeline: {exc}")
        print("Make sure all configuration files exist in configs/.")
        sys.exit(1)

    # Run or resume
    try:
        if resume_mode:
            result = pipeline.resume()
        else:
            result = pipeline.run()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        print("Run with --resume to continue from checkpoint.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        print(f"\nERROR: Pipeline failed: {exc}")
        print("Run with --resume to continue from checkpoint.")
        sys.exit(1)

    # Print results
    print()
    print("=" * 60)
    print("  BUILD INDEX — COMPLETE")
    print("=" * 60)
    print(f"  Papers processed : {result['papers_processed']}")
    print(f"  Pages embedded   : {result['pages_embedded']}")
    print(f"  Failed downloads : {result['failed_downloads']}")
    print(f"  Failed parses    : {result['failed_parses']}")
    print(f"  ColPali time     : {result['colpali_time']:.1f}s")
    print(f"  SciNCL time      : {result['scincl_time']:.1f}s")
    print(f"  Total time       : {result['total_time']:.1f}s")
    print("=" * 60)

    # Check for errors
    if result["failed_downloads"] > 0 or result["failed_parses"] > 0:
        print()
        print("⚠ Some papers failed to process. Check the logs for details.")
        print("  You can run with --resume to retry failed papers.")

    if result["pages_embedded"] == 0:
        print()
        print("ERROR: No pages were embedded. The index is empty.")
        print("  Check that PDFs were downloaded and parsed successfully.")
        sys.exit(1)

    print("\n✓ Index built successfully. Ready for online queries.")


if __name__ == "__main__":
    main()
