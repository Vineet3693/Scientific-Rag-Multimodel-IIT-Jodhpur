#!/usr/bin/env python3
"""
Query — CLI Script.

Runs a single query through the online RAG pipeline and prints the
answer, confidence score, and source citations.

Usage:
    python scripts/query.py "What is Vision Transformer?"
    python scripts/query.py "How does self-attention work?" --verbose

Arguments:
    question : The question to ask (positional, required)
    --config : Path to pipeline config YAML (default: configs/pipeline_config.yaml)
    --verbose: Print detailed retrieval scores and self-check info

Example:
    $ python scripts/query.py "What is the Vision Transformer architecture?"
    Answer: The Vision Transformer (ViT) applies a pure transformer…
    Confidence: 85.0%
    Sources:
      - An Image is Worth 16x16 Words (Page 3)
      - Vision Transformer … (Page 5)
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
        description="Query the Scientific Multimodal RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python scripts/query.py "What is Vision Transformer?"\n'
            '  python scripts/query.py "How does self-attention work?" --verbose\n'
        ),
    )

    parser.add_argument(
        "question",
        type=str,
        help="The question to ask the RAG system",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline config YAML (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed retrieval scores and self-check info",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the query script.

    Initializes the online pipeline, runs the query, and prints the
    formatted result to stdout.
    """
    args = parse_args()
    question = args.question

    print(f"Question: {question}")
    print(f"Config   : {args.config}")
    print()

    # Initialize the online pipeline
    from pipelines.online_pipeline import OnlinePipeline

    try:
        pipeline = OnlinePipeline(config_path=args.config)
    except Exception as exc:
        logger.error("Failed to initialize pipeline: %s", exc)
        print(f"ERROR: Could not initialize pipeline: {exc}")
        print("Make sure the index has been built (run scripts/build_index.py first).")
        sys.exit(1)

    # Run the query
    try:
        result = pipeline.query(question)
    except ValueError as exc:
        print(f"Validation error: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        print(f"ERROR: Query failed: {exc}")
        sys.exit(1)

    # Print answer
    print("=" * 60)
    print("  ANSWER")
    print("=" * 60)
    if result.answer:
        print(result.answer)
    else:
        print("(No answer generated)")

    # Print confidence
    print()
    print("=" * 60)
    print("  CONFIDENCE")
    print("=" * 60)
    print(f"  {result.confidence:.1%}")

    # Print sources
    print()
    print("=" * 60)
    print("  SOURCES")
    print("=" * 60)
    if result.sources:
        for i, source in enumerate(result.sources, 1):
            title = source.paper_title if source.paper_title else source.paper_id
            pages = ", ".join(str(p) for p in source.page_numbers)
            print(f"  {i}. {title}")
            print(f"     Page(s): {pages}")
            print(f"     Score: {source.relevance_score:.4f}")
            if source.arxiv_url:
                print(f"     URL: {source.arxiv_url}")
            if source.text_snippet:
                snippet = source.text_snippet[:100].replace("\n", " ")
                print(f"     Snippet: {snippet}…")
            print()
    else:
        print("  (No sources found)")

    # Verbose output
    if args.verbose:
        print("=" * 60)
        print("  RETRIEVAL SCORES")
        print("=" * 60)
        for method, score in result.retrieval_scores.items():
            print(f"  {method}: {score:.4f}")

        print()
        print("=" * 60)
        print("  SELF-CHECK")
        print("=" * 60)
        if result.check_result:
            cr = result.check_result
            print(f"  Overall      : {'PASS' if cr.passed else 'FAIL'}")
            print(f"  Attribution  : {'PASS' if cr.attribution_passed else 'FAIL'}")
            print(f"  Faithfulness : {'PASS' if cr.faithfulness_passed else 'FAIL'}")
            print(f"  Confidence   : {'PASS' if cr.confidence_passed else 'FAIL'}")
            print(f"  Details      : {cr.details}")
        else:
            print("  (No self-check result)")

    # Print timing
    print()
    print("=" * 60)
    print("  TIMING")
    print("=" * 60)
    print(f"  Total time : {result.total_time:.2f}s")
    print(f"  Retries    : {result.retries}")


if __name__ == "__main__":
    main()
