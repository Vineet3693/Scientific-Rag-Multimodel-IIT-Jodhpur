#!/usr/bin/env python3
"""
Push to Kaggle — CLI Script.

Pushes the built index data as a Kaggle Dataset using the Kaggle API.
This allows the offline pipeline output (embeddings, ChromaDB, metadata)
to be shared and reused in a separate Kaggle notebook for online
inference without re-running the expensive indexing step.

Usage:
    python scripts/push_to_kaggle.py
    python scripts/push_to_kaggle.py --dataset-name sci-rag-index
    python scripts/push_to_kaggle.py --dataset-name my-rag-index --index-dir data/indices/

Prerequisites:
    - Kaggle API key configured (~/.kaggle/kaggle.json)
    - kaggle package installed (pip install kaggle)

Arguments:
    --dataset-name  : Name for the Kaggle dataset (default: sci-rag-index)
    --dataset-title : Human-readable title (default: Scientific RAG Index)
    --index-dir     : Local directory with index data (default: from config)
    --update        : Update existing dataset instead of creating new one

Example:
    $ python scripts/push_to_kaggle.py --dataset-name sci-rag-index
    Creating Kaggle dataset 'sci-rag-index'…
    ✓ Dataset created: https://www.kaggle.com/datasets/username/sci-rag-index
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

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
        description="Push index data as a Kaggle Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/push_to_kaggle.py\n"
            "  python scripts/push_to_kaggle.py --dataset-name my-index\n"
            "  python scripts/push_to_kaggle.py --update\n"
        ),
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Kaggle dataset slug (default: from pipeline_config.yaml)",
    )
    parser.add_argument(
        "--dataset-title",
        type=str,
        default=None,
        help="Human-readable dataset title (default: from config)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Local directory with index data (default: from config)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Update existing dataset instead of creating a new one",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Kaggle username (default: from kaggle.json)",
    )

    return parser.parse_args()


def check_kaggle_api() -> None:
    """Verify that the Kaggle API is available and configured.

    Raises:
        ImportError: If the kaggle package is not installed.
        RuntimeError: If the API key is not configured.
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'kaggle' package is required.  "
            "Install it with: pip install kaggle"
        )

    # Check for API key
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise RuntimeError(
            "Kaggle API key not found at ~/.kaggle/kaggle.json.  "
            "Create it by following: "
            "https://www.kaggle.com/docs/api"
        )

    logger.info("Kaggle API configured.")


def create_dataset_metadata(
    dataset_dir: Path,
    dataset_name: str,
    dataset_title: str,
    username: str,
) -> None:
    """Create the dataset-metadata.json file required by Kaggle.

    Args:
        dataset_dir: Directory where the metadata file will be written.
        dataset_name: Dataset slug name.
        dataset_title: Human-readable dataset title.
        username: Kaggle username.
    """
    metadata = {
        "title": dataset_title,
        "id": f"{username}/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}],
    }

    metadata_path = dataset_dir / "dataset-metadata.json"
    with open(str(metadata_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Created dataset-metadata.json: %s", metadata_path)


def prepare_dataset_directory(
    index_dir: Path,
    dataset_name: str,
    dataset_title: str,
    username: str,
) -> Path:
    """Prepare a temporary directory with all files for the Kaggle dataset.

    Copies index files and creates the required metadata file.

    Args:
        index_dir: Source directory with index data.
        dataset_name: Dataset slug name.
        dataset_title: Human-readable dataset title.
        username: Kaggle username.

    Returns:
        Path to the prepared dataset directory.
    """
    # Create a temporary directory for staging
    staging_dir = Path(tempfile.mkdtemp(prefix="kaggle_dataset_"))

    logger.info("Staging dataset in: %s", staging_dir)

    # Copy all files from the index directory
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index directory not found: {index_dir}"
        )

    file_count = 0
    total_size = 0

    for item in index_dir.rglob("*"):
        if item.is_file():
            # Preserve directory structure
            rel_path = item.relative_to(index_dir)
            dest = staging_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(dest))
            file_count += 1
            total_size += item.stat().st_size

    logger.info(
        "Copied %d files (%.2f MB) to staging directory.",
        file_count,
        total_size / (1024 * 1024),
    )

    # Create metadata
    create_dataset_metadata(staging_dir, dataset_name, dataset_title, username)

    return staging_dir


def push_dataset(
    dataset_dir: Path,
    update: bool = False,
) -> str:
    """Push the dataset to Kaggle using the API.

    Args:
        dataset_dir: Directory containing the dataset files and metadata.
        update: If True, update an existing dataset instead of creating
            a new one.

    Returns:
        The URL of the created/updated dataset on Kaggle.

    Raises:
        RuntimeError: If the push fails.
    """
    import kaggle

    try:
        if update:
            logger.info("Updating existing Kaggle dataset…")
            kaggle.api.dataset_metadata_update(str(dataset_dir))
            kaggle.api.dataset_create_version(
                str(dataset_dir),
                version_notes="Updated index data",
            )
            logger.info("Dataset updated successfully.")
        else:
            logger.info("Creating new Kaggle dataset…")
            kaggle.api.dataset_create_new(str(dataset_dir))
            logger.info("Dataset created successfully.")

    except Exception as exc:
        raise RuntimeError(
            f"Failed to push dataset to Kaggle: {exc}"
        ) from exc

    # Construct URL
    metadata_path = dataset_dir / "dataset-metadata.json"
    if metadata_path.exists():
        with open(str(metadata_path), "r") as f:
            meta = json.load(f)
        dataset_id = meta.get("id", "unknown/unknown")
        url = f"https://www.kaggle.com/datasets/{dataset_id}"
    else:
        url = "https://www.kaggle.com/datasets"

    return url


def get_kaggle_username() -> str:
    """Get the Kaggle username from the API key file.

    Returns:
        The Kaggle username string.

    Raises:
        RuntimeError: If the username cannot be determined.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(str(kaggle_json), "r") as f:
                data = json.load(f)
            return data.get("username", "")
        except Exception:
            pass

    # Try environment variable
    username = os.environ.get("KAGGLE_USERNAME", "")
    if username:
        return username

    raise RuntimeError(
        "Could not determine Kaggle username.  "
        "Set KAGGLE_USERNAME environment variable or configure "
        "~/.kaggle/kaggle.json."
    )


def main() -> None:
    """Main entry point for the push_to_kaggle script.

    Checks prerequisites, stages the dataset, and pushes it to Kaggle.
    """
    args = parse_args()

    # Load config for defaults
    try:
        from src.utils.config_loader import load_config, resolve_paths

        pipeline_config = load_config("pipeline_config")
        data_config = load_config("data_config")
        resolved_data = resolve_paths(data_config)

        offline_cfg = pipeline_config.get("pipeline", {}).get("offline", {})
        paths_cfg = resolved_data.get("paths", {})
    except Exception:
        offline_cfg = {}
        paths_cfg = {}

    # Apply CLI overrides
    dataset_name = args.dataset_name or offline_cfg.get(
        "dataset_name", "sci-rag-index"
    )
    dataset_title = args.dataset_title or offline_cfg.get(
        "dataset_title", "Scientific RAG Index"
    )
    index_dir = args.index_dir or str(
        Path(
            paths_cfg.get("chroma_index", "data/indices/chroma_index/")
        ).parent
    )

    print(f"Dataset name  : {dataset_name}")
    print(f"Dataset title : {dataset_title}")
    print(f"Index dir     : {index_dir}")
    print(f"Mode          : {'Update' if args.update else 'Create new'}")
    print()

    # Check prerequisites
    try:
        check_kaggle_api()
    except (ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Get username
    try:
        username = args.username or get_kaggle_username()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Kaggle user   : {username}")
    print()

    # Prepare staging directory
    try:
        staging_dir = prepare_dataset_directory(
            index_dir=Path(index_dir),
            dataset_name=dataset_name,
            dataset_title=dataset_title,
            username=username,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Push to Kaggle
    try:
        url = push_dataset(staging_dir, update=args.update)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        # Clean up staging directory
        shutil.rmtree(str(staging_dir), ignore_errors=True)
        sys.exit(1)

    # Clean up staging directory
    shutil.rmtree(str(staging_dir), ignore_errors=True)

    # Print result
    print()
    print("=" * 50)
    print("  KAGGLE DATASET PUSHED")
    print("=" * 50)
    print(f"  Name  : {dataset_name}")
    print(f"  Title : {dataset_title}")
    print(f"  URL   : {url}")
    print("=" * 50)
    print()
    print("Use this dataset as input in your Kaggle notebook:")
    print(f"  /kaggle/input/{dataset_name}/")


if __name__ == "__main__":
    main()
