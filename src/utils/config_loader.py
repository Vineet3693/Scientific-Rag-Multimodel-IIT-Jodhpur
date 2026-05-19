"""
Configuration Loader — YAML Config Loading, Validation, and Path Resolution.

Provides utilities for loading YAML configuration files from the
``configs/`` directory, validating that required keys are present,
and automatically resolving file paths for different runtime
environments (Kaggle vs. local).

Path Resolution Logic
---------------------
The :func:`resolve_paths` function automatically detects the runtime
environment and adjusts data paths accordingly:

* **Kaggle**: If ``/kaggle/working/`` exists, data paths are resolved
  relative to ``/kaggle/working/``.
* **Local**: Data paths are resolved relative to the project root
  directory (parent of ``src/``).

This allows the same configuration files to work seamlessly in both
environments without manual edits.

Example:
    >>> from src.utils.config_loader import load_config, validate_config
    >>> config = load_config("model_config")
    >>> is_valid = validate_config(config, ["models", "models.colpali"])
    >>> print(f"Config valid: {is_valid}")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Try importing yaml; provide a helpful error if missing.
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# get_config_path
# ---------------------------------------------------------------------------

def get_config_path() -> Path:
    """Find the ``configs/`` directory.

    Searches for the ``configs/`` directory by walking up from the
    current file location to the project root.  The project root is
    identified as the first parent directory that contains a
    ``configs/`` subdirectory.

    Returns:
        The absolute path to the ``configs/`` directory.

    Raises:
        FileNotFoundError: If the ``configs/`` directory cannot be
            found after walking up to the filesystem root.
    """
    # Start from the directory of this file and walk up.
    current = Path(__file__).resolve().parent

    for _ in range(10):  # Limit depth to avoid infinite loop
        candidate = current / "configs"
        if candidate.is_dir():
            logger.debug("Found configs directory: %s", candidate)
            return candidate

        parent = current.parent
        if parent == current:
            break  # Reached filesystem root
        current = parent

    raise FileNotFoundError(
        "Could not find 'configs/' directory.  "
        "Ensure the project structure is intact."
    )


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def load_config(filename: str) -> Dict[str, Any]:
    """Load a YAML configuration file from the ``configs/`` directory.

    The function locates the ``configs/`` directory automatically,
    appends the ``.yaml`` extension if not present, and parses the
    file contents.

    Args:
        filename: Configuration file name without the ``configs/``
            prefix.  The ``.yaml`` extension is added automatically
            if not present.  Examples: ``"model_config"``,
            ``"retrieval_config"``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        ImportError: If ``pyyaml`` is not installed.
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file cannot be parsed as valid YAML.

    Example:
        >>> config = load_config("model_config")
        >>> print(config["models"]["colpali"]["model_name"])
        'vidore/colpali-v1.2'
    """
    if not _HAS_YAML:
        raise ImportError(
            "pyyaml is required for load_config().  "
            "Install it with: pip install pyyaml"
        )

    # Locate configs directory
    configs_dir = get_config_path()

    # Add .yaml extension if missing
    if not filename.endswith(".yaml") and not filename.endswith(".yml"):
        filename = f"{filename}.yaml"

    config_path = configs_dir / filename

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    logger.info("Loading config from: %s", config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        logger.debug(
            "Config loaded — top-level keys: %s",
            list(config.keys()) if isinstance(config, dict) else "non-dict",
        )

        return config

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse YAML config {config_path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate that all required keys exist in the configuration.

    Checks that each key in *required_keys* is present in the
    configuration dictionary.  Nested keys can be specified using
    dot notation (e.g. ``"models.colpali.model_name"``).

    Args:
        config: The configuration dictionary to validate.
        required_keys: A list of required key paths.  Use dot
            notation for nested keys, e.g. ``"models.colpali"``.

    Returns:
        ``True`` if all required keys are present, ``False`` otherwise.

    Example:
        >>> config = {"models": {"colpali": {"model_name": "vidore/colpali-v1.2"}}}
        >>> validate_config(config, ["models", "models.colpali", "models.colpali.model_name"])
        True
        >>> validate_config(config, ["models", "models.scincl"])
        False
    """
    if not isinstance(config, dict):
        logger.error("Config is not a dictionary: %s", type(config).__name__)
        return False

    missing_keys: List[str] = []

    for key_path in required_keys:
        if not _has_nested_key(config, key_path):
            missing_keys.append(key_path)

    if missing_keys:
        logger.warning(
            "Config validation failed — missing keys: %s",
            missing_keys,
        )
        return False

    logger.debug("Config validation passed — all %d keys present.", len(required_keys))
    return True


# ---------------------------------------------------------------------------
# resolve_paths
# ---------------------------------------------------------------------------

def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-resolve data paths for Kaggle or local environments.

    Detects the runtime environment and adjusts the ``paths`` section
    of the configuration:

    * **Kaggle** (``/kaggle/working/`` exists): Prefixes all data
      paths with ``/kaggle/working/``.
    * **Local**: Prefixes all data paths with the project root
      directory (parent of ``src/``).

    Only string values within the ``paths`` section are resolved.
    Other configuration sections are left unchanged.

    Args:
        config: The configuration dictionary.  Must contain a
            ``"paths"`` key for path resolution to take effect.

    Returns:
        A new configuration dictionary with resolved paths.  The
        original *config* is **not** modified.

    Example:
        >>> config = {"paths": {"raw_pdfs": "data/raw/", "chroma_index": "data/indices/"}}
        >>> resolved = resolve_paths(config)
        >>> # On Kaggle: resolved["paths"]["raw_pdfs"] == "/kaggle/working/data/raw/"
        >>> # Locally: resolved["paths"]["raw_pdfs"] == "/home/user/project/data/raw/"
    """
    import copy

    resolved = copy.deepcopy(config)

    if "paths" not in resolved:
        logger.debug("No 'paths' section in config — nothing to resolve.")
        return resolved

    # Detect base directory
    kaggle_working = Path("/kaggle/working/")
    if kaggle_working.exists():
        base_dir = kaggle_working
        logger.info("Detected Kaggle environment — base_dir: %s", base_dir)
    else:
        # Use project root (parent of src/)
        base_dir = Path(__file__).resolve().parent.parent.parent
        logger.info("Detected local environment — base_dir: %s", base_dir)

    # Resolve each path
    paths = resolved["paths"]
    for key, value in paths.items():
        if isinstance(value, str) and not Path(value).is_absolute():
            resolved_path = str(base_dir / value)
            logger.debug(
                "Resolved paths.%s: %s → %s",
                key,
                value,
                resolved_path,
            )
            paths[key] = resolved_path

    return resolved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_nested_key(config: Dict[str, Any], key_path: str) -> bool:
    """Check whether a nested key path exists in a dictionary.

    Supports dot notation for nested access, e.g.
    ``"models.colpali.model_name"``.

    Args:
        config: The dictionary to search.
        key_path: Dot-separated key path.

    Returns:
        ``True`` if the key path exists, ``False`` otherwise.
    """
    keys = key_path.split(".")
    current = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]

    return True
