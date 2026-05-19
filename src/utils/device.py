"""
Device Utilities — GPU Detection, VRAM Management, and Hardware Info.

Provides utilities for automatic device detection, VRAM monitoring,
memory cleanup, and GPU information display.  Designed for the
staggered model-loading strategy used in the Scientific Multimodal
RAG pipeline, where models are loaded one at a time and VRAM must
be carefully managed.

This module maintains the same interface as Gokul's medical RAG
repository for cross-project compatibility.

Example:
    >>> from src.utils.device import get_device, get_vram_usage, print_gpu_info
    >>> device = get_device()
    >>> print(f"Using device: {device}")
    >>> print(f"VRAM usage: {get_vram_usage():.2f} GB")
    >>> print_gpu_info()
"""

from __future__ import annotations

import gc
from typing import Optional

import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Checks for CUDA availability first, then falls back to CPU.
    The returned ``torch.device`` can be passed directly to model
    constructors and tensor operations.

    Returns:
        ``torch.device("cuda")`` if a CUDA-capable GPU is available,
        otherwise ``torch.device("cpu")``.

    Example:
        >>> device = get_device()
        >>> model = AutoModel.from_pretrained("bert-base").to(device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug("CUDA available — using device: %s", device)
    else:
        device = torch.device("cpu")
        logger.debug("CUDA not available — falling back to CPU.")

    return device


# ---------------------------------------------------------------------------
# get_vram_usage
# ---------------------------------------------------------------------------

def get_vram_usage() -> float:
    """Get current GPU VRAM usage in gigabytes.

    Returns the amount of GPU memory currently allocated by tensors
    (not including cached/pooled memory).  Useful for monitoring VRAM
    consumption during the staggered model-loading pipeline.

    Returns:
        Current VRAM usage in GB.  Returns 0.0 if no GPU is available.

    Example:
        >>> usage = get_vram_usage()
        >>> print(f"Currently using {usage:.2f} GB of VRAM")
    """
    if not torch.cuda.is_available():
        return 0.0

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    logger.debug("VRAM allocated: %.4f GB", allocated)
    return allocated


# ---------------------------------------------------------------------------
# get_vram_total
# ---------------------------------------------------------------------------

def get_vram_total() -> float:
    """Get total GPU VRAM capacity in gigabytes.

    Returns the total amount of GPU memory available on the current
    device, as reported by the CUDA driver.

    Returns:
        Total VRAM in GB.  Returns 0.0 if no GPU is available.

    Example:
        >>> total = get_vram_total()
        >>> print(f"GPU has {total:.2f} GB of VRAM")
    """
    if not torch.cuda.is_available():
        return 0.0

    total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    logger.debug("VRAM total: %.4f GB", total)
    return total


# ---------------------------------------------------------------------------
# free_vram
# ---------------------------------------------------------------------------

def free_vram() -> None:
    """Force-release GPU memory by running garbage collection and
    emptying the CUDA cache.

    This is essential in the staggered-loading pipeline: after
    unloading one model, calling ``free_vram()`` ensures that the
    GPU memory is actually returned to the allocator before the
    next model is loaded.

    The method performs two steps:

    1. ``gc.collect()`` — Force Python garbage collection to release
       any objects (tensors, models) that are no longer referenced.
    2. ``torch.cuda.empty_cache()`` — Release cached memory blocks
       back to the CUDA driver so they can be allocated by the next
       model.

    Example:
        >>> del model
        >>> free_vram()
        >>> # Now safe to load the next model
    """
    vram_before = get_vram_usage()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vram_after = get_vram_usage()
    freed = vram_before - vram_after

    logger.info(
        "free_vram() — before: %.2f GB, after: %.2f GB, freed: %.2f GB",
        vram_before,
        vram_after,
        freed,
    )


# ---------------------------------------------------------------------------
# print_gpu_info
# ---------------------------------------------------------------------------

def print_gpu_info() -> None:
    """Print comprehensive GPU information to the logger.

    Displays the GPU name, total VRAM, current VRAM usage, CUDA
    version, and PyTorch CUDA version.  Useful for debugging and
    verifying the GPU environment at pipeline startup.

    Example:
        >>> print_gpu_info()
        # Output:
        # GPU: Tesla P100
        # VRAM: 15.90 GB total, 2.50 GB used
        # CUDA: 12.1 (driver), 12.1 (PyTorch)
    """
    if not torch.cuda.is_available():
        logger.info("No GPU available — running on CPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = get_vram_total()
    used_vram = get_vram_usage()
    cuda_version_driver = torch.version.cuda or "N/A"
    cuda_version_pytorch = torch.backends.cuda.version_string() if torch.backends.cuda.is_built() else "N/A"

    logger.info("GPU: %s", gpu_name)
    logger.info(
        "VRAM: %.2f GB total, %.2f GB used (%.1f%%)",
        total_vram,
        used_vram,
        (used_vram / total_vram * 100) if total_vram > 0 else 0.0,
    )
    logger.info(
        "CUDA: %s (driver), %s (PyTorch)",
        cuda_version_driver,
        cuda_version_pytorch,
    )

    # Also print to stdout for notebook visibility
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.2f} GB total, {used_vram:.2f} GB used")
    print(f"CUDA: {cuda_version_driver} (driver), {cuda_version_pytorch} (PyTorch)")
