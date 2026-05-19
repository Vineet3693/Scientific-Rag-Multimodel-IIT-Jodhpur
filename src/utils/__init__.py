"""
Utilities Package — Device Management, Image Processing, Configuration,
Evaluation Metrics, Visualization, and Error Handling.

Re-exports the key functions and classes from all utility modules so
that downstream code can import from the package root::

    from src.utils import (
        get_logger,
        get_device, get_vram_usage, get_vram_total, free_vram, print_gpu_info,
        load_image, resize_for_colpali, get_image_info, save_image,
        load_config, validate_config, resolve_paths, get_config_path,
        compute_bleu4, compute_rouge_l, compute_anls, compute_f1, batch_compute,
        plot_metrics_comparison, plot_confidence_distribution, plot_retrieval_precision,
        retry_with_backoff, fallback_chain, handle_oom, validate_query, log_error,
        FALLBACK_COLPALI_TO_SCINCL, FALLBACK_SCINCL_TO_TFIDF, FALLBACK_VLM_TO_SOURCES,
    )
"""

# Logging
from src.utils.logging_utils import get_logger

# Device management
from src.utils.device import (
    get_device,
    get_vram_usage,
    get_vram_total,
    free_vram,
    print_gpu_info,
)

# Image utilities
from src.utils.image_utils import (
    load_image,
    resize_for_colpali,
    get_image_info,
    save_image,
)

# Configuration loader
from src.utils.config_loader import (
    load_config,
    validate_config,
    resolve_paths,
    get_config_path,
)

# Evaluation metrics
from src.utils.metrics import (
    compute_bleu4,
    compute_rouge_l,
    compute_anls,
    compute_f1,
    batch_compute,
    compare_with_baseline,
)

# Visualization
from src.utils.visualization import (
    plot_metrics_comparison,
    plot_confidence_distribution,
    plot_retrieval_precision,
)

# Error handling
from src.utils.error_handler import (
    retry_with_backoff,
    fallback_chain,
    handle_oom,
    validate_query,
    log_error,
    FALLBACK_COLPALI_TO_SCINCL,
    FALLBACK_SCINCL_TO_TFIDF,
    FALLBACK_VLM_TO_SOURCES,
)

__all__ = [
    # Logging
    "get_logger",
    # Device
    "get_device",
    "get_vram_usage",
    "get_vram_total",
    "free_vram",
    "print_gpu_info",
    # Image
    "load_image",
    "resize_for_colpali",
    "get_image_info",
    "save_image",
    # Config
    "load_config",
    "validate_config",
    "resolve_paths",
    "get_config_path",
    # Metrics
    "compute_bleu4",
    "compute_rouge_l",
    "compute_anls",
    "compute_f1",
    "batch_compute",
    "compare_with_baseline",
    # Visualization
    "plot_metrics_comparison",
    "plot_confidence_distribution",
    "plot_retrieval_precision",
    # Error handling
    "retry_with_backoff",
    "fallback_chain",
    "handle_oom",
    "validate_query",
    "log_error",
    "FALLBACK_COLPALI_TO_SCINCL",
    "FALLBACK_SCINCL_TO_TFIDF",
    "FALLBACK_VLM_TO_SOURCES",
]
