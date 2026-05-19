"""
Image Utilities — Loading, Resizing, and Inspecting Page Images.

Provides helper functions for working with PIL images in the
Scientific Multimodal RAG pipeline.  These utilities are used by
the embedder (ColPali), retriever (page image collection), context
builder (VLM input), and evaluation modules.

Key functions:

* :func:`load_image` — Load an image from a file path.
* :func:`resize_for_colpali` — Resize an image to ColPali's expected
  448×448 input resolution while preserving aspect ratio.
* :func:`get_image_info` — Return a dictionary of image metadata
  (size, mode, DPI).
* :func:`save_image` — Save an image to disk.

Example:
    >>> from src.utils.image_utils import load_image, resize_for_colpali
    >>> img = load_image("data/parsed/pages/2305.12345_page_3.png")
    >>> resized = resize_for_colpali(img)
    >>> print(resized.size)  # (448, 448)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import PIL.Image

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

def load_image(path: Union[str, Path]) -> PIL.Image.Image:
    """Load an image from a file path.

    Opens the image file and converts it to RGB mode to ensure
    compatibility with all downstream models (ColPali, Qwen2-VL).

    Args:
        path: Path to the image file.  Supports all formats that
            Pillow can read (PNG, JPEG, TIFF, BMP, etc.).

    Returns:
        A ``PIL.Image.Image`` in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be opened as an image.

    Example:
        >>> img = load_image("data/parsed/pages/paper_page_3.png")
        >>> print(img.size, img.mode)
        (595, 842) RGB
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        image = PIL.Image.open(str(path))
        # Convert to RGB to ensure compatibility with all models.
        if image.mode != "RGB":
            logger.debug(
                "Converting image from %s to RGB: %s",
                image.mode,
                path.name,
            )
            image = image.convert("RGB")

        logger.debug(
            "Loaded image: %s — size: %s, mode: %s",
            path.name,
            image.size,
            image.mode,
        )
        return image

    except PIL.UnidentifiedImageError:
        raise ValueError(
            f"Cannot open image file (unidentified format): {path}"
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to load image {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# resize_for_colpali
# ---------------------------------------------------------------------------

def resize_for_colpali(
    image: PIL.Image.Image,
    size: Tuple[int, int] = (448, 448),
) -> PIL.Image.Image:
    """Resize an image for ColPali input with aspect ratio preservation.

    ColPali expects input images of size 448×448 pixels.  This function
    first resizes the image so that the longer side matches the target
    size while preserving the aspect ratio, then pads the shorter side
    with white pixels to reach the exact target dimensions.

    This two-step approach (resize + pad) avoids distortion that would
    occur from a direct resize to the target dimensions.

    Args:
        image: Input ``PIL.Image.Image`` in any mode (will be
            converted to RGB).
        size: Target size as ``(width, height)``.  Defaults to
            ``(448, 448)`` which is ColPali's expected input resolution.

    Returns:
        A new ``PIL.Image.Image`` of the specified size in RGB mode,
        with the original content centered and white padding.

    Example:
        >>> img = PIL.Image.new("RGB", (800, 1200))
        >>> resized = resize_for_colpali(img)
        >>> print(resized.size)  # (448, 448)
    """
    target_w, target_h = size

    # Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Calculate scaling factor to fit within target size
    orig_w, orig_h = image.size
    scale = min(target_w / orig_w, target_h / orig_h)

    # Resize with aspect ratio preservation
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Ensure minimum dimensions
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    resized = image.resize((new_w, new_h), PIL.Image.LANCZOS)

    # If the resized image already matches the target, return it directly
    if new_w == target_w and new_h == target_h:
        return resized

    # Create a white canvas and paste the resized image centered
    canvas = PIL.Image.new("RGB", (target_w, target_h), (255, 255, 255))

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    canvas.paste(resized, (paste_x, paste_y))

    logger.debug(
        "Resized image from (%d, %d) to (%d, %d) with padding — "
        "scale: %.3f",
        orig_w, orig_h, target_w, target_h, scale,
    )

    return canvas


# ---------------------------------------------------------------------------
# get_image_info
# ---------------------------------------------------------------------------

def get_image_info(image: PIL.Image.Image) -> Dict[str, Any]:
    """Return a dictionary of image metadata.

    Extracts size, mode, DPI, and format information from a PIL
    image object.  Useful for logging and debugging.

    Args:
        image: A ``PIL.Image.Image`` instance.

    Returns:
        A dictionary with the following keys:

        * ``"size"`` — ``(width, height)`` tuple.
        * ``"width"`` — Image width in pixels.
        * ``"height"`` — Image height in pixels.
        * ``"mode"`` — PIL image mode string (e.g. ``"RGB"``,
          ``"L"``, ``"RGBA"``).
        * ``"dpi"`` — DPI as ``(x_dpi, y_dpi)`` tuple, or ``None``
          if not available.
        * ``"format"`` — Image format string (e.g. ``"PNG"``,
          ``"JPEG"``), or ``None`` if the image was created in
          memory.
        * ``"channels"`` — Number of color channels.

    Example:
        >>> img = PIL.Image.new("RGB", (800, 600))
        >>> info = get_image_info(img)
        >>> print(info["size"], info["mode"])
        (800, 600) RGB
    """
    dpi = image.info.get("dpi", None)

    # Determine number of channels from mode
    mode_channel_map = {
        "1": 1, "L": 1, "P": 1, "I": 1, "F": 1,
        "RGB": 3, "YCbCr": 3, "LAB": 3, "HSV": 3,
        "RGBA": 4, "CMYK": 4, "RGBX": 4, "RGBa": 4,
        "PA": 2, "LA": 2,
    }
    channels = mode_channel_map.get(image.mode, None)

    info: Dict[str, Any] = {
        "size": image.size,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "dpi": dpi,
        "format": image.format,
        "channels": channels,
    }

    logger.debug("Image info: %s", info)
    return info


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------

def save_image(
    image: PIL.Image.Image,
    path: Union[str, Path],
    format: Optional[str] = None,
    quality: int = 95,
) -> None:
    """Save an image to a file.

    Creates the parent directory if it does not exist.  The output
    format is inferred from the file extension if not explicitly
    specified.

    Args:
        image: The ``PIL.Image.Image`` to save.
        path: Destination file path.  The extension determines the
            output format (e.g. ``.png``, ``.jpg``, ``.tiff``).
        format: Optional explicit format override (e.g. ``"PNG"``,
            ``"JPEG"``).  If ``None``, the format is inferred from
            the file extension.
        quality: JPEG quality (1-100).  Only used for JPEG format.
            Defaults to 95.

    Raises:
        ValueError: If the image cannot be saved (e.g. unsupported
            format, invalid path).
        IOError: If the file cannot be written.

    Example:
        >>> img = PIL.Image.new("RGB", (100, 100))
        >>> save_image(img, "output/page_3.png")
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Infer format from extension if not provided
    if format is None:
        ext_to_format = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".tiff": "TIFF",
            ".tif": "TIFF",
            ".bmp": "BMP",
            ".webp": "WEBP",
        }
        ext = path.suffix.lower()
        format = ext_to_format.get(ext, "PNG")

    try:
        save_kwargs: Dict[str, Any] = {}
        if format == "JPEG":
            save_kwargs["quality"] = quality
            # JPEG does not support alpha channels
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

        image.save(str(path), format=format, **save_kwargs)

        logger.info(
            "Saved image to %s — format: %s, size: %s",
            path,
            format,
            image.size,
        )

    except Exception as exc:
        raise IOError(
            f"Failed to save image to {path}: {exc}"
        ) from exc
