"""
Models Package — Vision-Language Model Backends.

Re-exports the core classes so that downstream modules can import
from the package root::

    from src.models import BaseVLM, VLMOutput, Qwen2VLModel, ModelFactory
"""

from src.models.base_vlm import BaseVLM, VLMOutput
from src.models.qwen2vl_model import Qwen2VLModel
from src.models.model_factory import ModelFactory

__all__ = [
    "BaseVLM",
    "VLMOutput",
    "Qwen2VLModel",
    "ModelFactory",
]
