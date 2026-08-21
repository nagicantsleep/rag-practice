"""Controlled multimodal retrieval primitives."""

from .clip import CLIPTextToImageRetriever
from .ppm import PALETTE, RasterImage, read_p3_ppm
from .retrieval import ImageAsset, MultimodalImageIndex, RetrievalResult, VisualRequest

__all__ = [
    "CLIPTextToImageRetriever",
    "PALETTE",
    "RasterImage",
    "read_p3_ppm",
    "ImageAsset",
    "MultimodalImageIndex",
    "RetrievalResult",
    "VisualRequest",
]
