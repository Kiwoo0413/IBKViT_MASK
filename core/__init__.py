"""
core package for VFX IBK Keying & ViT Masking Toolkit.
Pure Python/PyTorch/OpenCV algorithmic core.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from .ibk_engine import IBKEngine, ScreenType, IBKMatteResult
from .vit_engine import ViTEngine, ViTMatteResult
from .matte_fusion import MatteFusionEngine, FusionConfig
from .io_utils import VideoIO, ImageSequenceIO

__all__ = [
    "IBKEngine",
    "ScreenType",
    "IBKMatteResult",
    "ViTEngine",
    "ViTMatteResult",
    "MatteFusionEngine",
    "FusionConfig",
    "VideoIO",
    "ImageSequenceIO",
]
