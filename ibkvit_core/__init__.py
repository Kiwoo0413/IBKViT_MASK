"""
core package for VFX IBK Keying & ViT Masking Toolkit.
Pure Python/PyTorch/OpenCV algorithmic core.
"""

import os
import sys
from pathlib import Path

_LIB_ROOT = Path(__file__).resolve().parent.parent
if str(_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIB_ROOT))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from .ibk_engine import IBKEngine, ScreenType, IBKMatteResult
from .core_engine import CoreEngine, parse_coords, parse_box
from .vit_engine import ViTEngine, ViTMatteEngine, ViTMatteResult
from .matte_fusion import MatteFusionEngine, FusionConfig
from .io_utils import VideoIO, ImageSequenceIO

__all__ = [
    "IBKEngine",
    "ScreenType",
    "IBKMatteResult",
    "CoreEngine",
    "ViTMatteEngine",
    "ViTEngine",
    "ViTMatteResult",
    "parse_coords",
    "parse_box",
    "MatteFusionEngine",
    "FusionConfig",
    "VideoIO",
    "ImageSequenceIO",
]
