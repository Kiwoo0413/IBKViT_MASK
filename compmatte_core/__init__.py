"""
CompMatte Core Package
VFX Compositing Architecture-Based Hybrid Matting Pipeline.

Core Algorithms:
- CoreEngine: Pure-white core matte extraction, black envelope, and spatio-temporal stabilization.
- IBKEngine: Optical color-difference transmission edge keyer.
- ViTMatteEngine: Vision Transformer neural edge refiner for unknown transition zones.
- MatteFusionEngine: Compositing fusion with non-destructive fine edge detail re-injection.
- VideoIO, ImageSequenceIO: High-performance 4K video and OpenEXR/PNG sequence I/O.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure library root is in sys.path
_LIB_ROOT = Path(__file__).resolve().parent.parent
if str(_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIB_ROOT))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Re-export core algorithms from algorithmic backend
from ibkvit_core.ibk_engine import IBKEngine, ScreenType, IBKMatteResult
from ibkvit_core.core_engine import CoreEngine, parse_coords, parse_box
from ibkvit_core.vit_engine import ViTEngine, ViTMatteEngine, ViTMatteResult
from ibkvit_core.matte_fusion import MatteFusionEngine, FusionConfig
from ibkvit_core.io_utils import VideoIO, ImageSequenceIO

# Compositing-grade aliases
CompMatteCoreEngine = CoreEngine
CompMatteEdgeEngine = ViTMatteEngine
CompMatteFusionEngine = MatteFusionEngine
CompMatteFusionConfig = FusionConfig
CompMatteIO = VideoIO

__all__ = [
    # Primary CompMatte names
    "CompMatteCoreEngine",
    "CompMatteEdgeEngine",
    "CompMatteFusionEngine",
    "CompMatteFusionConfig",
    "CompMatteIO",
    # Underlying core classes
    "CoreEngine",
    "IBKEngine",
    "ViTMatteEngine",
    "ViTEngine",
    "MatteFusionEngine",
    "FusionConfig",
    "VideoIO",
    "ImageSequenceIO",
    "ScreenType",
    "IBKMatteResult",
    "ViTMatteResult",
    "parse_coords",
    "parse_box",
]
