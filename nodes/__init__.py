"""
nodes package for CompMatte: VFX Compositing-Grade Matting Custom Nodes for Griptape Nodes Desktop.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LIB_ROOT = Path(__file__).resolve().parent.parent
if str(_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIB_ROOT))

from .ibk_clean_plate_node import IBKCleanPlateNode, CompMatteCleanPlateNode
from .ibk_keyer_node import IBKKeyerNode, CompMatteKeyerNode
from .vit_mask_extractor_node import ViTMaskExtractorNode, CompMatteViTEdgeNode
from .vfx_matte_refiner_node import VFXMatteRefinerNode, CompMatteRefinerNode
from .vfx_mask_export_node import VFXMaskExportNode, CompMatteExportNode
from .vfx_all_in_one_node import VFXKeyingViTAllInOneNode, CompMatteAllInOneNode

__all__ = [
    # Primary CompMatte Node Classes
    "CompMatteAllInOneNode",
    "CompMatteCleanPlateNode",
    "CompMatteKeyerNode",
    "CompMatteViTEdgeNode",
    "CompMatteRefinerNode",
    "CompMatteExportNode",
    # Legacy Aliases
    "IBKCleanPlateNode",
    "IBKKeyerNode",
    "ViTMaskExtractorNode",
    "VFXMatteRefinerNode",
    "VFXMaskExportNode",
    "VFXKeyingViTAllInOneNode",
]
