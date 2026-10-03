"""
nodes package for VFX IBK Keying & ViT Masking Custom Nodes for Griptape Nodes Desktop.
"""

import sys
from pathlib import Path

_LIB_ROOT = Path(__file__).resolve().parent.parent
if str(_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIB_ROOT))

from .ibk_clean_plate_node import IBKCleanPlateNode
from .ibk_keyer_node import IBKKeyerNode
from .vit_mask_extractor_node import ViTMaskExtractorNode
from .vfx_matte_refiner_node import VFXMatteRefinerNode
from .vfx_mask_export_node import VFXMaskExportNode
from .vfx_all_in_one_node import VFXKeyingViTAllInOneNode

__all__ = [
    "IBKCleanPlateNode",
    "IBKKeyerNode",
    "ViTMaskExtractorNode",
    "VFXMatteRefinerNode",
    "VFXMaskExportNode",
    "VFXKeyingViTAllInOneNode",
]
