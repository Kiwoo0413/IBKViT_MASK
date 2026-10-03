"""
VFX IBK & ViT Masking Custom Nodes for Griptape Nodes Desktop.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_NODES_DIR = Path(__file__).resolve().parent / "nodes"


def _load_node(file_name: str, class_name: str):
    node_file = _NODES_DIR / file_name
    spec = importlib.util.spec_from_file_location(f"ibkvit_nodes_{node_file.stem}", str(node_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load node module from {node_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


IBKCleanPlateNode = _load_node("ibk_clean_plate_node.py", "IBKCleanPlateNode")
IBKKeyerNode = _load_node("ibk_keyer_node.py", "IBKKeyerNode")
ViTMaskExtractorNode = _load_node("vit_mask_extractor_node.py", "ViTMaskExtractorNode")
VFXMatteRefinerNode = _load_node("vfx_matte_refiner_node.py", "VFXMatteRefinerNode")
VFXMaskExportNode = _load_node("vfx_mask_export_node.py", "VFXMaskExportNode")
VFXKeyingViTAllInOneNode = _load_node("vfx_all_in_one_node.py", "VFXKeyingViTAllInOneNode")

__all__ = [
    "IBKCleanPlateNode",
    "IBKKeyerNode",
    "ViTMaskExtractorNode",
    "VFXMatteRefinerNode",
    "VFXMaskExportNode",
    "VFXKeyingViTAllInOneNode",
]
