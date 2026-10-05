"""
tests/test_compmatte.py
Unit tests verifying CompMatte package re-exports, node aliases, and compositing pipeline integrity.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestCompMatteReExports:
    """Verify compmatte_core module exports and compositing aliases."""

    def test_compmatte_core_imports(self):
        import compmatte_core as cm

        assert hasattr(cm, "CompMatteCoreEngine")
        assert hasattr(cm, "CompMatteEdgeEngine")
        assert hasattr(cm, "CompMatteFusionEngine")
        assert hasattr(cm, "CompMatteFusionConfig")
        assert hasattr(cm, "CompMatteIO")
        assert hasattr(cm, "CoreEngine")
        assert hasattr(cm, "ViTMatteEngine")
        assert hasattr(cm, "IBKEngine")

    def test_compmatte_nodes_imports(self):
        from nodes import (
            CompMatteAllInOneNode,
            CompMatteCleanPlateNode,
            CompMatteKeyerNode,
            CompMatteViTEdgeNode,
            CompMatteRefinerNode,
            CompMatteExportNode,
        )

        all_in_one = CompMatteAllInOneNode()
        assert "input_video" in all_in_one.parameters
        assert "use_vitmatte_refinement" in all_in_one.parameters
        assert "enable_edge_jitter_filter" in all_in_one.parameters

        clean = CompMatteCleanPlateNode()
        assert "clean_plate_dir" in clean.parameters

        keyer = CompMatteKeyerNode()
        assert "alpha_matte_dir" in keyer.parameters

        vit = CompMatteViTEdgeNode()
        assert "vit_alpha_dir" in vit.parameters

        refiner = CompMatteRefinerNode()
        assert "stabilized_matte_dir" in refiner.parameters
        assert "enable_edge_jitter_filter" in refiner.parameters

        exporter = CompMatteExportNode()
        assert "exported_dir" in exporter.parameters

    def test_compmatte_root_init_imports(self):
        import __init__ as root_pkg

        assert hasattr(root_pkg, "CompMatteAllInOneNode")
        assert hasattr(root_pkg, "CompMatteCleanPlateNode")
        assert hasattr(root_pkg, "CompMatteKeyerNode")
        assert hasattr(root_pkg, "CompMatteViTEdgeNode")
        assert hasattr(root_pkg, "CompMatteRefinerNode")
        assert hasattr(root_pkg, "CompMatteExportNode")
        # Legacy aliases
        assert hasattr(root_pkg, "VFXKeyingViTAllInOneNode")
        assert hasattr(root_pkg, "IBKCleanPlateNode")

    def test_compmatte_fusion_edge_re_injection(self):
        """Verify CompMatte compositing detail re-injection via CompMatteFusionEngine."""
        from compmatte_core import CompMatteFusionEngine, CompMatteFusionConfig

        config = CompMatteFusionConfig(restore_fine_edges=True, edge_restore_band_radius=40)
        fusion = CompMatteFusionEngine(config=config)

        h, w = 100, 100
        core = np.zeros((h, w), dtype=np.float32)
        core[30:70, 30:70] = 1.0

        # Raw fine edge detail outside core
        raw_edge = np.zeros((h, w), dtype=np.float32)
        raw_edge[25, 25:35] = 0.85  # Fine hair strand

        # Refined matte that erroneously clipped the hair strand
        refined = core.copy()

        result = fusion.inject_fine_edge_detail(refined, raw_edge, core)

        # The hair strand must be 100% recovered
        np.testing.assert_allclose(result[25, 25:35], 0.85, atol=1e-5)

    def test_compmatte_edge_jitter_filtering(self):
        """Verify CompMatte unknown zone geometric jitter filtering."""
        from compmatte_core import CompMatteFusionEngine, CompMatteFusionConfig

        config = CompMatteFusionConfig(enable_edge_jitter_filter=True)
        fusion = CompMatteFusionEngine(config=config)

        h, w = 60, 60
        core = np.zeros((h, w), dtype=np.float32)
        core[20:40, 20:40] = 1.0

        env = np.zeros((h, w), dtype=np.float32)
        env[10:50, 10:50] = 1.0

        base = core.copy()
        # 1px hair strand touching core
        base[14:20, 30] = 0.90
        # 1px isolated noise dot in unknown zone
        base[15, 15] = 0.65

        cleaned = fusion.filter_edge_jitter(base, core_mask=core, envelope_mask=env)

        # Hair strand must remain 0.90
        np.testing.assert_allclose(cleaned[14:20, 30], 0.90)
        # Noise must be wiped to 0.0
        assert cleaned[15, 15] == 0.0
        # Core must stay 1.0
        assert np.all(cleaned[20:40, 20:40] == 1.0)

