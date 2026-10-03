"""
tests/test_nodes.py
Unit tests for Griptape custom node declarations and parameter schemas.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes import (
    IBKCleanPlateNode,
    IBKKeyerNode,
    ViTMaskExtractorNode,
    VFXMatteRefinerNode,
    VFXMaskExportNode,
    VFXKeyingViTAllInOneNode,
)
from ibkvit_core.griptape_compat import ParameterMode


class TestNodesSchema:
    """Verify parameters, defaults, and modes for all nodes."""

    def test_ibk_clean_plate_node(self):
        node = IBKCleanPlateNode()
        assert "input_video" in node.parameters
        assert "screen_type" in node.parameters
        assert "clean_plate_dir" in node.parameters
        assert "clean_plate_video" in node.parameters

        assert ParameterMode.INPUT in node.parameters["input_video"].allowed_modes
        assert ParameterMode.OUTPUT in node.parameters["clean_plate_dir"].allowed_modes

    def test_ibk_keyer_node(self):
        node = IBKKeyerNode()
        assert "input_video" in node.parameters
        assert "alpha_matte_dir" in node.parameters
        assert "output_resolution" in node.parameters
        assert node.parameters["output_resolution"].default_value == "4k"
        assert "despill_method" not in node.parameters

    def test_vit_mask_extractor_node(self):
        node = ViTMaskExtractorNode()
        assert "seed_coords" in node.parameters
        assert "box_coords" in node.parameters
        assert "core_mask_dir" in node.parameters
        assert "vit_alpha_dir" in node.parameters
        assert "preview_video_path" in node.parameters

    def test_vfx_matte_refiner_node(self):
        node = VFXMatteRefinerNode()
        assert "edge_matte_dir" in node.parameters
        assert "core_matte_dir" in node.parameters
        assert "input_video" in node.parameters
        assert "auto_detect_polarity" in node.parameters
        assert "invert_matte" in node.parameters
        assert "temporal_smoothing" in node.parameters
        assert "stabilized_matte_dir" in node.parameters
        assert "raw_matte_dir" in node.parameters
        assert "stabilized_video_path" in node.parameters
        assert "raw_video_path" in node.parameters
        assert "output_resolution" in node.parameters
        assert node.parameters["output_resolution"].default_value == "4k"

    def test_vfx_mask_export_node(self):
        node = VFXMaskExportNode()
        assert "matte_dir" in node.parameters
        assert "export_format" in node.parameters
        assert "exported_dir" in node.parameters
        assert "output_resolution" in node.parameters
        assert node.parameters["output_resolution"].default_value == "4k"

    def test_vfx_all_in_one_node(self):
        node = VFXKeyingViTAllInOneNode()
        assert "input_video" in node.parameters
        assert "screen_type" in node.parameters
        assert "auto_detect_polarity" in node.parameters
        assert "invert_matte" in node.parameters
        assert "temporal_smoothing" in node.parameters
        assert "output_resolution" in node.parameters
        assert node.parameters["output_resolution"].default_value == "4k"
        assert "stabilized_sequence_dir" in node.parameters
        assert "raw_sequence_dir" in node.parameters
        assert "stabilized_video_path" in node.parameters
        assert "raw_video_path" in node.parameters
        assert "red_overlay_video_path" in node.parameters
        assert "despill_method" not in node.parameters

    def test_vfx_all_in_one_node_process(self, tmp_path):
        import cv2
        import numpy as np
        from ibkvit_core.io_utils import VideoIO

        # Create a synthetic 2-frame MP4 video
        video_path = tmp_path / "greenscreen_shot.mp4"
        frames = []
        for _ in range(2):
            f = np.zeros((72, 128, 3), dtype=np.uint8)
            f[:, :] = [10, 220, 20]  # Green screen
            f[20:50, 40:80] = [210, 60, 40]  # Subject
            frames.append(f)
        VideoIO.write_video(frames, video_path, fps=24.0)

        node = VFXKeyingViTAllInOneNode()
        node.set_parameter_value("input_video", str(video_path))
        node.set_parameter_value("screen_type", "green")
        node.set_parameter_value("output_resolution", "native")
        node.set_parameter_value("export_format", "png16")
        node.set_parameter_value("max_frames", 2)

        # Execute process() - must not throw unsupported operand type(s) for /: 'str' and 'str'
        node.process()

        stab_dir = node.get_parameter_value("stabilized_sequence_dir")
        raw_dir = node.get_parameter_value("raw_sequence_dir")
        stab_vid = node.get_parameter_value("stabilized_video_path")
        raw_vid = node.get_parameter_value("raw_video_path")
        red_vid = node.get_parameter_value("red_overlay_video_path")

        assert Path(stab_dir).exists()
        assert Path(raw_dir).exists()
        assert Path(stab_vid).exists()
        assert Path(raw_vid).exists()
        # Verify dynamic resolution in video parent directory
        assert Path(stab_dir).parent.parent == tmp_path / "greenscreen_shot_masks_4k"

        # Verify exported sequence is strictly single-channel grayscale 16-bit PNG
        first_png = list(Path(stab_dir).glob("*.png"))[0]
        img = cv2.imread(str(first_png), cv2.IMREAD_UNCHANGED)
        assert img.ndim == 2, f"Expected 1-channel grayscale, got shape {img.shape}"
        assert img.dtype == np.uint16, f"Expected uint16, got dtype {img.dtype}"

    def test_vfx_all_in_one_node_process_exr(self, tmp_path):
        import numpy as np
        from ibkvit_core.io_utils import VideoIO

        video_path = tmp_path / "greenscreen_shot2.mp4"
        frames = []
        for _ in range(2):
            f = np.zeros((36, 64, 3), dtype=np.uint8)
            f[:, :] = [10, 220, 20]
            f[10:26, 20:44] = [210, 60, 40]
            frames.append(f)
        VideoIO.write_video(frames, video_path, fps=24.0)

        node = VFXKeyingViTAllInOneNode()
        node.set_parameter_value("input_video", str(video_path))
        node.set_parameter_value("export_format", "exr")
        node.set_parameter_value("output_resolution", "native")
        node.set_parameter_value("max_frames", 2)

        node.process()

        stab_dir = node.get_parameter_value("stabilized_sequence_dir")
        first_exr = list(Path(stab_dir).glob("*.exr"))[0]
        assert first_exr.exists()
        assert first_exr.stat().st_size > 0


    def test_vfx_matte_refiner_node_process(self, tmp_path):
        import numpy as np
        import cv2
        from ibkvit_core.io_utils import VideoIO

        video_path = tmp_path / "test_shot.mp4"
        f = np.zeros((36, 64, 3), dtype=np.uint8)
        f[:, :] = [10, 220, 20]
        VideoIO.write_video([f], video_path, fps=24.0)

        edge_dir = tmp_path / "edge_mattes"
        edge_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(edge_dir / "edge.00000.png"), np.full((36, 64), 128, dtype=np.uint8))

        node = VFXMatteRefinerNode()
        node.set_parameter_value("edge_matte_dir", str(edge_dir))
        node.set_parameter_value("input_video", str(video_path))
        node.set_parameter_value("screen_type", "green")
        node.set_parameter_value("output_resolution", "native")

        # Execute process()
        node.process()

        stab_dir = node.get_parameter_value("stabilized_matte_dir")
        raw_dir = node.get_parameter_value("raw_matte_dir")
        assert Path(stab_dir).exists()
        assert Path(raw_dir).exists()
