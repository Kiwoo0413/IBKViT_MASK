"""
tests/test_matte_fusion.py
Unit tests for Matte Fusion, Polarity Auto-Correction, Dual (Stabilized vs Raw) Outputs, and Dynamic Folder Resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ibkvit_core.matte_fusion import MatteFusionEngine, FusionConfig
from ibkvit_core.io_utils import VideoIO


class TestMatteFusion:
    def setup_method(self):
        self.engine = MatteFusionEngine(FusionConfig(temporal_smoothing_alpha=0.3))

    def test_core_and_edge_fusion(self):
        core = np.zeros((32, 32), dtype=np.float32)
        core[10:22, 10:22] = 1.0

        edge = np.zeros((32, 32), dtype=np.float32)
        edge[6:26, 6:26] = 0.5

        fused = self.engine.fuse_core_and_edge(core, edge)
        assert np.all(fused[10:22, 10:22] == 1.0)
        assert np.all(fused[7:9, 7:9] == 0.5)
        assert np.all(fused[:5, :5] == 0.0)

    def test_envelope_fusion(self):
        # Inner core (pure white)
        core = np.zeros((32, 32), dtype=np.float32)
        core[12:20, 12:20] = 1.0

        # Outer envelope (covers transition zone)
        env = np.zeros((32, 32), dtype=np.float32)
        env[8:24, 8:24] = 1.0

        # IBK pristine transmission edge
        edge = np.full((32, 32), 0.45, dtype=np.float32)

        fused = self.engine.fuse_core_and_edge(core, edge, envelope_matte=env)
        # Inside core: must be strictly 1.0
        assert np.all(fused[12:20, 12:20] == 1.0)
        # In transition band (in envelope, outside core): must be exact IBK edge transmission
        assert np.allclose(fused[9:11, 9:11], 0.45)
        # Outside envelope: must be strictly 0.0 (pure black background)
        assert np.all(fused[:5, :5] == 0.0)
        assert np.all(fused[26:, 26:] == 0.0)

    def test_core_hole_filling(self):
        core = np.zeros((32, 32), dtype=np.float32)
        core[8:24, 8:24] = 1.0
        core[14:18, 14:18] = 0.0

        filled = MatteFusionEngine.fill_core_holes(core)
        assert np.all(filled[14:18, 14:18] == 1.0)

    def test_upscale_to_4k(self):
        m = np.zeros((9, 16), dtype=np.float32)
        m[3:6, 5:11] = 1.0

        scaled_4k = MatteFusionEngine.upscale_to_4k(m, target_width=3840, preserve_aspect=True)
        assert scaled_4k.shape[1] == 3840
        assert scaled_4k.shape[0] == 2160
        assert scaled_4k[1080, 1920] == 1.0
        assert scaled_4k[100, 100] == 0.0

    def test_detect_and_correct_polarity_inverted(self):
        # Create a synthetic frame: Green screen background, Red/Brown subject in center
        h, w = 64, 64
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = [20, 220, 30]  # Green screen background
        frame[20:44, 20:44] = [210, 80, 60]  # Subject in center

        # Intentionally create an inverted matte:
        # Subject is 0.0, Green background is 1.0
        inverted_matte = np.ones((h, w), dtype=np.float32)
        inverted_matte[20:44, 20:44] = 0.0

        corrected, was_inverted = MatteFusionEngine.detect_and_correct_polarity(
            matte=inverted_matte,
            rgb_frame=frame,
            screen_type="green",
        )

        assert was_inverted is True
        # After correction, subject must be 1.0, background must be 0.0
        assert np.mean(corrected[20:44, 20:44]) > 0.95
        assert np.mean(corrected[:10, :10]) < 0.05

    def test_detect_and_correct_polarity_correct_already(self):
        # Correct matte: Subject is 1.0, Green background is 0.0
        h, w = 64, 64
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = [20, 220, 30]
        frame[20:44, 20:44] = [210, 80, 60]

        correct_matte = np.zeros((h, w), dtype=np.float32)
        correct_matte[20:44, 20:44] = 1.0

        corrected, was_inverted = MatteFusionEngine.detect_and_correct_polarity(
            matte=correct_matte,
            rgb_frame=frame,
            screen_type="green",
        )

        assert was_inverted is False
        assert np.mean(corrected[20:44, 20:44]) > 0.95
        assert np.mean(corrected[:10, :10]) < 0.05

    def test_retrack_object_saliency(self):
        h, w = 64, 64
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = [10, 240, 20]  # Bright green screen
        frame[20:44, 20:44] = [180, 50, 50]  # Subject

        saliency = MatteFusionEngine.retrack_object_saliency(frame, screen_type="green")
        assert saliency.shape == (h, w)
        assert np.mean(saliency[24:40, 24:40]) > 0.9  # Subject center solid
        assert np.mean(saliency[:10, :10]) < 0.1  # Background screen empty

    def test_process_frame_dual(self):
        engine = MatteFusionEngine(FusionConfig(temporal_smoothing_alpha=0.5))

        h, w = 36, 64  # 16:9 aspect ratio -> upscales to exactly 3840x2160
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = [15, 230, 25]  # Green screen
        frame[10:26, 20:45] = [200, 60, 50]  # Subject

        core = np.zeros((h, w), dtype=np.float32)
        core[12:24, 22:43] = 1.0

        edge = np.zeros((h, w), dtype=np.float32)
        edge[9:27, 19:46] = 0.6

        # Frame 0
        stabilized_0, raw_0 = engine.process_frame_dual(
            core_matte=core,
            edge_matte=edge,
            rgb_guide=frame,
            screen_type="green",
            scale_to_4k=True,
        )
        assert stabilized_0.shape == (2160, 3840)
        assert raw_0.shape == (2160, 3840)

        # Perturb edge in Frame 1 to simulate high-frequency jitter/noise
        edge_jitter = edge.copy()
        edge_jitter[6:10, 15:20] = 0.8  # Sudden noise flick

        stabilized_1, raw_1 = engine.process_frame_dual(
            core_matte=core,
            edge_matte=edge_jitter,
            rgb_guide=frame,
            screen_type="green",
            scale_to_4k=True,
        )

        # Raw must reflect immediate frame change (no temporal EMA)
        # Stabilized must be smoothed by previous frame history
        assert not np.array_equal(stabilized_1, raw_1)
        # Both must preserve 100% pure white solid core and pure black background
        assert np.all(raw_1[1000:1100, 1900:2000] == 1.0)
        assert np.all(stabilized_1[1000:1100, 1900:2000] == 1.0)
        assert np.all(raw_1[:200, :200] == 0.0)
        assert np.all(stabilized_1[:200, :200] == 0.0)

    def test_dynamic_output_dir_resolution(self, tmp_path):
        dummy_video = tmp_path / "footage" / "act1_shot01.mp4"
        dummy_video.parent.mkdir(parents=True, exist_ok=True)
        dummy_video.touch()

        # When custom_output_dir is empty, dynamically resolve in video parent dir
        resolved = VideoIO.resolve_output_dir(str(dummy_video), custom_output_dir="", subfolder_suffix="masks_4k")
        expected = (tmp_path / "footage" / "act1_shot01_masks_4k").resolve()
        assert Path(resolved) == expected

        # When custom_output_dir is provided, honor custom directory
        custom_target = str(tmp_path / "custom_exports")
        resolved_custom = VideoIO.resolve_output_dir(str(dummy_video), custom_output_dir=custom_target)
        assert Path(resolved_custom) == Path(custom_target).resolve()
