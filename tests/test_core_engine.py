"""
tests/test_core_engine.py
Unit tests for CoreEngine trimap generation, core/envelope extraction,
blur profiling, and temporal stabilization.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ibkvit_core.core_engine import CoreEngine, parse_box, parse_coords


class TestCoreEngine:
    def test_parse_coords_and_box(self):
        pts = parse_coords("100, 200")
        assert pts == [(100.0, 200.0)]

        pts_list = parse_coords("[[10, 20], [30, 40]]")
        assert pts_list == [(10.0, 20.0), (30.0, 40.0)]

        box = parse_box("[10, 20, 100, 200]")
        assert box == [10.0, 20.0, 100.0, 200.0]

    def test_trimap_generation(self):
        # Create a 64x64 square mask in center
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255

        trimap = CoreEngine.generate_trimap(mask, erode_kernel_size=5, dilate_kernel_size=5)

        assert trimap.shape == (64, 64)
        unique_vals = set(np.unique(trimap))
        assert unique_vals.issubset({0, 128, 255})

        # Deep center must be solid foreground 255
        assert trimap[32, 32] == 255
        # Far corner must be background 0
        assert trimap[2, 2] == 0
        # Perimeter must be unknown transition 128
        assert trimap[18, 32] == 128 or trimap[45, 32] == 128

    def test_extract_core_and_envelope(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255

        core, env = CoreEngine.extract_core_and_envelope(mask, erode_kernel_size=5, dilate_kernel_size=5)
        assert core.shape == (64, 64)
        assert env.shape == (64, 64)
        # Deep center must be solid pure white core (1.0)
        assert core[32, 32] == 1.0
        # Transition perimeter must be in envelope but outside core
        assert env[18, 32] == 1.0
        assert core[18, 32] == 0.0
        # Background must be strictly pure black (0.0) in both
        assert env[2, 2] == 0.0
        assert core[2, 2] == 0.0

    def test_detect_subject_coarse_mask(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = [20, 220, 30]  # Green screen background
        frame[20:44, 20:44] = [200, 80, 50]  # Foreground subject

        mask = CoreEngine.detect_subject_coarse_mask(frame, screen_type="green")
        assert mask.shape == (64, 64)
        assert np.mean(mask[22:42, 22:42]) > 0.9
        assert np.mean(mask[:10, :10]) < 0.1

    def test_stabilize_mask_sequence(self):
        m0 = np.zeros((32, 32), dtype=np.float32)
        m0[10:22, 10:22] = 1.0

        m1 = m0.copy()
        m1[9:11, 9:11] = 1.0  # Jitter on corner

        seq = [m0, m1]
        stab_core = CoreEngine.stabilize_mask_sequence(seq, temporal_factor=0.5, is_core=True)
        assert len(stab_core) == 2
        # Deep center remains rock-solid pure white
        assert stab_core[1][16, 16] == 1.0
        assert stab_core[1][0, 0] == 0.0

    def test_motion_frame_independence_zero_accumulation(self):
        """Verify that moving subjects leave zero ghosting / residue across consecutive frames."""
        h, w = 128, 128
        f0 = np.zeros((h, w, 3), dtype=np.uint8)
        f0[:, :] = [20, 220, 30]
        f0[30:90, 10:30] = [200, 50, 40]

        f1 = np.zeros((h, w, 3), dtype=np.uint8)
        f1[:, :] = [20, 220, 30]
        f1[30:90, 80:100] = [200, 50, 40]

        engine = CoreEngine()
        stab_cores, stab_envelopes, raw_cores, raw_envelopes = engine.track_and_stabilize_stream(
            frame_sequence=[f0, f1],
            screen_type="green",
            erode_radius=3,
            dilate_radius=5,
        )

        assert np.mean(raw_cores[0][35:85, 12:28]) > 0.9
        assert np.all(raw_cores[0][30:90, 80:100] == 0.0)

        assert np.mean(raw_cores[1][35:85, 82:98]) > 0.9
        assert np.all(raw_cores[1][30:90, 10:30] == 0.0)
        assert np.all(raw_envelopes[1][30:90, 10:30] == 0.0)
        assert np.all(stab_cores[1][30:90, 10:30] == 0.0)

    def test_estimate_edge_blur_profile(self):
        h, w = 128, 128
        sharp_img = np.zeros((h, w, 3), dtype=np.uint8)
        sharp_img[:, :] = [20, 220, 30]
        sharp_img[30:90, 30:90] = [200, 60, 40]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[30:90, 30:90] = 255

        e_sharp, d_sharp, blur_sharp = CoreEngine.estimate_edge_blur_profile(
            sharp_img, mask, screen_type="green", base_erode=10, base_dilate=15
        )
        assert blur_sharp < 0.4
        assert e_sharp <= 10

        blurred_img = cv2.GaussianBlur(sharp_img, (21, 21), 9.0)
        e_blur, d_blur, blur_val = CoreEngine.estimate_edge_blur_profile(
            blurred_img, mask, screen_type="green", base_erode=10, base_dilate=15
        )
        assert blur_val > blur_sharp
        assert d_blur >= d_sharp
