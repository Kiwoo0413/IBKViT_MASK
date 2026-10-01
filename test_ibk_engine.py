"""
tests/test_ibk_engine.py
Unit tests for VFX IBK Engine (Clean Plate, Color Difference, Matte Pulling).
Runs on CPU without external model dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ibk_engine import IBKEngine, ScreenType, IBKMatteResult


class TestIBKEngine:
    """Test IBK mathematical operations."""

    def setup_method(self):
        self.engine = IBKEngine(screen_type=ScreenType.GREEN, red_weight=0.5, blue_weight=0.5)

    def test_screen_difference_green(self):
        # Create a 10x10 pure green image: R=0, G=255, B=0
        green_img = np.zeros((10, 10, 3), dtype=np.uint8)
        green_img[:, :, 1] = 255

        diff = self.engine.compute_screen_difference(green_img)
        assert diff.shape == (10, 10)
        # Green diff = 1.0 - (0.5*0 + 0.5*0) = 1.0
        assert np.allclose(diff, 1.0, atol=1e-3)

        # Create a white image: R=255, G=255, B=255
        white_img = np.full((10, 10, 3), 255, dtype=np.uint8)
        diff_white = self.engine.compute_screen_difference(white_img)
        # White diff = 1.0 - (0.5*1.0 + 0.5*1.0) = 0.0
        assert np.allclose(diff_white, 0.0, atol=1e-3)

        # Create a red image: R=255, G=0, B=0
        red_img = np.zeros((10, 10, 3), dtype=np.uint8)
        red_img[:, :, 0] = 255
        diff_red = self.engine.compute_screen_difference(red_img)
        # Red diff = 0.0 - 0.5*1.0 = -0.5
        assert np.allclose(diff_red, -0.5, atol=1e-3)

    def test_clean_plate_generation(self):
        h, w = 64, 64
        plate = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            plate[y, :, 1] = int(200 + (y / h) * 50)

        import cv2
        cv2.circle(plate, (32, 32), 14, (200, 20, 20), -1)

        clean_plate = self.engine.generate_clean_plate(
            plate, patch_size=5, blur_radius=7, iterations=3
        )

        assert clean_plate.shape == (h, w, 3)
        center_g = clean_plate[32, 32, 1]
        center_r = clean_plate[32, 32, 0]
        assert center_g > 150
        assert center_r < 100

    def test_matte_pulling(self):
        h, w = 32, 32
        clean_plate = np.zeros((h, w, 3), dtype=np.uint8)
        clean_plate[:, :, 1] = 255

        plate = np.zeros((h, w, 3), dtype=np.uint8)
        plate[:, :16, 1] = 255  # Green screen -> Should be alpha 0.0 (pure black)
        plate[:, 16:, 0] = 255  # Red object -> Should be alpha 1.0 (pure white)

        alpha, _, _, _ = self.engine.pull_matte(plate, clean_plate=clean_plate)

        assert alpha.shape == (h, w)
        # Background must be strictly pure black 0.0
        assert np.all(alpha[:, :15] == 0.0)
        # Foreground must be strictly pure white 1.0
        assert np.all(alpha[:, 17:] == 1.0)

    def test_pure_black_and_white_locks(self):
        # Verify noise clamping
        h, w = 16, 16
        clean_plate = np.full((h, w, 3), [0, 255, 0], dtype=np.uint8)
        noisy_plate = np.full((h, w, 3), [2, 253, 2], dtype=np.uint8)

        alpha, _, _, _ = self.engine.pull_matte(noisy_plate, clean_plate=clean_plate, black_clip=0.02)
        # Residual screen noise must be clamped to 0.0
        assert np.all(alpha == 0.0)

    def test_full_execute_keying(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 1] = 255
        img[10:22, 10:22, :] = [200, 30, 30]

        result = self.engine.execute_keying(img)
        assert isinstance(result, IBKMatteResult)
        assert result.alpha.shape == (32, 32)
        assert result.clean_plate.shape == (32, 32, 3)
