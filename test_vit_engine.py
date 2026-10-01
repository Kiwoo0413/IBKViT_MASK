"""
tests/test_vit_engine.py
Unit tests for ViT Engine trimap generation and matting.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.vit_engine import ViTEngine


class TestViTEngine:
    def test_trimap_generation(self):
        # Create a 64x64 square mask in center
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255

        trimap = ViTEngine.generate_trimap(mask, erode_kernel_size=5, dilate_kernel_size=5)

        assert trimap.shape == (64, 64)
        # Unique values should only be in {0, 128, 255}
        unique_vals = set(np.unique(trimap))
        assert unique_vals.issubset({0, 128, 255})

        # Deep center must be solid foreground 255
        assert trimap[32, 32] == 255
        # Far corner must be background 0
        assert trimap[2, 2] == 0
        # Perimeter must be unknown transition 128
        assert trimap[18, 32] == 128 or trimap[45, 32] == 128

    def test_guided_matting_fallback(self):
        engine = ViTEngine(device="cpu")
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        trimap = np.zeros((32, 32), dtype=np.uint8)
        trimap[10:22, 10:22] = 255
        trimap[8:10, :] = 128

        alpha = engine.predict_vitmatte(img, trimap)
        assert alpha.shape == (32, 32)
        assert alpha[16, 16] == 1.0
        assert alpha[0, 0] == 0.0

    def test_extract_core_and_envelope(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255

        core, env = ViTEngine.extract_core_and_envelope(mask, erode_kernel_size=5, dilate_kernel_size=5)
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

        mask = ViTEngine.detect_subject_coarse_mask(frame, screen_type="green")
        assert mask.shape == (64, 64)
        assert np.mean(mask[22:42, 22:42]) > 0.9
        assert np.mean(mask[:10, :10]) < 0.1

    def test_stabilize_mask_sequence(self):
        # Create sequence with high-frequency boundary jitter
        m0 = np.zeros((32, 32), dtype=np.float32)
        m0[10:22, 10:22] = 1.0

        m1 = m0.copy()
        m1[9:11, 9:11] = 1.0  # Jitter on corner

        seq = [m0, m1]
        stab_core = ViTEngine.stabilize_mask_sequence(seq, temporal_factor=0.5, is_core=True)
        assert len(stab_core) == 2
        # Deep center remains rock-solid pure white
        assert stab_core[1][16, 16] == 1.0
        assert stab_core[1][0, 0] == 0.0
