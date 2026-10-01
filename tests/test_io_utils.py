"""
tests/test_io_utils.py
Unit tests for Video Frame and Image Sequence I/O.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.io_utils import ImageSequenceIO, VideoIO


class TestIOUtils:
    def test_exr_and_png_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            m = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)

            # Test EXR save
            exr_path = tmp / "test.exr"
            ImageSequenceIO.save_exr(m, exr_path, is_alpha_only=True)
            assert exr_path.exists()
            assert exr_path.stat().st_size > 0

            # Test 16-bit PNG save
            png_path = tmp / "test_16.png"
            ImageSequenceIO.save_png_16bit(m, png_path)
            assert png_path.exists()
            assert png_path.stat().st_size > 0

    def test_sequence_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seq = [np.ones((16, 16), dtype=np.float32) * (i / 5.0) for i in range(5)]

            files = ImageSequenceIO.export_sequence(seq, tmp / "seq", prefix="test_matte", format_type="exr")
            assert len(files) == 5
            for f in files:
                assert Path(f).exists()
