"""
tests/conftest.py
Pytest configuration ensuring OpenCV OpenEXR is enabled before importing cv2.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
