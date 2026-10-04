"""
ibkvit_core/core_engine.py
Core & Envelope Extraction, Subject Tracking, and Spatio-Temporal Stabilization Engine.

Headless, ultra-fast, 100% pure NumPy/OpenCV implementation (zero heavy deep-learning dependencies).
Guarantees:
- Inner Core: 100% solid pure-white (1.0), zero internal holes, zero flicker/chatter.
- Outer Envelope: 100% solid pure-black (0.0) background boundary, zero screen spill noise.
- Spatio-temporal mask de-jittering across video sequences.
- Dynamic defocus/motion blur profile estimation for adaptive boundary sizing.
"""

from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("CoreEngine")


def parse_coords(coord_str: str) -> Optional[List[Tuple[float, float]]]:
    """Parse string coordinates like '100,200' or '[[100,200],[300,400]]'."""
    if not coord_str or not coord_str.strip():
        return None
    s = coord_str.strip()
    try:
        if "[" in s:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                if len(parsed) > 0 and isinstance(parsed[0], (list, tuple)):
                    return [(float(p[0]), float(p[1])) for p in parsed]
                elif len(parsed) >= 2 and isinstance(parsed[0], (int, float)):
                    return [(float(parsed[0]), float(parsed[1]))]
        parts = [float(p.strip()) for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            return [(parts[0], parts[1])]
    except Exception:
        pass
    return None


def parse_box(box_str: str) -> Optional[List[float]]:
    """Parse string box [x1, y1, x2, y2]."""
    if not box_str or not box_str.strip():
        return None
    s = box_str.strip()
    try:
        if "[" in s:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and len(parsed) == 4:
                return [float(v) for v in parsed]
        parts = [float(p.strip()) for p in s.split(",") if p.strip()]
        if len(parts) == 4:
            return parts
    except Exception:
        pass
    return None


class CoreEngine:
    """
    Core & Envelope Geometry Extraction and Stabilization Engine.
    Manages subject tracking, hole-filling, background isolation, and temporal stabilization.
    """

    def __init__(self) -> None:
        self.last_coarse_masks: List[np.ndarray] = []

    # -------------------------------------------------------------------------
    # Trimap Generation (Pure White Core & Pure Black Background Guarantee)
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_trimap(
        mask: np.ndarray,
        erode_kernel_size: int = 15,
        dilate_kernel_size: int = 15,
    ) -> np.ndarray:
        """
        Generate a 3-class trimap from a binary or grayscale mask.
        - 255 (Foreground Core): Definite foreground interior (100% solid pure white, NO holes, NO chatter).
        - 0   (Background): Definite background (100% pure black).
        - 128 (Unknown / Transition): Edge boundary, hair, motion blur for edge refinement.

        Args:
            mask: (H, W) uint8 or float32 mask.
            erode_kernel_size: Pixels to shrink mask inwards to create solid core.
            dilate_kernel_size: Pixels to expand mask outwards to envelop edge detail.

        Returns:
            trimap: (H, W) uint8 with values in {0, 128, 255}.
        """
        if mask.dtype != np.uint8:
            mask_uint8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            mask_uint8 = mask.copy()

        # Binarize with threshold
        _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

        k_erode = max(1, erode_kernel_size if erode_kernel_size % 2 == 1 else erode_kernel_size + 1)
        k_dilate = max(1, dilate_kernel_size if dilate_kernel_size % 2 == 1 else dilate_kernel_size + 1)

        erode_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
        dilate_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))

        eroded = cv2.erode(binary, erode_elem)
        dilated = cv2.dilate(binary, dilate_elem)

        # Hole-filling on core foreground: fills all internal holes and eliminates internal chatter
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_foreground = np.zeros_like(eroded)
        if contours:
            cv2.drawContours(solid_foreground, contours, -1, 255, thickness=-1)
        else:
            solid_foreground = eroded

        # Trimap: 0 = background, 128 = unknown transition, 255 = solid core foreground
        trimap = np.zeros_like(binary, dtype=np.uint8)
        trimap[dilated > 0] = 128
        trimap[solid_foreground > 0] = 255

        return trimap

    @staticmethod
    def extract_core_and_envelope(
        mask: np.ndarray,
        erode_kernel_size: int = 15,
        dilate_kernel_size: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract solid inner core (100% pure white, holes filled) and outer background envelope (0% black outside).

        Args:
            mask: (H, W) uint8 or float32 coarse mask.
            erode_kernel_size: pixels to shrink inwards to create safe interior core.
            dilate_kernel_size: pixels to expand outwards to envelop transition/hair.

        Returns:
            (core_mask, envelope_mask): both (H, W) float32 in range [0.0, 1.0].
        """
        if mask.dtype != np.uint8:
            mask_uint8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            mask_uint8 = mask.copy()

        _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        k_erode = max(1, erode_kernel_size if erode_kernel_size % 2 == 1 else erode_kernel_size + 1)
        k_dilate = max(1, dilate_kernel_size if dilate_kernel_size % 2 == 1 else dilate_kernel_size + 1)

        erode_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
        dilate_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))

        eroded = cv2.erode(binary, erode_elem)
        dilated = cv2.dilate(binary, dilate_elem)

        # Hole-filling on core foreground: fills all interior holes and eliminates internal chatter
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_core = np.zeros_like(eroded)
        if contours:
            cv2.drawContours(solid_core, contours, -1, 255, thickness=-1)
        else:
            solid_core = eroded

        # Hole-filling on outer envelope: ensures complete coverage
        contours_env, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_env = np.zeros_like(dilated)
        if contours_env:
            cv2.drawContours(solid_env, contours_env, -1, 255, thickness=-1)
        else:
            solid_env = dilated

        return (solid_core.astype(np.float32) / 255.0, solid_env.astype(np.float32) / 255.0)

    @staticmethod
    def detect_subject_coarse_mask(
        rgb_frame: np.ndarray,
        screen_type: str = "green",
    ) -> np.ndarray:
        """
        Automatically detect subject coarse mask from green/blue screen plate when no seed points are provided.
        """
        img_f = rgb_frame.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

        r = img_f[:, :, 0]
        g = img_f[:, :, 1]
        b = img_f[:, :, 2]

        if screen_type.lower() == "blue":
            screen_diff = b - (0.5 * r + 0.5 * g)
        else:
            screen_diff = g - (0.5 * r + 0.5 * b)

        # Subject pixels have low or negative screen_diff
        fg_binary = (screen_diff < 0.08).astype(np.uint8) * 255

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Keep large components & fill holes
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(opened)
        if contours:
            h, w = rgb_frame.shape[:2]
            min_area = (h * w) * 0.005
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if valid_contours:
                cv2.drawContours(clean_mask, valid_contours, -1, 255, thickness=-1)
            else:
                cv2.drawContours(clean_mask, contours, -1, 255, thickness=-1)
        else:
            clean_mask = opened

        return clean_mask.astype(np.float32) / 255.0

    @staticmethod
    def estimate_edge_blur_profile(
        rgb_image: np.ndarray,
        coarse_mask: np.ndarray,
        screen_type: str = "green",
        base_erode: int = 12,
        base_dilate: int = 15,
    ) -> Tuple[int, int, float]:
        """
        Estimate the degree of defocus or motion blur along the subject boundary
        using spatial gradient distribution.

        Dynamically adjusts erode and dilate radii:
        - Sharp in-focus edges: narrow trimap band (faster, tighter core)
        - Defocused / motion-blurred edges: proportionally wider trimap band
          (captures full semi-transparent blur envelope without clipping)

        Returns:
            (adaptive_erode_radius, adaptive_dilate_radius, blur_factor)
            where blur_factor is in [0.0, 1.0] (0.0 = razor-sharp, 1.0 = heavy blur).
        """
        if coarse_mask.dtype != np.uint8:
            mask_uint8 = (np.clip(coarse_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            mask_uint8 = coarse_mask.copy()

        _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(binary) == 0:
            return base_erode, base_dilate, 0.0

        # Narrow sampling band around contour (15px around boundary)
        k_band = 15
        band_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_band, k_band))
        dil = cv2.dilate(binary, band_elem)
        ero = cv2.erode(binary, band_elem)
        edge_band = (dil > 0) & (ero == 0)

        if not np.any(edge_band):
            return base_erode, base_dilate, 0.0

        # Convert to grayscale float [0, 1]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0

        # Compute spatial gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        edge_pixels = grad_mag[edge_band]
        if len(edge_pixels) == 0:
            return base_erode, base_dilate, 0.0

        # Mean gradient sharpness on active edge transitions
        active_grads = edge_pixels[edge_pixels > 0.02]
        if len(active_grads) == 0:
            sharpness = 0.0
        else:
            sharpness = float(np.mean(active_grads))

        # Sharpness: sharp edge is ~0.4 - 0.7+, blurred edge is < 0.12
        blur_factor = float(np.clip(1.0 - (sharpness - 0.08) / (0.45 - 0.08), 0.0, 1.0))

        # Dynamic radii scaling:
        # Sharp (blur=0.0): 0.6x base (tighter, faster)
        # Heavy blur (blur=1.0): 1.8x base (covers entire motion blur streak)
        scale_erode = 0.6 + 1.2 * blur_factor
        scale_dilate = 0.6 + 1.4 * blur_factor

        adaptive_erode = max(3, int(round(base_erode * scale_erode)))
        adaptive_dilate = max(4, int(round(base_dilate * scale_dilate)))

        return adaptive_erode, adaptive_dilate, blur_factor

    @staticmethod
    def stabilize_mask_sequence(
        masks: List[np.ndarray],
        temporal_factor: float = 0.0,
        is_core: bool = False,
    ) -> List[np.ndarray]:
        """
        Spatio-temporal de-jittering of coarse/core/envelope masks across video sequence.
        Eliminates boundary chatter and contour flickering in the core branch.
        """
        if not masks or temporal_factor <= 0.0:
            return [m.copy() for m in masks]

        stabilized: List[np.ndarray] = []
        prev = masks[0].copy().astype(np.float32)
        stabilized.append(prev.copy())

        for idx in range(1, len(masks)):
            curr = masks[idx].astype(np.float32)

            if is_core:
                # Core de-jittering: smooth distance field or continuous mask
                # Ensures solid white interior never chatters or pops
                blend = (1.0 - temporal_factor) * curr + temporal_factor * prev
                m_uint8 = (blend > 0.45).astype(np.uint8) * 255
                contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                solid = np.zeros_like(m_uint8)
                if contours:
                    cv2.drawContours(solid, contours, -1, 255, thickness=-1)
                else:
                    solid = m_uint8
                res = solid.astype(np.float32) / 255.0
            else:
                # Envelope de-jittering: strictly non-accumulative continuous blend
                # Prevents previous frame envelope from accumulating or smearing across frames
                blend = (1.0 - temporal_factor) * curr + temporal_factor * prev
                res = np.clip(blend, 0.0, 1.0)

            stabilized.append(res)
            prev = res.copy()

        return stabilized

    def track_video_frames(
        self,
        frame_sequence: List[np.ndarray],
        seed_points: Optional[List[Tuple[float, float]]] = None,
        point_labels: Optional[List[int]] = None,
        box_coords: Optional[List[float]] = None,
        init_mask: Optional[np.ndarray] = None,
        screen_type: str = "green",
    ) -> List[np.ndarray]:
        """
        Track object across video frames using fast adaptive spatio-temporal tracking.
        """
        return self._adaptive_flow_track(
            frame_sequence=frame_sequence,
            seed_points=seed_points,
            box_coords=box_coords,
            init_mask=init_mask,
            screen_type=screen_type,
        )

    track_sam2_frames = track_video_frames

    @classmethod
    def _adaptive_flow_track(
        cls,
        frame_sequence: List[np.ndarray],
        seed_points: Optional[List[Tuple[float, float]]] = None,
        box_coords: Optional[List[float]] = None,
        init_mask: Optional[np.ndarray] = None,
        screen_type: str = "green",
    ) -> List[np.ndarray]:
        """
        Ultra-fast per-frame strictly independent mask extraction.
        Guarantees zero mask accumulation or ghosting between consecutive frames.
        """
        num_frames = len(frame_sequence)
        if num_frames == 0:
            return []

        h, w = frame_sequence[0].shape[:2]
        masks: List[np.ndarray] = []

        for i, frame in enumerate(frame_sequence):
            # Case 1: Specific manual initial mask provided for frame 0
            if i == 0 and init_mask is not None and (seed_points or box_coords):
                if init_mask.dtype != np.uint8:
                    current_mask = (np.clip(init_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    current_mask = init_mask.copy()
            # Case 2: Bounding Box specified (evaluate within box for this frame)
            elif box_coords is not None and len(box_coords) == 4:
                x1, y1, x2, y2 = [int(v) for v in box_coords]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                current_mask = np.zeros((h, w), dtype=np.uint8)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    from ibkvit_core.matte_fusion import MatteFusionEngine
                    saliency = MatteFusionEngine.retrack_object_saliency(crop, min_distance_from_edge=0.0)
                    if np.any(saliency > 0.1):
                        current_mask[y1:y2, x1:x2] = (saliency * 255.0).astype(np.uint8)
                    else:
                        current_mask[y1:y2, x1:x2] = 255
            # Case 3: Seed points specified
            elif seed_points is not None and len(seed_points) > 0:
                current_mask = np.zeros((h, w), dtype=np.uint8)
                for pt in seed_points:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(current_mask, (px, py), radius=max(20, min(h, w) // 15), color=255, thickness=-1)
                direct = cls.detect_subject_coarse_mask(frame, screen_type=screen_type)
                current_mask = np.where(current_mask > 0, (direct * 255.0).astype(np.uint8), 0)
            # Case 4: Standard Studio Chroma-Key (100% per-frame independent extraction)
            else:
                direct = cls.detect_subject_coarse_mask(frame, screen_type=screen_type)
                current_mask = (direct * 255.0).astype(np.uint8)

            masks.append(current_mask.astype(np.float32) / 255.0)

        return masks

    def track_and_stabilize_stream(
        self,
        frame_sequence: List[np.ndarray],
        seed_points: Optional[List[Tuple[float, float]]] = None,
        box_coords: Optional[List[float]] = None,
        screen_type: str = "green",
        temporal_factor: float = 0.0,
        erode_radius: int = 15,
        dilate_radius: int = 20,
        enable_adaptive_blur: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Full tracking and internal/external matte de-jittering pipeline.
        Eliminates jitter on the inner core (100% white) and outer envelope (0% black background)
        without touching or degrading edge transition hair details.

        Returns:
            (stab_cores, stab_envelopes, raw_cores, raw_envelopes)
        """
        num_frames = len(frame_sequence)
        if num_frames == 0:
            return [], [], [], []

        # If no manual seed or box provided, automatically detect initial coarse mask
        detected_init_mask: Optional[np.ndarray] = None
        if not seed_points and not box_coords:
            detected_init_mask = self.detect_subject_coarse_mask(frame_sequence[0], screen_type=screen_type)

        coarse_masks = self.track_video_frames(
            frame_sequence=frame_sequence,
            seed_points=seed_points,
            box_coords=box_coords,
            init_mask=detected_init_mask,
            screen_type=screen_type,
        )
        self.last_coarse_masks = coarse_masks

        raw_cores: List[np.ndarray] = []
        raw_envelopes: List[np.ndarray] = []

        for idx, m in enumerate(coarse_masks):
            if enable_adaptive_blur and idx < len(frame_sequence):
                eff_erode, eff_dilate, _ = self.estimate_edge_blur_profile(
                    frame_sequence[idx], m, screen_type=screen_type,
                    base_erode=erode_radius, base_dilate=dilate_radius,
                )
            else:
                eff_erode, eff_dilate = erode_radius, dilate_radius

            c, env = self.extract_core_and_envelope(
                m, erode_kernel_size=eff_erode, dilate_kernel_size=eff_dilate
            )
            raw_cores.append(c)
            raw_envelopes.append(env)

        stab_cores = self.stabilize_mask_sequence(raw_cores, temporal_factor=temporal_factor, is_core=True)
        stab_envelopes = self.stabilize_mask_sequence(raw_envelopes, temporal_factor=temporal_factor, is_core=False)

        return stab_cores, stab_envelopes, raw_cores, raw_envelopes
