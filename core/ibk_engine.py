"""
core/ibk_engine.py
VFX-grade Image Based Keyer (IBK) Engine for Alpha Matte Extraction.
Implements IBKColour (Clean Plate / Screen generation) and
IBKGizmo (Color Difference transmission matte pulling).
Focuses purely on high-detail alpha matte extraction (hair, motion blur, transparency).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class ScreenType(str, Enum):
    GREEN = "green"
    BLUE = "blue"
    CUSTOM = "custom"


@dataclass
class IBKMatteResult:
    """Result of IBK Keying operation."""
    alpha: np.ndarray             # float32 [0.0, 1.0] single channel (H, W)
    clean_plate: np.ndarray       # uint8 [0..255] (H, W, 3) in RGB
    screen_diff_fg: np.ndarray    # float32 (H, W) foreground color difference
    screen_diff_clean: np.ndarray # float32 (H, W) clean plate color difference


class IBKEngine:
    """
    Image Based Keyer (IBK) Algorithm Engine.
    Compatible with Nuke IBKColour / IBKGizmo compositing workflows.
    """

    def __init__(
        self,
        screen_type: ScreenType | str = ScreenType.GREEN,
        red_weight: float = 0.5,
        blue_weight: float = 0.5,
        green_weight: float = 0.5,
        custom_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        if isinstance(screen_type, str):
            screen_type = ScreenType(screen_type.lower())
        self.screen_type = screen_type
        self.red_weight = red_weight
        self.blue_weight = blue_weight
        self.green_weight = green_weight
        self.custom_color = custom_color or (0, 255, 0)

    # -------------------------------------------------------------------------
    # Screen Difference Calculation
    # -------------------------------------------------------------------------

    def compute_screen_difference(
        self,
        rgb_image: np.ndarray,
        screen_type: Optional[ScreenType] = None,
        red_weight: Optional[float] = None,
        blue_weight: Optional[float] = None,
        green_weight: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute color difference map.
        Green Screen: diff = G - (w_r * R + w_b * B)
        Blue Screen:  diff = B - (w_r * R + w_g * G)
        Custom: normalized difference based on custom screen color vector.

        Args:
            rgb_image: (H, W, 3) image in RGB, uint8 or float32.

        Returns:
            diff: (H, W) float32 difference map. Positive values indicate screen color.
        """
        st = screen_type or self.screen_type
        wr = self.red_weight if red_weight is None else red_weight
        wb = self.blue_weight if blue_weight is None else blue_weight
        wg = self.green_weight if green_weight is None else green_weight

        img_f = rgb_image.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

        r = img_f[:, :, 0]
        g = img_f[:, :, 1]
        b = img_f[:, :, 2]

        if st == ScreenType.GREEN:
            weight_comp = wr * r + wb * b
            diff = g - weight_comp
        elif st == ScreenType.BLUE:
            weight_comp = wr * r + wg * g
            diff = b - weight_comp
        elif st == ScreenType.CUSTOM:
            target = np.array(self.custom_color, dtype=np.float32) / 255.0
            norm = np.linalg.norm(target) + 1e-6
            target_unit = target / norm
            proj = (img_f * target_unit).sum(axis=-1)
            intensity = (r + g + b) / 3.0
            diff = proj - intensity
        else:
            raise ValueError(f"Unsupported screen type: {st}")

        return diff.astype(np.float32)

    # -------------------------------------------------------------------------
    # Clean Plate Generator (IBKColour emulation)
    # -------------------------------------------------------------------------

    def generate_clean_plate(
        self,
        rgb_image: np.ndarray,
        patch_size: int = 5,
        blur_radius: int = 15,
        iterations: int = 4,
        darks: float = 0.0,
        lights: float = 1.0,
        fg_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate synthetic Clean Plate (backing screen image) from original footage.
        Simulates Nuke's IBKColour patch levels by removing foreground subjects and
        expanding the backing screen gradient across the frame.

        Args:
            rgb_image: (H, W, 3) uint8 or float32 image in RGB format.
            patch_size: Morphological kernel size for growing screen color.
            blur_radius: Gaussian blur sigma to smooth the clean screen lighting.
            iterations: Number of hierarchical patch expansion iterations.
            darks: Lower threshold clamp for screen luminance [0.0..1.0].
            lights: Upper threshold clamp for screen luminance [0.0..1.0].
            fg_mask: Optional coarse mask (H, W) where >0 marks foreground to inpaint.

        Returns:
            clean_plate: (H, W, 3) uint8 RGB clean screen image.
        """
        is_uint8 = rgb_image.dtype == np.uint8
        h, w = rgb_image.shape[:2]

        img_f = rgb_image.astype(np.float32)
        if is_uint8:
            img_f = img_f / 255.0

        diff = self.compute_screen_difference(img_f)

        if fg_mask is not None:
            screen_mask = (fg_mask == 0).astype(np.uint8)
        else:
            diff_thresh = max(0.02, float(np.percentile(diff[diff > 0], 25)) if np.any(diff > 0) else 0.05)
            screen_mask = (diff > diff_thresh).astype(np.uint8)

        if np.sum(screen_mask) < (h * w * 0.01):
            if self.screen_type == ScreenType.GREEN:
                fallback_screen = np.zeros_like(img_f)
                fallback_screen[:, :, 1] = 0.8
                return (fallback_screen * 255).astype(np.uint8)
            elif self.screen_type == ScreenType.BLUE:
                fallback_screen = np.zeros_like(img_f)
                fallback_screen[:, :, 2] = 0.8
                return (fallback_screen * 255).astype(np.uint8)
            else:
                return rgb_image.copy()

        clean = img_f.copy()
        current_mask = screen_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, patch_size), max(3, patch_size)))

        for it in range(iterations):
            scale_factor = 2 ** (it + 1)
            scaled_w = max(16, w // scale_factor)
            scaled_h = max(16, h // scale_factor)

            small_img = cv2.resize(clean, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(current_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

            inpaint_mask = ((1 - small_mask) * 255).astype(np.uint8)
            if np.any(inpaint_mask > 0):
                small_uint8 = np.clip(small_img * 255.0, 0, 255).astype(np.uint8)
                inpainted_small = cv2.inpaint(small_uint8, inpaint_mask, 5, cv2.INPAINT_TELEA)
                small_img = inpainted_small.astype(np.float32) / 255.0

            upsampled = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_LINEAR)
            clean = np.where(current_mask[:, :, None] == 1, clean, upsampled)
            current_mask = cv2.dilate(current_mask, kernel, iterations=2)

        if blur_radius > 0:
            ksize = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
            clean = cv2.GaussianBlur(clean, (ksize, ksize), 0)

        if darks > 0.0 or lights < 1.0:
            clean = np.clip((clean - darks) / max(1e-5, (lights - darks)), 0.0, 1.0)

        clean_uint8 = np.clip(clean * 255.0, 0, 255).astype(np.uint8)
        return clean_uint8

    # -------------------------------------------------------------------------
    # Matte Pulling (IBKGizmo emulation)
    # -------------------------------------------------------------------------

    def pull_matte(
        self,
        rgb_image: np.ndarray,
        clean_plate: Optional[np.ndarray] = None,
        red_weight: Optional[float] = None,
        blue_weight: Optional[float] = None,
        green_weight: Optional[float] = None,
        darks: float = 0.0,
        lights: float = 1.0,
        gamma: float = 1.0,
        black_clip: float = 0.01,
        white_clip: float = 0.99,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pull high-detail transmission alpha matte using clean plate comparison.

        Formula:
            T = diff_fg / max(diff_clean, epsilon)
            alpha = 1.0 - T
            Level adjustment: clamp((alpha - black_clip) / (white_clip - black_clip), 0, 1)^gamma

        Returns:
            alpha: (H, W) float32 in range [0.0, 1.0]
            clean_plate: (H, W, 3) uint8 clean plate used
            diff_fg: (H, W) float32 difference of foreground
            diff_clean: (H, W) float32 difference of clean plate
        """
        img_f = rgb_image.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

        if clean_plate is None:
            clean_plate = self.generate_clean_plate(rgb_image)

        clean_f = clean_plate.astype(np.float32)
        if clean_f.max() > 1.0:
            clean_f = clean_f / 255.0

        diff_fg = self.compute_screen_difference(
            img_f, red_weight=red_weight, blue_weight=blue_weight, green_weight=green_weight
        )
        diff_clean = self.compute_screen_difference(
            clean_f, red_weight=red_weight, blue_weight=blue_weight, green_weight=green_weight
        )

        # Transmission calculation: screen brightness ratio
        diff_clean_safe = np.maximum(diff_clean, 0.01)
        transmission = np.clip(diff_fg / diff_clean_safe, 0.0, 1.0)
        alpha = 1.0 - transmission

        # Darks and Lights leveling (adjust contrast in shadow/highlight areas)
        if darks > 0.0 or lights < 1.0:
            alpha = np.clip((alpha - darks) / max(1e-5, (lights - darks)), 0.0, 1.0)

        # Black and White clip (ensure background is clean black and foreground holds solid)
        if black_clip > 0.0 or white_clip < 1.0:
            span = max(1e-5, white_clip - black_clip)
            alpha = np.clip((alpha - black_clip) / span, 0.0, 1.0)

        # Gamma curve
        if gamma != 1.0 and gamma > 0.0:
            alpha = np.power(alpha, 1.0 / gamma)

        # Strict thresholding: pure black background guarantee
        alpha[alpha < 0.005] = 0.0
        # Strict thresholding: pure white core guarantee
        alpha[alpha > 0.995] = 1.0

        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        return alpha, clean_plate, diff_fg, diff_clean

    # -------------------------------------------------------------------------
    # Full Keying Pipeline
    # -------------------------------------------------------------------------

    def execute_keying(
        self,
        rgb_image: np.ndarray,
        clean_plate: Optional[np.ndarray] = None,
        patch_size: int = 5,
        blur_radius: int = 15,
        darks: float = 0.0,
        lights: float = 1.0,
        gamma: float = 1.0,
        black_clip: float = 0.01,
        white_clip: float = 0.99,
        fg_mask: Optional[np.ndarray] = None,
    ) -> IBKMatteResult:
        """
        Execute IBK keying workflow: Clean Plate + Alpha Matte extraction.

        Args:
            rgb_image: Input RGB frame (H, W, 3).
            clean_plate: Optional pre-existing clean plate. If None, generated automatically.
            patch_size: IBK clean plate patch size.
            blur_radius: Clean plate smoothing.
            darks, lights: IBK levels.
            gamma, black_clip, white_clip: Matte adjustments.
            fg_mask: Optional coarse mask to guide clean plate.

        Returns:
            IBKMatteResult with alpha, clean_plate, and difference maps.
        """
        if clean_plate is None:
            clean_plate = self.generate_clean_plate(
                rgb_image,
                patch_size=patch_size,
                blur_radius=blur_radius,
                darks=darks,
                lights=lights,
                fg_mask=fg_mask,
            )

        alpha, clean_plate, diff_fg, diff_clean = self.pull_matte(
            rgb_image=rgb_image,
            clean_plate=clean_plate,
            darks=darks,
            lights=lights,
            gamma=gamma,
            black_clip=black_clip,
            white_clip=white_clip,
        )

        return IBKMatteResult(
            alpha=alpha,
            clean_plate=clean_plate,
            screen_diff_fg=diff_fg,
            screen_diff_clean=diff_clean,
        )
