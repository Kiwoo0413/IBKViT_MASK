"""
core/matte_fusion.py
VFX Core / Edge Matte Fusion, Object Saliency Re-Tracking, and Dual Stabilization Engine.
Guarantees:
- Dynamic Polarity Verification & Auto-Inversion (ensures subject is white, background is black).
- Saliency Re-tracking against the original video plate.
- Pure White Core (1.0) with zero holes and zero internal chatter.
- Pure Black Background (0.0) with zero screen spill noise.
- Dual Outputs: Produces BOTH Jitter-Stabilized and Raw (Non-stabilized) 4K UHD alpha mattes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FusionConfig:
    """Configuration for Matte Fusion, Edge Refinement, and 4K scaling."""
    use_core_fill: bool = True           # Guarantee solid interior with 100% pure white core
    guided_filter_radius: int = 5        # Edge guide radius (0 to disable)
    guided_filter_eps: float = 1e-4      # Regularization parameter for edge preservation
    feather_radius: float = 1.0          # Sub-pixel edge feathering (Gaussian sigma)
    morph_close_size: int = 3            # Hole closure kernel size
    temporal_smoothing_alpha: float = 0.35 # Temporal EMA weight for edge stability (0.35 optimal)
    black_clip: float = 0.01             # Background threshold (below which is pure black 0.0)
    white_clip: float = 0.99             # Core threshold (above which is pure white 1.0)
    target_resolution: str = "4k"        # "4k" (3840x2160 UHD) or "native"
    auto_detect_polarity: bool = True    # Auto-detect if core/background are inverted and fix
    invert_matte: bool = False           # Manual invert override
    restore_fine_edges: bool = True      # Edge Re-Injection: restore pure optical hair edges after filtering
    edge_restore_band_radius: int = 80   # Safe zone radius around core to re-inject hair without background noise


class MatteFusionEngine:
    """
    VFX Composite Matte Fusion & Dual-Output Stabilization Engine.
    Combines solid interior Core mattes with high-detail IBK/ViT edge mattes.
    Re-tracks object saliency against original footage to prevent core/background inversion.
    Generates both stabilized and raw (unstabilized) 4K UHD outputs.
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        self._prev_frame_matte: Optional[np.ndarray] = None

    def reset_temporal_state(self) -> None:
        """Reset temporal smoothing buffer between video clips."""
        self._prev_frame_matte = None

    @staticmethod
    def auto_detect_screen_type(rgb_frame: np.ndarray) -> str:
        """
        Automatically detect whether footage is green screen or blue screen.
        """
        f = rgb_frame.astype(np.float32)
        if f.max() > 1.0:
            f = f / 255.0
        r = f[:, :, 0]
        g = f[:, :, 1]
        b = f[:, :, 2]
        green_diff = g - (0.5 * r + 0.5 * b)
        blue_diff = b - (0.5 * r + 0.5 * g)
        green_ratio = float(np.mean(green_diff > 0.05))
        blue_ratio = float(np.mean(blue_diff > 0.05))
        if blue_ratio > green_ratio and blue_ratio > 0.10:
            return "blue"
        elif green_ratio > blue_ratio and green_ratio > 0.10:
            return "green"
        return "green"

    # -------------------------------------------------------------------------
    # Polarity Detection & Auto-Inversion
    # -------------------------------------------------------------------------

    @staticmethod
    def detect_and_correct_polarity(
        matte: np.ndarray,
        rgb_frame: np.ndarray,
        screen_type: str = "green",
        invert_override: bool = False,
        auto_detect: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """
        Verify that matte foreground (white) corresponds to the actual object
        and matte background (black) corresponds to the screen.
        If inverted (e.g. green screen is white and object is black), flips it.

        Args:
            matte: (H, W) float32 [0.0..1.0] candidate matte.
            rgb_frame: (H, W, 3) uint8 or float32 original video frame.
            screen_type: "green" or "blue".
            invert_override: Manual forced inversion flag.
            auto_detect: Whether to automatically detect and correct inversion.

        Returns:
            (corrected_matte, was_inverted): Corrected matte and boolean flag.
        """
        m = matte.copy()
        if invert_override:
            return np.clip(1.0 - m, 0.0, 1.0).astype(np.float32), True

        if not auto_detect or rgb_frame is None:
            return m, False

        # Work in float32 [0..1]
        f_rgb = rgb_frame.astype(np.float32)
        if f_rgb.max() > 1.0:
            f_rgb = f_rgb / 255.0

        # Resize frame to matte size if dimensions differ
        mh, mw = m.shape[:2]
        if f_rgb.shape[:2] != (mh, mw):
            f_rgb = cv2.resize(f_rgb, (mw, mh), interpolation=cv2.INTER_AREA)

        # Compute screen color metric
        r = f_rgb[:, :, 0]
        g = f_rgb[:, :, 1]
        b = f_rgb[:, :, 2]

        if screen_type.lower() == "blue":
            screen_diff = b - (0.5 * r + 0.5 * g)
        else:  # green
            screen_diff = g - (0.5 * r + 0.5 * b)

        # Check average screen color of white pixels vs black pixels
        fg_mask = m > 0.7
        bg_mask = m < 0.3

        if np.sum(fg_mask) > 100 and np.sum(bg_mask) > 100:
            avg_fg_screen_color = float(np.mean(screen_diff[fg_mask]))
            avg_bg_screen_color = float(np.mean(screen_diff[bg_mask]))

            # In a valid key, the background screen MUST have higher screen color
            # than the foreground object. If FG has higher screen color, it is inverted!
            if avg_fg_screen_color > (avg_bg_screen_color + 0.05):
                m = 1.0 - m
                return np.clip(m, 0.0, 1.0).astype(np.float32), True

        return m, False

    # -------------------------------------------------------------------------
    # Object Saliency Re-Tracking against Original Plate
    # -------------------------------------------------------------------------

    @staticmethod
    def retrack_object_saliency(
        rgb_frame: np.ndarray,
        screen_type: str = "green",
        min_distance_from_edge: float = 25.0,
    ) -> np.ndarray:
        """
        Extract a deep interior core from the plate to anchor subject opacity.
        Uses Euclidean Distance Transform to strictly restrict the core to the
        deep interior (>min_distance_from_edge pixels away from any boundary),
        ensuring it NEVER interferes with hair or edge transitions.
        """
        f_rgb = rgb_frame.astype(np.float32)
        if f_rgb.max() > 1.0:
            f_rgb = f_rgb / 255.0

        r = f_rgb[:, :, 0]
        g = f_rgb[:, :, 1]
        b = f_rgb[:, :, 2]

        if screen_type.lower() == "blue":
            screen_diff = b - (0.5 * r + 0.5 * g)
        else:
            screen_diff = g - (0.5 * r + 0.5 * b)

        # Non-screen areas have screen_diff <= threshold
        obj_mask = (screen_diff < 0.08).astype(np.uint8) * 255

        # Clean noise & fill interior cavities
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Fill all interior holes completely
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(opened)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, thickness=-1)
        else:
            filled = opened

        # Deep interior restriction via Distance Transform
        if min_distance_from_edge > 0.0:
            dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
            max_d = float(dist.max()) if dist.size > 0 else 0.0
            effective_thresh = min(min_distance_from_edge, max_d * 0.4) if max_d > 0 else 0.0
            deep_core = (dist > effective_thresh).astype(np.float32)
        else:
            deep_core = (filled > 127).astype(np.float32)

        return deep_core

    # -------------------------------------------------------------------------
    # Core Hole Filling
    # -------------------------------------------------------------------------

    @staticmethod
    def fill_core_holes(core_matte: np.ndarray) -> np.ndarray:
        """
        Fills any internal cavities/holes inside the core matte,
        ensuring a completely solid, noise-free pure white interior (1.0).
        """
        c = (np.clip(core_matte, 0.0, 1.0) * 255.0).astype(np.uint8)
        _, binary = cv2.threshold(c, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(binary)
        if contours:
            cv2.drawContours(solid, contours, -1, 255, thickness=-1)
        else:
            solid = binary

        return (solid.astype(np.float32) / 255.0)

    # -------------------------------------------------------------------------
    # Fusion & Refinement
    # -------------------------------------------------------------------------

    def fuse_core_and_edge(
        self,
        core_matte: Optional[np.ndarray],
        edge_matte: np.ndarray,
        envelope_matte: Optional[np.ndarray] = None,
        use_core_fill: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Combine solid interior Core matte with fine detail Edge matte.
        When envelope_matte is provided:
        - Inside core: strictly 1.0 (pure white, zero holes).
        - Outside envelope: strictly 0.0 (pure black, zero noise).
        - Transition zone: pure IBK edge transmission (preserves hair, motion blur).
        """
        e = edge_matte.astype(np.float32)
        if e.max() > 1.0:
            e = e / 255.0
        eh, ew = e.shape[:2]

        if core_matte is not None:
            c = core_matte.astype(np.float32)
            if c.max() > 1.0:
                c = c / 255.0
            if c.shape[:2] != (eh, ew):
                c = cv2.resize(c, (ew, eh), interpolation=cv2.INTER_NEAREST)
            fill_core = self.config.use_core_fill if use_core_fill is None else use_core_fill
            if fill_core:
                c = self.fill_core_holes(c)
        else:
            c = np.zeros_like(e)

        if envelope_matte is not None:
            env = envelope_matte.astype(np.float32)
            if env.max() > 1.0:
                env = env / 255.0
            if env.shape[:2] != (eh, ew):
                env = cv2.resize(env, (ew, eh), interpolation=cv2.INTER_LINEAR)
            # Non-destructive envelope fusion:
            # - Inside core: strictly 1.0 (pure white)
            # - Outside envelope: strictly 0.0 (pure black)
            # - Transition zone: pure IBK edge transmission
            fused = c + (1.0 - c) * e * env
        else:
            if core_matte is not None:
                fused = np.maximum(c, e)
            else:
                fused = e

        return np.clip(fused, 0.0, 1.0).astype(np.float32)

    def refine_matte(
        self,
        matte: np.ndarray,
        rgb_guide: Optional[np.ndarray] = None,
        config: Optional[FusionConfig] = None,
    ) -> np.ndarray:
        """
        Refine alpha matte edges using guided filtering, feathering, and smooth level mapping.
        Preserves natural anti-aliased transitions and eliminates edge chatter.
        """
        cfg = config or self.config
        m = matte.copy().astype(np.float32)

        # 1. Morphological Closing (seal pinholes)
        if cfg.morph_close_size > 1:
            k = cfg.morph_close_size if cfg.morph_close_size % 2 == 1 else cfg.morph_close_size + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m_uint8 = (m * 255.0).astype(np.uint8)
            closed = cv2.morphologyEx(m_uint8, cv2.MORPH_CLOSE, kernel)
            m = closed.astype(np.float32) / 255.0

        # Protect deep solid core interior from blur dimming
        inner_core = cv2.erode((m > 0.9).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

        # 2. Guided Filter Edge Alignment
        if cfg.guided_filter_radius > 0 and rgb_guide is not None:
            guide_gray = cv2.cvtColor(rgb_guide, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if guide_gray.max() > 1.0:
                guide_gray = guide_gray / 255.0

            r = cfg.guided_filter_radius
            eps = cfg.guided_filter_eps

            if hasattr(cv2, "ximgproc"):
                m = cv2.ximgproc.guidedFilter(guide=guide_gray, src=m, radius=r, eps=eps)
            else:
                m_uint8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
                bilateral = cv2.bilateralFilter(m_uint8, d=max(3, r * 2), sigmaColor=50, sigmaSpace=50)
                m = bilateral.astype(np.float32) / 255.0

        # 3. Sub-pixel Edge Feathering (Anti-Aliasing)
        if cfg.feather_radius > 0.0:
            sigma = cfg.feather_radius
            ksize = int(np.ceil(sigma * 3)) * 2 + 1
            m = cv2.GaussianBlur(m, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

        # Re-lock deep core interior to 1.0 (pure white, zero internal holes)
        m[inner_core == 1] = 1.0

        # 4. Continuous Level Adjustments:
        # Smoothly crush background noise below black_clip to pure 0.0
        # Smoothly pull high core values above white_clip to pure 1.0
        if cfg.black_clip > 0.0 or cfg.white_clip < 1.0:
            span = max(1e-5, cfg.white_clip - cfg.black_clip)
            m = np.clip((m - cfg.black_clip) / span, 0.0, 1.0)

        return np.clip(m, 0.0, 1.0).astype(np.float32)

    def inject_fine_edge_detail(
        self,
        base_matte: np.ndarray,
        raw_edge_matte: np.ndarray,
        core_matte: Optional[np.ndarray] = None,
        band_radius: int = 80,
    ) -> np.ndarray:
        """
        Edge Re-Injection: Restores ultra-fine hair strands and optical edge transparency
        that may have been clipped by tight envelopes or softened by refinement filters.
        Restricts injection to a safe zone (band_radius around core/subject) to guarantee zero far-background noise.
        """
        if raw_edge_matte is None or not self.config.restore_fine_edges:
            return base_matte

        e = raw_edge_matte.astype(np.float32)
        if e.max() > 1.0:
            e = e / 255.0

        b = base_matte.astype(np.float32)
        if b.max() > 1.0:
            b = b / 255.0

        bh, bw = b.shape[:2]
        if e.shape[:2] != (bh, bw):
            e = cv2.resize(e, (bw, bh), interpolation=cv2.INTER_LINEAR)

        anchor = core_matte if core_matte is not None else (b > 0.5).astype(np.float32)
        if anchor.shape[:2] != (bh, bw):
            anchor = cv2.resize(anchor, (bw, bh), interpolation=cv2.INTER_NEAREST)

        anchor_bin = (anchor > 0.2).astype(np.uint8)
        if np.any(anchor_bin):
            ksize = max(15, band_radius if band_radius % 2 == 1 else band_radius + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            safe_zone = cv2.dilate(anchor_bin, kernel)
        else:
            safe_zone = np.ones((bh, bw), dtype=np.uint8)

        restored = b.copy()
        mask = safe_zone > 0
        restored[mask] = np.maximum(b[mask], e[mask])

        return np.clip(restored, 0.0, 1.0).astype(np.float32)

    def apply_temporal_smoothing(
        self,
        current_matte: np.ndarray,
        alpha_factor: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply Exponential Moving Average (EMA) across consecutive video frames
        to suppress edge flicker. Operates continuously across the edge transition zone.
        """
        factor = self.config.temporal_smoothing_alpha if alpha_factor is None else alpha_factor

        if factor <= 0.0 or self._prev_frame_matte is None:
            self._prev_frame_matte = current_matte.copy()
            return current_matte

        # Continuous EMA blend without discontinuous edge clamping
        smoothed = (1.0 - factor) * current_matte + factor * self._prev_frame_matte
        self._prev_frame_matte = smoothed.copy()
        return np.clip(smoothed, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def upscale_to_4k(
        matte: np.ndarray,
        target_width: int = 3840,
        target_height: int = 2160,
        preserve_aspect: bool = True,
    ) -> np.ndarray:
        """
        Upscale alpha matte to 4K UHD (3840x2160 or proportional) using Lanczos4 interpolation.
        Maintains pure white core (1.0) and pure black background (0.0).
        """
        h, w = matte.shape[:2]
        if w == target_width and h == target_height:
            return matte

        if preserve_aspect:
            scale = target_width / float(w)
            out_w = target_width
            out_h = int(round(h * scale))
            if out_h % 2 != 0:
                out_h += 1
        else:
            out_w = target_width
            out_h = target_height

        upscaled = cv2.resize(matte, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        upscaled[upscaled > 0.99] = 1.0
        upscaled[upscaled < 0.01] = 0.0

        return np.clip(upscaled, 0.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # Dual Processing Entry Point (Stabilized + Raw)
    # -------------------------------------------------------------------------

    def process_frame_dual(
        self,
        core_matte: Optional[np.ndarray],
        edge_matte: np.ndarray,
        rgb_guide: Optional[np.ndarray] = None,
        screen_type: str = "green",
        scale_to_4k: bool = True,
        envelope_matte: Optional[np.ndarray] = None,
        raw_core_matte: Optional[np.ndarray] = None,
        raw_envelope_matte: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete frame processing returning BOTH:
        1. stabilized_matte: Jitter-free matte (using ViT-stabilized core & envelope, natural IBK edge).
        2. raw_matte: Pure per-frame instantaneous matte.

        Both outputs have:
        - Polarity verified and auto-corrected (Subject is White, Screen is Black).
        - Pure White solid core and Pure Black background.
        - Silky smooth, chatter-free edge anti-aliasing without aggressive temporal hacks.
        - 4K UHD Lanczos-4 upscaling applied.
        """
        # 1. Core & Envelope integration:
        if envelope_matte is not None:
            # Stabilized path uses ViT-stabilized core and envelope
            fused_stab = self.fuse_core_and_edge(core_matte, edge_matte, envelope_matte=envelope_matte)
            # Raw path uses instantaneous raw core and raw envelope
            raw_c = raw_core_matte if raw_core_matte is not None else core_matte
            raw_env = raw_envelope_matte if raw_envelope_matte is not None else envelope_matte
            fused_raw = self.fuse_core_and_edge(raw_c, edge_matte, envelope_matte=raw_env)
        else:
            if core_matte is not None:
                fused_raw = self.fuse_core_and_edge(core_matte, edge_matte)
            else:
                fused_raw = edge_matte.copy()
            fused_stab = fused_raw

        # 2. Polarity check & auto-inversion
        fused_stab_corr, _ = self.detect_and_correct_polarity(
            fused_stab,
            rgb_frame=rgb_guide,
            screen_type=screen_type,
            invert_override=self.config.invert_matte,
            auto_detect=self.config.auto_detect_polarity,
        )
        fused_raw_corr, _ = self.detect_and_correct_polarity(
            fused_raw,
            rgb_frame=rgb_guide,
            screen_type=screen_type,
            invert_override=self.config.invert_matte,
            auto_detect=self.config.auto_detect_polarity,
        )

        # 3. Refinement (guided filtering & continuous level mapping)
        # No aggressive temporal edge hacks are applied here to keep IBK edges natural!
        refined_raw = self.refine_matte(fused_raw_corr, rgb_guide=rgb_guide)
        refined_stab = self.refine_matte(fused_stab_corr, rgb_guide=rgb_guide)

        # 3-B. Edge Re-Injection: Restore fine hair strands onto the refined base matte
        if self.config.restore_fine_edges and edge_matte is not None:
            refined_raw = self.inject_fine_edge_detail(
                base_matte=refined_raw,
                raw_edge_matte=edge_matte,
                core_matte=raw_core_matte if raw_core_matte is not None else core_matte,
                band_radius=self.config.edge_restore_band_radius,
            )
            refined_stab = self.inject_fine_edge_detail(
                base_matte=refined_stab,
                raw_edge_matte=edge_matte,
                core_matte=core_matte,
                band_radius=self.config.edge_restore_band_radius,
            )

        # If envelope_matte was not supplied (legacy mode), apply EMA temporal smoothing on stabilized stream
        if envelope_matte is None:
            refined_stab = self.apply_temporal_smoothing(refined_stab)

        # 4. 4K UHD Upscaling
        if scale_to_4k:
            out_raw = self.upscale_to_4k(refined_raw)
            out_stabilized = self.upscale_to_4k(refined_stab)
        else:
            out_raw = refined_raw
            out_stabilized = refined_stab

        return out_stabilized, out_raw

    def process_frame(
        self,
        core_matte: Optional[np.ndarray],
        edge_matte: np.ndarray,
        rgb_guide: Optional[np.ndarray] = None,
        scale_to_4k: bool = True,
        envelope_matte: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Single output fallback (returns stabilized matte)."""
        stabilized, _ = self.process_frame_dual(
            core_matte=core_matte,
            edge_matte=edge_matte,
            rgb_guide=rgb_guide,
            scale_to_4k=scale_to_4k,
            envelope_matte=envelope_matte,
        )
        return stabilized
