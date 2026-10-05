"""
scripts/verify_edge_jitter_filter.py
Experimental verification for geometric/connectivity-based jitter reduction in the Unknown zone.
"""

from __future__ import annotations

import time
from pathlib import Path
import cv2
import numpy as np


class EdgeJitterFilter:
    """
    Geometric & Connectivity Prior Jitter Filter for the Unknown Transition Zone.
    
    Attributes:
    - aspect_ratio_thresh: Minimum aspect ratio (L/W) to be classified as an elongated hair strand (default: 2.2).
    - min_hair_len: Minimum diagonal length for connected strands (default: 6 px).
    - core_anchor_dilation: Dilation radius around Core to check if strand originates from core (default: 3 px).
    - max_jitter_area: Maximum area for isolated unanchored blobs to be considered noise (default: 30 px).
    """

    def __init__(
        self,
        aspect_ratio_thresh: float = 2.2,
        min_hair_len: float = 6.0,
        core_anchor_dilation: int = 3,
        max_jitter_area: int = 30,
        alpha_thresh: float = 0.05,
    ):
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_hair_len = min_hair_len
        self.core_anchor_dilation = core_anchor_dilation
        self.max_jitter_area = max_jitter_area
        self.alpha_thresh = alpha_thresh

    def filter_matte(
        self,
        base_matte: np.ndarray,
        core_mask: np.ndarray,
        envelope_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Applies geometric connectivity filtering exclusively within the Unknown zone.
        
        Args:
            base_matte: (H, W) float32 [0, 1] input alpha matte.
            core_mask: (H, W) float32 or uint8 [0, 1] solid core (1.0 = core).
            envelope_mask: (H, W) float32 or uint8 [0, 1] envelope (1.0 = valid subject band).
            
        Returns:
            filtered_matte: (H, W) float32 [0, 1] cleaned alpha matte.
            stats: dict of diagnostic metrics.
        """
        t0 = time.perf_counter()
        h, w = base_matte.shape[:2]

        core_bin = (core_mask > 0.5).astype(np.uint8)
        env_bin = (envelope_mask > 0.5).astype(np.uint8)

        # 1. Strictly isolate the Unknown Transition Zone
        unknown_zone = (env_bin == 1) & (core_bin == 0)

        # 2. Extract edge detail candidates above threshold in the unknown zone
        candidate_bin = ((base_matte > self.alpha_thresh) & unknown_zone).astype(np.uint8)

        # Dilate core to test for connectivity/origin
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.core_anchor_dilation + 1, 2 * self.core_anchor_dilation + 1),
        )
        dilated_core = cv2.dilate(core_bin, kernel)
        core_touch_zone = (dilated_core == 1) & unknown_zone

        # 3. Connected Components Analysis (8-connectivity)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate_bin, connectivity=8
        )

        jitter_mask = np.zeros((h, w), dtype=bool)
        hair_mask = np.zeros((h, w), dtype=bool)

        hair_count = 0
        jitter_count = 0

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            bx_w = stats[label, cv2.CC_STAT_WIDTH]
            bx_h = stats[label, cv2.CC_STAT_HEIGHT]
            diag = np.sqrt(bx_w**2 + bx_h**2)
            aspect_ratio = max(bx_w, bx_h) / (min(bx_w, bx_h) + 1e-5)

            comp_pixels = (labels == label)

            # Check if this component touches the core boundary
            touches_core = np.any(comp_pixels & core_touch_zone)

            is_hair = False
            if touches_core:
                # Strands originating from the core: relaxed criteria
                if diag >= self.min_hair_len or aspect_ratio >= 1.5:
                    is_hair = True
            else:
                # Floating/detached components: strict high aspect ratio requirement
                if aspect_ratio >= self.aspect_ratio_thresh and diag >= self.min_hair_len:
                    is_hair = True

            if is_hair:
                hair_mask |= comp_pixels
                hair_count += 1
            else:
                # If area is within noise threshold or aspect ratio is low (clump/blob)
                if area <= self.max_jitter_area or aspect_ratio < self.aspect_ratio_thresh:
                    jitter_mask |= comp_pixels
                    jitter_count += 1
                else:
                    # Very large unanchored structure: keep safely
                    hair_mask |= comp_pixels

        # 4. Construct cleaned matte: suppress jitter only in unknown zone
        filtered_matte = base_matte.copy()
        filtered_matte[jitter_mask] = 0.0

        # Guarantee Core (1.0) and Envelope (0.0 outside) are 100% unchanged
        filtered_matte[core_bin == 1] = 1.0
        filtered_matte[env_bin == 0] = 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metrics = {
            "elapsed_ms": elapsed_ms,
            "total_components": num_labels - 1,
            "hair_components": hair_count,
            "jitter_components": jitter_count,
            "jitter_pixels_removed": int(np.sum(jitter_mask)),
            "hair_pixels_preserved": int(np.sum(hair_mask)),
        }
        return filtered_matte, metrics


def run_synthetic_benchmark():
    """Generates synthetic test scene with 1px hair strands, 1px noise dots, and clumps."""
    print("=" * 70)
    print("Running Edge Jitter Filter Verification Benchmark")
    print("=" * 70)

    h, w = 1080, 1920
    core_mask = np.zeros((h, w), dtype=np.float32)
    envelope_mask = np.zeros((h, w), dtype=np.float32)
    base_matte = np.zeros((h, w), dtype=np.float32)

    # 1. Subject Core: Circle centered at (960, 540) radius 200
    cv2.circle(core_mask, (960, 540), 200, 1.0, -1)

    # 2. Envelope: 60px dilation around core
    cv2.circle(envelope_mask, (960, 540), 260, 1.0, -1)

    base_matte = core_mask.copy()

    # 3. Ground Truth Hair Strands (Thin, curvilinear, high aspect ratio)
    hair_gt_mask = np.zeros((h, w), dtype=bool)

    # Hair 1: Curved 1px strand extending from (960, 340) upward to (940, 290)
    pts1 = np.array([[960, 340], [955, 320], [945, 305], [940, 290]], dtype=np.int32)
    for i in range(len(pts1) - 1):
        cv2.line(base_matte, tuple(pts1[i]), tuple(pts1[i+1]), 0.85, 1)
        cv2.line(hair_gt_mask.view(np.uint8), tuple(pts1[i]), tuple(pts1[i+1]), 1, 1)

    # Hair 2: Fine 1px diagonal strand extending to the right
    pts2 = np.array([[1160, 540], [1180, 535], [1205, 530]], dtype=np.int32)
    for i in range(len(pts2) - 1):
        cv2.line(base_matte, tuple(pts2[i]), tuple(pts2[i+1]), 0.75, 1)
        cv2.line(hair_gt_mask.view(np.uint8), tuple(pts2[i]), tuple(pts2[i+1]), 1, 1)

    # Hair 3: Detached fine 1px whisps in unknown zone (aspect ratio > 6:1)
    cv2.line(base_matte, (820, 380), (845, 395), 0.70, 1)
    cv2.line(hair_gt_mask.view(np.uint8), (820, 380), (845, 395), 1, 1)

    # Hair 4: Very fine 1px vertical strand
    cv2.line(base_matte, (960, 740), (960, 785), 0.90, 1)
    cv2.line(hair_gt_mask.view(np.uint8), (960, 740), (960, 785), 1, 1)

    # 4. Jitter Noise (Single pixels and isotropic clumps) in Unknown Zone
    noise_gt_mask = np.zeros((h, w), dtype=bool)
    np.random.seed(42)

    unknown_zone = (envelope_mask > 0.5) & (core_mask < 0.5)
    unknown_coords = np.argwhere(unknown_zone)

    # 4a. 40 isolated single-pixel noise dots
    chosen_pts = unknown_coords[np.random.choice(len(unknown_coords), 40, replace=False)]
    for y, x in chosen_pts:
        if not hair_gt_mask[y, x]:
            base_matte[y, x] = np.random.uniform(0.3, 0.9)
            noise_gt_mask[y, x] = True

    # 4b. 10 round clumpy blobs (3x3 and 4x4)
    clump_centers = unknown_coords[np.random.choice(len(unknown_coords), 10, replace=False)]
    for cy, cx in clump_centers:
        r = np.random.randint(1, 3)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and unknown_zone[ny, nx] and not hair_gt_mask[ny, nx]:
                    base_matte[ny, nx] = 0.65
                    noise_gt_mask[ny, nx] = True

    hair_pixel_count = np.sum(hair_gt_mask & unknown_zone)
    noise_pixel_count = np.sum(noise_gt_mask)

    print(f"Test Scene Initialized:")
    print(f"- Resolution: {w}x{h}")
    print(f"- Ground Truth Hair Pixels (in Unknown Zone): {hair_pixel_count}")
    print(f"- Ground Truth Jitter Noise Pixels: {noise_pixel_count}")

    # Run the filter
    filt = EdgeJitterFilter(aspect_ratio_thresh=2.2, min_hair_len=6.0, core_anchor_dilation=3)
    filtered_matte, metrics = filt.filter_matte(base_matte, core_mask, envelope_mask)

    # Verification calculations
    # 1. Hair preservation: What fraction of hair GT survived?
    preserved_hair_pixels = np.sum((filtered_matte > 0.05) & hair_gt_mask & unknown_zone)
    hair_preservation_rate = (preserved_hair_pixels / hair_pixel_count) * 100.0

    # 2. Jitter removal: What fraction of noise GT was removed?
    removed_noise_pixels = np.sum((filtered_matte <= 0.05) & noise_gt_mask)
    noise_removal_rate = (removed_noise_pixels / noise_pixel_count) * 100.0

    # 3. Core integrity: Must be 100% white
    core_diff = np.sum(np.abs(filtered_matte[core_mask > 0.5] - 1.0))

    # 4. Background integrity: Must be 100% black outside envelope
    bg_diff = np.sum(np.abs(filtered_matte[envelope_mask == 0.0]))

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS:")
    print("=" * 70)
    print(f"[Speed] Execution Speed: {metrics['elapsed_ms']:.2f} ms")
    print(f"[Count] Jitter Components Removed: {metrics['jitter_components']}")
    print(f"[Count] Hair Components Preserved: {metrics['hair_components']}")
    print(f"[Accuracy] Hair Preservation Rate: {hair_preservation_rate:.2f}% ({preserved_hair_pixels}/{hair_pixel_count} px)")
    print(f"[Accuracy] Jitter Removal Rate: {noise_removal_rate:.2f}% ({removed_noise_pixels}/{noise_pixel_count} px)")
    print(f"[Integrity] Core Integrity Diff (should be 0.0): {core_diff:.4f}")
    print(f"[Integrity] Background Integrity Diff (should be 0.0): {bg_diff:.4f}")

    # Test 4K scaling speed
    h4k, w4k = 2160, 3840
    base_4k = cv2.resize(base_matte, (w4k, h4k), interpolation=cv2.INTER_NEAREST)
    core_4k = cv2.resize(core_mask, (w4k, h4k), interpolation=cv2.INTER_NEAREST)
    env_4k = cv2.resize(envelope_mask, (w4k, h4k), interpolation=cv2.INTER_NEAREST)

    _, metrics_4k = filt.filter_matte(base_4k, core_4k, env_4k)
    print(f"[Speed] 4K UHD (3840x2160) Execution Speed: {metrics_4k['elapsed_ms']:.2f} ms")
    print("=" * 70)

    # Save visual comparison crops
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "jitter_test_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Crop around the top hair strand (x: 900-1020, y: 260-380)
    crop_before = (base_matte[260:420, 880:1040] * 255).astype(np.uint8)
    crop_after = (filtered_matte[260:420, 880:1040] * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "crop_before.png"), crop_before)
    cv2.imwrite(str(out_dir / "crop_after.png"), crop_after)

    # Full frame saves
    cv2.imwrite(str(out_dir / "matte_before.png"), (base_matte * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "matte_after.png"), (filtered_matte * 255).astype(np.uint8))
    print(f"Visual crops saved to: {out_dir}")

    return {
        "hair_rate": hair_preservation_rate,
        "noise_rate": noise_removal_rate,
        "time_1080p": metrics["elapsed_ms"],
        "time_4k": metrics_4k["elapsed_ms"],
    }


def run_temporal_sequence_benchmark():
    """Simulates a 15-frame sequence to measure temporal jitter/flicker variance reduction."""
    print("\n" + "=" * 70)
    print("Running Temporal Sequence Jitter & Flicker Variance Benchmark")
    print("=" * 70)

    h, w = 540, 960  # Fast HD test
    num_frames = 15
    np.random.seed(123)

    core_mask = np.zeros((h, w), dtype=np.float32)
    envelope_mask = np.zeros((h, w), dtype=np.float32)

    cv2.circle(core_mask, (480, 270), 100, 1.0, -1)
    cv2.circle(envelope_mask, (480, 270), 140, 1.0, -1)

    unknown_zone = (envelope_mask > 0.5) & (core_mask < 0.5)
    unknown_coords = np.argwhere(unknown_zone)

    # Hair strand GT
    hair_mask = np.zeros((h, w), dtype=bool)
    # Hair 1: Top curved strand
    pts = np.array([[480, 170], [475, 155], [468, 142], [460, 132]], dtype=np.int32)
    for i in range(len(pts) - 1):
        cv2.line(hair_mask.view(np.uint8), tuple(pts[i]), tuple(pts[i+1]), 1, 1)

    raw_frames = []
    filtered_frames = []
    filt = EdgeJitterFilter(aspect_ratio_thresh=2.2, min_hair_len=6.0, core_anchor_dilation=3)

    for f_idx in range(num_frames):
        f = core_mask.copy()
        # Consistent hair strand with subtle intensity breathing [0.80 ~ 0.85]
        hair_val = 0.80 + 0.05 * np.sin(f_idx * 0.5)
        for i in range(len(pts) - 1):
            cv2.line(f, tuple(pts[i]), tuple(pts[i+1]), hair_val, 1)

        # Randomly flickering jitter noise in the unknown zone per frame
        # 25 single pixel spikes
        spikes = unknown_coords[np.random.choice(len(unknown_coords), 25, replace=False)]
        for y, x in spikes:
            if not hair_mask[y, x]:
                f[y, x] = np.random.uniform(0.4, 0.95)

        # 3 flickering clumps
        clumps = unknown_coords[np.random.choice(len(unknown_coords), 3, replace=False)]
        for cy, cx in clumps:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and unknown_zone[ny, nx] and not hair_mask[ny, nx]:
                        f[ny, nx] = 0.70

        filtered, _ = filt.filter_matte(f, core_mask, envelope_mask)
        raw_frames.append(f)
        filtered_frames.append(filtered)

    raw_arr = np.array(raw_frames)          # (T, H, W)
    filt_arr = np.array(filtered_frames)    # (T, H, W)

    # Calculate temporal variance (flicker) across frames in the Unknown zone outside hair
    noise_zone = unknown_zone & (~hair_mask)
    raw_var = np.mean(np.var(raw_arr[:, noise_zone], axis=0))
    filt_var = np.mean(np.var(filt_arr[:, noise_zone], axis=0))
    flicker_reduction = ((raw_var - filt_var) / (raw_var + 1e-8)) * 100.0

    # Calculate hair strand preservation across all frames
    hair_preservations = []
    for f_idx in range(num_frames):
        pres = np.sum(filt_arr[f_idx, hair_mask] > 0.05) / np.sum(hair_mask)
        hair_preservations.append(pres)

    avg_hair_pres = np.mean(hair_preservations) * 100.0

    print(f"[Temporal] Raw Unknown Noise Variance (Flicker): {raw_var:.6f}")
    print(f"[Temporal] Filtered Unknown Noise Variance (Flicker): {filt_var:.6f}")
    print(f"[Temporal] Edge Flicker Reduction: {flicker_reduction:.2f}%")
    print(f"[Temporal] Hair Strand Average Preservation: {avg_hair_pres:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_synthetic_benchmark()
    run_temporal_sequence_benchmark()

