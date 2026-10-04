"""
ibkvit_core/vit_engine.py
Vision Transformer (ViTMatte) Neural Matting Engine and Unified Compatibility Wrapper.

Contains:
1. ViTMatteEngine: Pure Hugging Face Vision Transformer (VitMatteForImageMatting) inference engine.
   Specialized in sub-pixel transition zone alpha estimation with ROI-bounding-box acceleration.
2. ViTEngine: Unified composite engine subclassing CoreEngine and integrating ViTMatteEngine
   for 100% backward compatibility with existing workflows and nodes.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from ibkvit_core.core_engine import CoreEngine, parse_box, parse_coords

logger = logging.getLogger("ViTEngine")


@dataclass
class ViTMatteResult:
    """Result of ViT-based mask extraction."""
    alpha: np.ndarray        # float32 [0.0, 1.0] (H, W) high-precision alpha matte
    core_mask: np.ndarray    # uint8 {0, 255} (H, W) solid interior foreground mask (pure white, zero holes)
    trimap: np.ndarray       # uint8 {0, 128, 255} (H, W) trimap (0=bg, 128=unknown, 255=fg)
    device_used: str         # "cuda" or "cpu"
    envelope_mask: Optional[np.ndarray] = None # uint8 {0, 255} (H, W) outer transition envelope


class ViTMatteEngine:
    """
    Hugging Face Vision Transformer (ViTMatte) Edge Matting Engine.
    Handles neural network model loading, ROI-cropped fast sub-pixel inference,
    and fallback guided matting.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        vitmatte_model_id: str = "hustvl/vitmatte-small-composition-1k",
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.vitmatte_model_id = vitmatte_model_id
        self._vitmatte_model = None
        self._vitmatte_processor = None

    def _load_vitmatte(self) -> None:
        """Lazy load Hugging Face VitMatte model and image processor."""
        if self._vitmatte_model is not None:
            return

        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            from ibkvit_core.griptape_model_manager import resolve_model_weights_and_config
            model_path, _ = resolve_model_weights_and_config(self.vitmatte_model_id)
            load_target = model_path if model_path else self.vitmatte_model_id

            logger.info("Loading ViTMatte model: %s onto %s", load_target, self.device)
            self._vitmatte_processor = VitMatteImageProcessor.from_pretrained(load_target)
            self._vitmatte_model = VitMatteForImageMatting.from_pretrained(
                load_target,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            ).to(self.device)
            self._vitmatte_model.eval()
        except Exception as e:
            logger.warning("Could not load Hugging Face VitMatte (%s). Falling back to guided filter matting.", e)
            self._vitmatte_model = False

    def predict_vitmatte(
        self,
        rgb_image: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        """
        Run ViTMatte inference on an RGB frame and trimap.
        Guarantees:
        - trimap == 255 -> strictly 1.0 (pure white core)
        - trimap == 0   -> strictly 0.0 (pure black background)
        """
        self._load_vitmatte()

        if self._vitmatte_model and self._vitmatte_model is not False:
            try:
                from PIL import Image

                pil_img = Image.fromarray(rgb_image)
                pil_trimap = Image.fromarray(trimap)

                inputs = self._vitmatte_processor(
                    images=pil_img,
                    trimaps=pil_trimap,
                    return_tensors="pt",
                )

                dtype = torch.float16 if self.device == "cuda" else torch.float32
                pixel_values = inputs["pixel_values"].to(device=self.device, dtype=dtype)

                with torch.inference_mode():
                    outputs = self._vitmatte_model(pixel_values=pixel_values)
                    alphas = outputs.alphas.squeeze().float().cpu().numpy()

                h, w = rgb_image.shape[:2]
                if alphas.shape[0] >= h and alphas.shape[1] >= w:
                    alphas = alphas[:h, :w]
                elif alphas.shape[:2] != (h, w):
                    alphas = cv2.resize(alphas, (w, h), interpolation=cv2.INTER_LINEAR)

                alphas[trimap == 0] = 0.0
                alphas[trimap == 255] = 1.0
                return np.clip(alphas, 0.0, 1.0).astype(np.float32)

            except Exception as ex:
                logger.warning("ViTMatte forward failed (%s). Using guided matting fallback.", ex)

        return self._fallback_guided_matting(rgb_image, trimap)

    @staticmethod
    def _fallback_guided_matting(
        rgb_image: np.ndarray,
        trimap: np.ndarray,
        radius: int = 8,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """
        Fast Guided Filter matting fallback when transformer weights are not yet downloaded.
        """
        gray_guide = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        p = (trimap.astype(np.float32) / 255.0)

        refined = cv2.ximgproc.guidedFilter(
            guide=gray_guide,
            src=p,
            radius=radius,
            eps=eps,
        ) if hasattr(cv2, "ximgproc") else cv2.bilateralFilter((p * 255).astype(np.uint8), 9, 75, 75).astype(np.float32) / 255.0

        refined[trimap == 0] = 0.0
        refined[trimap == 255] = 1.0
        return np.clip(refined, 0.0, 1.0).astype(np.float32)

    def extract_vit_matte(
        self,
        rgb_image: np.ndarray,
        coarse_mask: np.ndarray,
        erode_radius: int = 12,
        dilate_radius: int = 15,
        enable_adaptive_blur: bool = True,
        enable_roi_crop: bool = True,
        roi_padding: int = 32,
        screen_type: str = "green",
    ) -> ViTMatteResult:
        """
        Execute ViT-based mask extraction on a single image given a coarse mask.
        Supports:
        - Adaptive blur/defocus estimation to dynamically size the trimap transition band.
        - Tight ROI bounding-box cropping: evaluates ViT ONLY on the unknown transition zone,
          reducing computational load by 70~90% while guaranteeing rock-solid 1.0 core & 0.0 background.
        """
        if enable_adaptive_blur:
            eff_erode, eff_dilate, _ = CoreEngine.estimate_edge_blur_profile(
                rgb_image=rgb_image,
                coarse_mask=coarse_mask,
                screen_type=screen_type,
                base_erode=erode_radius,
                base_dilate=dilate_radius,
            )
        else:
            eff_erode, eff_dilate = erode_radius, dilate_radius

        trimap = CoreEngine.generate_trimap(
            mask=coarse_mask,
            erode_kernel_size=eff_erode,
            dilate_kernel_size=eff_dilate,
        )

        core_mask = (trimap == 255).astype(np.uint8) * 255
        envelope_mask = (trimap > 0).astype(np.uint8) * 255
        h, w = rgb_image.shape[:2]

        if enable_roi_crop:
            unknown_pts = np.argwhere(trimap == 128)
            if len(unknown_pts) == 0:
                alpha = (trimap == 255).astype(np.float32)
                return ViTMatteResult(
                    alpha=alpha,
                    core_mask=core_mask,
                    envelope_mask=envelope_mask,
                    trimap=trimap,
                    device_used=self.device,
                )

            ymin, xmin = unknown_pts.min(axis=0)
            ymax, xmax = unknown_pts.max(axis=0)

            x1 = max(0, int(xmin) - roi_padding)
            y1 = max(0, int(ymin) - roi_padding)
            x2 = min(w, int(xmax) + roi_padding + 1)
            y2 = min(h, int(ymax) + roi_padding + 1)

            rgb_crop = rgb_image[y1:y2, x1:x2]
            trimap_crop = trimap[y1:y2, x1:x2]

            crop_alpha = self.predict_vitmatte(
                rgb_image=rgb_crop,
                trimap=trimap_crop,
            )

            alpha = np.zeros((h, w), dtype=np.float32)
            alpha[trimap == 255] = 1.0

            alpha[y1:y2, x1:x2] = np.where(
                trimap_crop == 128,
                crop_alpha,
                np.where(trimap_crop == 255, 1.0, 0.0),
            )
        else:
            alpha = self.predict_vitmatte(
                rgb_image=rgb_image,
                trimap=trimap,
            )

        alpha[trimap == 255] = 1.0
        alpha[trimap == 0] = 0.0

        return ViTMatteResult(
            alpha=alpha,
            core_mask=core_mask,
            envelope_mask=envelope_mask,
            trimap=trimap,
            device_used=self.device,
        )

    def release_memory(self) -> None:
        """Flush VRAM & free transformer weights."""
        if self._vitmatte_model is not None and self._vitmatte_model is not False:
            del self._vitmatte_model
            self._vitmatte_model = None
        if self._vitmatte_processor is not None:
            del self._vitmatte_processor
            self._vitmatte_processor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("ViTMatteEngine VRAM released successfully.")


class ViTEngine(CoreEngine):
    """
    Unified ViT Engine combining CoreEngine (tracking, trimap, core/envelope stabilization)
    and ViTMatteEngine (neural edge matting) for complete backward compatibility.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        vitmatte_model_id: str = "hustvl/vitmatte-small-composition-1k",
    ) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vitmatte_model_id = vitmatte_model_id
        self._matte_engine: Optional[ViTMatteEngine] = None

    @property
    def matte_engine(self) -> ViTMatteEngine:
        if self._matte_engine is None:
            self._matte_engine = ViTMatteEngine(device=self.device, vitmatte_model_id=self.vitmatte_model_id)
        return self._matte_engine

    def predict_vitmatte(self, rgb_image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        return self.matte_engine.predict_vitmatte(rgb_image=rgb_image, trimap=trimap)

    def extract_vit_matte(
        self,
        rgb_image: np.ndarray,
        coarse_mask: np.ndarray,
        erode_radius: int = 12,
        dilate_radius: int = 15,
        enable_adaptive_blur: bool = True,
        enable_roi_crop: bool = True,
        roi_padding: int = 32,
        screen_type: str = "green",
    ) -> ViTMatteResult:
        return self.matte_engine.extract_vit_matte(
            rgb_image=rgb_image,
            coarse_mask=coarse_mask,
            erode_radius=erode_radius,
            dilate_radius=dilate_radius,
            enable_adaptive_blur=enable_adaptive_blur,
            enable_roi_crop=enable_roi_crop,
            roi_padding=roi_padding,
            screen_type=screen_type,
        )

    def release_memory(self) -> None:
        if self._matte_engine is not None:
            self._matte_engine.release_memory()
            self._matte_engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("ViTEngine VRAM released successfully.")
