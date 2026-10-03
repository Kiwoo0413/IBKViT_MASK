"""
nodes/vit_mask_extractor_node.py
Node 03: ViT Mask Extractor Node for Griptape Nodes Desktop.
Integrates SAM 2 spatio-temporal tracking and ViTMatte Vision Transformer matting.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Ensure library root and its virtualenv site-packages are in sys.path
_LIB_ROOT = Path(__file__).resolve().parent.parent
if str(_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIB_ROOT))
_VENV_SITE = _LIB_ROOT / ".venv" / "Lib" / "site-packages"
if _VENV_SITE.exists() and str(_VENV_SITE) not in sys.path:
    sys.path.append(str(_VENV_SITE))

# Ensure host Python packages (cv2, torch, etc.) are reachable if running inside embedded engine
try:
    import cv2  # noqa: F401
except ImportError:
    import shutil
    _sys_py = shutil.which("python")
    if _sys_py:
        _sys_site = Path(_sys_py).resolve().parent / "Lib" / "site-packages"
        if _sys_site.exists() and str(_sys_site) not in sys.path:
            sys.path.append(str(_sys_site))

import cv2
import numpy as np

from ibkvit_core.io_utils import ImageSequenceIO, VideoIO
from ibkvit_core.vit_engine import ViTEngine, parse_coords, parse_box
from ibkvit_core.griptape_compat import DataNode, Parameter, ParameterMode


def parse_coords(coord_str: str) -> Optional[List[Tuple[float, float]]]:
    """Parse string representations of coordinates into float pairs."""
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


class ViTMaskExtractorNode(DataNode):
    """
    Node 03: ViT Mask Extractor (Vision Transformer)
    SAM 2 기반 시공간 객체 추적과 ViTMatte(Vision Transformer Matting)를 결합하여
    전경 코어 마스크(Core), 3구역 트라이맵(Trimap), 고정밀 서브픽셀 알파(ViT Alpha)를 추출합니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        try:
            from ibkvit_core.griptape_model_manager import sync_griptape_config_models
            sync_griptape_config_models()
        except Exception:
            pass

        # ── Inputs ───────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="input_video",
                type="str",
                default_value="",
                tooltip="입력 비디오 파일의 절대 경로",
                display_name="Input Video Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="seed_coords",
                type="str",
                default_value="",
                tooltip="추적 대상 시드 키포인트 좌표 (예: '640,360' 또는 '[[640,360]]')",
                display_name="Seed Coords",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="box_coords",
                type="str",
                default_value="",
                tooltip="바운딩 박스 좌표 (예: '100,100,500,500')",
                display_name="Box Coords",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="trimap_erode",
                type="int",
                default_value=12,
                tooltip="솔리드 코어 생성을 위한 트라이맵 침식(Erode) 크기",
                display_name="Trimap Erode",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="trimap_dilate",
                type="int",
                default_value=15,
                tooltip="경계 영역 포괄을 위한 트라이맵 팽창(Dilate) 크기",
                display_name="Trimap Dilate",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="max_frames",
                type="int",
                default_value=0,
                tooltip="처리할 최대 프레임 수 (0 = 전체)",
                display_name="Max Frames",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="reference_sequence_dir",
                type="str",
                default_value="",
                tooltip="선택적: 프레임 수를 동기화할 Clean Plate 또는 Edge 마스크 폴더 경로",
                display_name="Reference Sequence Dir (Optional)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_dir",
                type="str",
                default_value="",
                tooltip="마스크 저장 디렉토리 (비워두면 원본 비디오 폴더 내 자동 생성)",
                display_name="Output Directory",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="enable_adaptive_blur",
                type="bool",
                default_value=True,
                tooltip="디포커스/모션블러 정도를 자동 감지하여 트라이맵 경계 폭을 가변 조절",
                display_name="Adaptive Blur Detection",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="enable_roi_crop",
                type="bool",
                default_value=True,
                tooltip="경계 Unknown 영역만 타이트하게 Bounding Box Crop하여 ViT 연산 속도 대폭 향상 (70~90% 절감)",
                display_name="Tight ROI Crop (Fast ViT)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="roi_padding",
                type="int",
                default_value=32,
                tooltip="타이트 ROI 크롭 시 경계 안전 여백 (픽셀)",
                display_name="ROI Safety Padding",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ── Outputs ──────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="core_mask_dir",
                type="str",
                tooltip="내부 구멍 없는 100% 솔리드 전경 코어 마스크 폴더",
                display_name="Core Mask Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="vit_alpha_dir",
                type="str",
                tooltip="ViT 기반 고정밀 알파 마스크 폴더",
                display_name="ViT Alpha Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="preview_video_path",
                type="str",
                tooltip="빨간색 마스크 검수용 오버레이 MP4 비디오 경로",
                display_name="Preview Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_count",
                type="int",
                default_value=0,
                tooltip="추출 완료된 총 프레임 수",
                display_name="Frame Count",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="status",
                type="str",
                tooltip="실행 상태 요약",
                display_name="Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        try:
            from ibkvit_core.griptape_model_manager import ensure_all_models_ready
            ensure_all_models_ready(auto_download=True)
        except Exception:
            pass

        input_video = str(self.get_parameter_value("input_video") or "").strip()
        if not input_video or not Path(input_video).exists():
            raise ValueError(f"Valid input_video path is required: '{input_video}'")

        seed_str = str(self.get_parameter_value("seed_coords") or "").strip()
        box_str = str(self.get_parameter_value("box_coords") or "").strip()
        erode_r = int(self.get_parameter_value("trimap_erode") or 12)
        dilate_r = int(self.get_parameter_value("trimap_dilate") or 15)
        max_frames = int(self.get_parameter_value("max_frames") or 0)
        ref_dir = str(self.get_parameter_value("reference_sequence_dir") or "").strip()
        out_dir = str(self.get_parameter_value("output_dir") or "").strip()
        enable_adaptive_blur = bool(self.get_parameter_value("enable_adaptive_blur") if self.get_parameter_value("enable_adaptive_blur") is not None else True)
        enable_roi_crop = bool(self.get_parameter_value("enable_roi_crop") if self.get_parameter_value("enable_roi_crop") is not None else True)
        roi_pad = int(self.get_parameter_value("roi_padding") or 32)

        seed_points = parse_coords(seed_str)
        box_coords = parse_box(box_str)

        out_dir = VideoIO.resolve_output_dir(input_video, custom_output_dir=out_dir, subfolder_suffix="vit_masks")
        out_path = Path(out_dir)
        core_dir = out_path / "core_masks"
        vit_dir = out_path / "vit_alpha"
        core_dir.mkdir(parents=True, exist_ok=True)
        vit_dir.mkdir(parents=True, exist_ok=True)

        engine = ViTEngine()
        info = VideoIO.get_video_info(input_video)
        fps = info["fps"]
        limit = max_frames if max_frames > 0 else None

        ref_files = []
        if ref_dir and Path(ref_dir).exists():
            ref_files = sorted(list(Path(ref_dir).glob("*.png")) + list(Path(ref_dir).glob("*.exr")))

        # If a reference sequence (Edge matte or Clean plate) is provided, synchronize frame count!
        if max_frames == 0 and len(ref_files) > 1:
            limit = len(ref_files)

        frames = VideoIO.read_frames(input_video, max_frames=limit)

        from ibkvit_core.matte_fusion import MatteFusionEngine
        detected_screen = "green"
        detected_init_mask = None
        if len(frames) > 0:
            detected_screen = MatteFusionEngine.auto_detect_screen_type(frames[0])
            if not seed_points and not box_coords:
                detected_init_mask = engine.detect_subject_coarse_mask(frames[0], screen_type=detected_screen)

        # 1. Spatio-temporal tracking
        coarse_masks = engine.track_sam2_frames(
            frame_sequence=frames,
            seed_points=seed_points,
            box_coords=box_coords,
            init_mask=detected_init_mask,
            screen_type=detected_screen,
        )

        core_mattes: List[np.ndarray] = []
        vit_alphas: List[np.ndarray] = []

        # 2. Per-frame ViT Matting and Trimap
        for idx, (frame, coarse_m) in enumerate(zip(frames, coarse_masks)):
            res = engine.extract_vit_matte(
                rgb_image=frame,
                coarse_mask=coarse_m,
                erode_radius=erode_r,
                dilate_radius=dilate_r,
                enable_adaptive_blur=enable_adaptive_blur,
                enable_roi_crop=enable_roi_crop,
                roi_padding=roi_pad,
                screen_type=detected_screen,
            )

            core_mattes.append(res.core_mask.astype(np.float32) / 255.0)
            vit_alphas.append(res.alpha)

            # Save core mask
            cv2.imwrite(str(core_dir / f"core.{idx:05d}.png"), res.core_mask)
            # Save 16-bit ViT alpha
            ImageSequenceIO.save_png_16bit(res.alpha, vit_dir / f"vit_alpha.{idx:05d}.png")

        # 3. Create Red overlay QA video
        preview_video = str(out_path / "mask_overlay_preview.mp4")
        VideoIO.create_red_overlay_video(
            frames=frames,
            masks=vit_alphas,
            output_path=preview_video,
            fps=fps,
        )

        # Release transformer VRAM
        engine.release_memory()

        self.set_parameter_value("core_mask_dir", str(core_dir))
        self.set_parameter_value("vit_alpha_dir", str(vit_dir))
        self.set_parameter_value("preview_video_path", preview_video)
        total_video_frames = info.get("frame_count", len(frames))
        self.set_parameter_value("status", f"Extracted ViT masks for {len(frames)}/{total_video_frames} frames.")
