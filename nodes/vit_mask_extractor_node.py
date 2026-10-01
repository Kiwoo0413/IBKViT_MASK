"""
nodes/vit_mask_extractor_node.py
Node 03: ViT Mask Extractor Node for Griptape Nodes Desktop.
Integrates SAM 2 spatio-temporal tracking and ViTMatte Vision Transformer matting.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from core.io_utils import ImageSequenceIO, VideoIO
from core.vit_engine import ViTEngine
from nodes.griptape_compat import DataNode, Parameter, ParameterMode


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
                name="output_dir",
                type="str",
                default_value="",
                tooltip="마스크 저장 디렉토리 (비워두면 원본 비디오 폴더 내 자동 생성)",
                display_name="Output Directory",
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
                name="status",
                type="str",
                tooltip="실행 상태 요약",
                display_name="Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        input_video = str(self.get_parameter_value("input_video") or "").strip()
        if not input_video or not Path(input_video).exists():
            raise ValueError(f"Valid input_video path is required: '{input_video}'")

        seed_str = str(self.get_parameter_value("seed_coords") or "").strip()
        box_str = str(self.get_parameter_value("box_coords") or "").strip()
        erode_r = int(self.get_parameter_value("trimap_erode") or 12)
        dilate_r = int(self.get_parameter_value("trimap_dilate") or 15)
        max_frames = int(self.get_parameter_value("max_frames") or 0)
        out_dir = str(self.get_parameter_value("output_dir") or "").strip()

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

        frames = VideoIO.read_frames(input_video, max_frames=limit)

        # 1. Spatio-temporal tracking
        coarse_masks = engine.track_sam2_frames(
            frame_sequence=frames,
            seed_points=seed_points,
            box_coords=box_coords,
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
        self.set_parameter_value("status", f"Extracted ViT masks for {len(frames)} frames.")
