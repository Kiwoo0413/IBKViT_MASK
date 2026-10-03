"""
nodes/vfx_matte_refiner_node.py
Node 04: VFX Matte Refiner & Dual Stabilization Node for Griptape Nodes Desktop.
Features:
- Dynamic Output: Automatically creates output mask folder in the source video's parent directory.
- Saliency Re-Tracking: Cross-references original video frames to re-track true foreground objects.
- Polarity Verification: Auto-detects and corrects inverted core/background (guarantees white subject, black screen).
- Dual Outputs: Exports BOTH Jitter-Stabilized and Raw (Non-stabilized) 4K sequences & preview videos.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, List

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
from ibkvit_core.matte_fusion import FusionConfig, MatteFusionEngine
from nodes.griptape_compat import DataNode, Parameter, ParameterMode


class VFXMatteRefinerNode(DataNode):
    """
    Node 04: VFX Matte Refiner & Dual Stabilization
    원본 영상과 대조하여 객체를 재추적하고 코어/배경 반전(Polarity)을 자동 교정하며,
    지터 방지가 적용된 안정화 시퀀스(Stabilized)와 적용되지 않은 원본 시퀀스(Raw)를 동시에 생성합니다.
    출력 폴더는 원본 영상이 위치한 폴더 내부에 자동으로 동적 생성됩니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # ── Inputs ───────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="edge_matte_dir",
                type="str",
                default_value="",
                tooltip="IBK 또는 ViT 미세 엣지 알파 마스크 시퀀스 폴더",
                display_name="Edge Matte Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="input_video",
                type="str",
                default_value="",
                tooltip="원본 비디오 경로 (객체 대조 재추적 및 극성 자동 판별에 필수)",
                display_name="Input Video Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="core_matte_dir",
                type="str",
                default_value="",
                tooltip="선택적: 내부 구멍과 자글거림을 방지할 솔리드 코어 마스크 폴더",
                display_name="Core Matte Dir (Optional)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="screen_type",
                type="str",
                default_value="green",
                tooltip="배경 스크린 색상 ('green', 'blue')",
                display_name="Screen Type",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="auto_detect_polarity",
                type="bool",
                default_value=True,
                tooltip="코어/배경 반전 자동 감지 및 교정 (피사체=백색, 배경=흑색 보장)",
                display_name="Auto Detect Polarity",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="invert_matte",
                type="bool",
                default_value=False,
                tooltip="수동 마스크 반전 (강제 반전 필요시 True)",
                display_name="Invert Matte",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="temporal_smoothing",
                type="float",
                default_value=0.0,
                tooltip="시간축 스무딩 가중치 (0.0=완전 프레임 독립/누적 없음, 높을수록 떨림 억제)",
                display_name="Temporal Smoothing (0.0=Independent)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_resolution",
                type="str",
                default_value="4k",
                tooltip="출력 해상도 ('4k': 3840x2160 UHD, 'native': 원본 해상도)",
                display_name="Output Resolution",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_dir",
                type="str",
                default_value="",
                tooltip="결과 저장 폴더 (비워두면 원본 비디오가 위치한 폴더 내부에 자동 동적 생성)",
                display_name="Output Directory (Auto if empty)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ── Outputs ──────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="stabilized_matte_dir",
                type="str",
                tooltip="지터 방지가 적용된 안정화 4K 알파 마스크 폴더",
                display_name="Stabilized Matte Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="raw_matte_dir",
                type="str",
                tooltip="지터 방지가 미적용된 원본 4K 알파 마스크 폴더",
                display_name="Raw Matte Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="stabilized_video_path",
                type="str",
                tooltip="지터 방지가 적용된 알파 검수 비디오 경로",
                display_name="Stabilized Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="raw_video_path",
                type="str",
                tooltip="지터 방지가 미적용된 원본 알파 검수 비디오 경로",
                display_name="Raw Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_count",
                type="int",
                default_value=0,
                tooltip="합성 완료된 총 프레임 수",
                display_name="Frame Count",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="status",
                type="str",
                tooltip="작업 완료 상태",
                display_name="Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        edge_dir = str(self.get_parameter_value("edge_matte_dir") or "").strip()
        if not edge_dir or not Path(edge_dir).exists():
            raise ValueError(f"Valid edge_matte_dir is required: '{edge_dir}'")

        input_video = str(self.get_parameter_value("input_video") or "").strip()
        core_dir = str(self.get_parameter_value("core_matte_dir") or "").strip()
        screen_type = str(self.get_parameter_value("screen_type") or "green").lower()
        auto_polarity = bool(self.get_parameter_value("auto_detect_polarity") if self.get_parameter_value("auto_detect_polarity") is not None else True)
        invert_m = bool(self.get_parameter_value("invert_matte") or False)
        ts_val = self.get_parameter_value("temporal_smoothing")
        temporal_a = float(ts_val) if ts_val is not None else 0.0
        res_opt = str(self.get_parameter_value("output_resolution") or "4k").lower()
        out_dir_param = str(self.get_parameter_value("output_dir") or "").strip()

        # Dynamic Output Directory: automatically inside source video's folder
        if input_video and Path(input_video).exists():
            base_out_dir = Path(VideoIO.resolve_output_dir(input_video, custom_output_dir=out_dir_param, subfolder_suffix="refiner_4k"))
        else:
            base_out_dir = Path(out_dir_param) if out_dir_param else Path(edge_dir).parent / "refiner_4k"
            base_out_dir.mkdir(parents=True, exist_ok=True)

        stab_dir = base_out_dir / "stabilized"
        raw_dir = base_out_dir / "raw"
        stab_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        is_4k = res_opt == "4k"
        config = FusionConfig(
            temporal_smoothing_alpha=temporal_a,
            black_clip=0.01,
            white_clip=0.99,
            use_core_fill=True,
            auto_detect_polarity=auto_polarity,
            invert_matte=invert_m,
        )
        engine = MatteFusionEngine(config=config)

        edge_files = sorted(
            list(Path(edge_dir).glob("*.png")) + list(Path(edge_dir).glob("*.exr"))
        )
        if not edge_files:
            raise ValueError(f"No mask images found in '{edge_dir}'")

        core_files = []
        if core_dir and Path(core_dir).exists():
            core_files = sorted(list(Path(core_dir).glob("*.png")) + list(Path(core_dir).glob("*.exr")))

        guide_frames = []
        fps = 24.0
        if input_video and Path(input_video).exists():
            guide_frames = VideoIO.read_frames(input_video, max_frames=len(edge_files))
            fps = VideoIO.get_video_info(input_video)["fps"]

        # Screen type dynamic detection from input video if available
        if len(guide_frames) > 0:
            detected_screen = MatteFusionEngine.auto_detect_screen_type(guide_frames[0])
            if screen_type in ("auto", "") or (screen_type == "green" and detected_screen == "blue"):
                screen_type = detected_screen

        stab_mattes: List[np.ndarray] = []
        raw_mattes: List[np.ndarray] = []
        stab_previews: List[np.ndarray] = []
        raw_previews: List[np.ndarray] = []

        for idx, ef in enumerate(edge_files):
            edge_img = cv2.imread(str(ef), cv2.IMREAD_UNCHANGED)
            if edge_img is None:
                continue

            if edge_img.dtype == np.uint16:
                edge_m = edge_img.astype(np.float32) / 65535.0
            elif edge_img.dtype == np.uint8:
                edge_m = edge_img.astype(np.float32) / 255.0
            else:
                edge_m = edge_img.astype(np.float32)

            if edge_m.ndim == 3:
                edge_m = edge_m[:, :, 0]

            core_m = None
            if core_files:
                # If sequence of core files exists, match by index; if 1 static core mask, hold across all frames!
                c_idx = min(idx, len(core_files) - 1)
                c_img = cv2.imread(str(core_files[c_idx]), cv2.IMREAD_UNCHANGED)
                if c_img is not None:
                    core_m = (c_img.astype(np.float32) / 255.0)
                    if core_m.ndim == 3:
                        core_m = core_m[:, :, 0]

            guide = guide_frames[idx] if idx < len(guide_frames) else None

            # Process DUAL outputs: Jitter-Stabilized and Raw (Non-stabilized)
            m_stab, m_raw = engine.process_frame_dual(
                core_matte=core_m,
                edge_matte=edge_m,
                rgb_guide=guide,
                screen_type=screen_type,
                scale_to_4k=is_4k,
            )

            stab_mattes.append(m_stab)
            raw_mattes.append(m_raw)

            # Save 16-bit PNGs
            ImageSequenceIO.save_png_16bit(m_stab, stab_dir / f"alpha_stabilized.{idx:05d}.png")
            ImageSequenceIO.save_png_16bit(m_raw, raw_dir / f"alpha_raw.{idx:05d}.png")

            # Previews
            s_gray = (m_stab * 255.0).astype(np.uint8)
            r_gray = (m_raw * 255.0).astype(np.uint8)
            if is_4k:
                s_gray = cv2.resize(s_gray, (1920, 1080), interpolation=cv2.INTER_AREA)
                r_gray = cv2.resize(r_gray, (1920, 1080), interpolation=cv2.INTER_AREA)

            stab_previews.append(np.stack([s_gray, s_gray, s_gray], axis=-1))
            raw_previews.append(np.stack([r_gray, r_gray, r_gray], axis=-1))

        stab_video = str(base_out_dir / "stabilized_preview.mp4")
        raw_video = str(base_out_dir / "raw_preview.mp4")

        VideoIO.write_video(stab_previews, stab_video, fps=fps)
        VideoIO.write_video(raw_previews, raw_video, fps=fps)

        self.set_parameter_value("stabilized_matte_dir", str(stab_dir))
        self.set_parameter_value("raw_matte_dir", str(raw_dir))
        self.set_parameter_value("stabilized_video_path", stab_video)
        self.set_parameter_value("raw_video_path", raw_video)
        total_video_frames = VideoIO.get_video_info(input_video)["frame_count"] if input_video and Path(input_video).exists() else len(stab_mattes)
        self.set_parameter_value("frame_count", len(stab_mattes))
        self.set_parameter_value("status", f"Generated {len(stab_mattes)}/{total_video_frames} frames (Stabilized + Raw) in {base_out_dir}")
