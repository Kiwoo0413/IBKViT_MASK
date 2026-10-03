"""
nodes/vfx_all_in_one_node.py
All-in-One IBK Keying & ViT Masking Node for Griptape Nodes Desktop.
Features:
- Dynamic Output Directory: Automatically creates mask folder directly in the source video's folder.
- Polarity Auto-Correction: Verifies original video color to ensure white subject and black background.
- Saliency Re-Tracking: Re-anchors object core against the plate.
- Dual 4K Sequences: Generates BOTH Jitter-Stabilized and Raw (Non-stabilized) 4K OpenEXR/PNG sequences & preview videos.
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

from ibkvit_core.ibk_engine import IBKEngine, ScreenType
from ibkvit_core.io_utils import ImageSequenceIO, VideoIO
from ibkvit_core.matte_fusion import FusionConfig, MatteFusionEngine
from ibkvit_core.vit_engine import ViTEngine
from nodes.griptape_compat import DataNode, Parameter, ParameterMode
from nodes.vit_mask_extractor_node import parse_box, parse_coords


class VFXKeyingViTAllInOneNode(DataNode):
    """
    VFX IBK & ViT Keyer (All-in-One)
    - IBK 브랜치: 광학적 컬러 차이 기반으로 머리카락, 모션블러, 서브픽셀 투명도 엣지를 원본 그대로 보존합니다.
    - ViT 브랜치: 시공간 추적을 통해 내부 코어(Pure White 1.0)와 외부 배경(Pure Black 0.0)의 지터를 제거하고 시간축으로 안정화합니다.
    - Refine & Fusion: 엣지 왜곡을 유발하는 무리한 지터 필터를 배제하고 안정화된 코어/배경 엔벨로프와 IBK 엣지를 클린 합성합니다.
    - 경량 Grayscale 알파 출력: 불필요한 RGB 색상 오버헤드 없이 단일 채널 4K UHD 32-bit Float EXR 및 16-bit PNG 시퀀스를 출력합니다.
    - 동적 폴더: 원본 영상이 위치한 폴더 내부에 자동으로 mask 폴더를 동적 생성합니다.
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
                name="screen_type",
                type="str",
                default_value="auto",
                tooltip="배경 스크린 타입 ('auto': 인풋 영상 자동 감지, 'green', 'blue')",
                display_name="Screen Type",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="seed_coords",
                type="str",
                default_value="",
                tooltip="선택적: 전경 피사체 클릭 좌표 (예: '640,360')",
                display_name="Seed Coords (Optional)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="box_coords",
                type="str",
                default_value="",
                tooltip="선택적: 바운딩 박스 (예: '100,100,500,500')",
                display_name="Box Coords (Optional)",
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
                name="enable_adaptive_blur",
                type="bool",
                default_value=True,
                tooltip="디포커스/모션블러 정도를 자동 감지하여 마스크 경계 폭을 가변 조절",
                display_name="Adaptive Blur Detection",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="use_vitmatte_refinement",
                type="bool",
                default_value=True,
                tooltip="엣지 전이 영역(Unknown Zone)에 ViTMatte 신경망 서브픽셀 정밀 추론 적용 (True: 고품질 신경망-광학 하이브리드 모드, False: 초고속 60fps 순수 광학 융합 모드)",
                display_name="Use ViTMatte Refinement",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_resolution",
                type="str",
                default_value="4k",
                tooltip="출력 해상도 ('4k': 3840x2160 UHD 업스케일링, 'native': 인풋 영상 해상도 유지)",
                display_name="Output Resolution",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="export_format",
                type="str",
                default_value="exr",
                tooltip="출력 시퀀스 포맷 ('exr': 단일 채널 32-bit Float EXR, 'png16': 단일 채널 16-bit PNG, 'png8': 단일 채널 8-bit PNG)",
                display_name="Export Format",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="max_frames",
                type="int",
                default_value=0,
                tooltip="처리할 최대 프레임 수 (0 = 인풋 영상 전체 프레임 자동 감지)",
                display_name="Max Frames",
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
                name="stabilized_sequence_dir",
                type="str",
                tooltip="지터 방지가 적용된 안정화 4K 알파 마스크 시퀀스 폴더 (EXR/PNG)",
                display_name="Stabilized Sequence Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="raw_sequence_dir",
                type="str",
                tooltip="지터 방지가 미적용된 원본 4K 알파 마스크 시퀀스 폴더 (EXR/PNG)",
                display_name="Raw Sequence Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="stabilized_video_path",
                type="str",
                tooltip="지터 방지가 적용된 흑백 알파 마스크 비디오 경로",
                display_name="Stabilized Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="raw_video_path",
                type="str",
                tooltip="지터 방지가 미적용된 원본 흑백 알파 마스크 비디오 경로",
                display_name="Raw Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="red_overlay_video_path",
                type="str",
                tooltip="빨간색 마스크 검수용 오버레이 MP4 비디오 경로",
                display_name="Red Overlay Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_count",
                type="int",
                default_value=0,
                tooltip="처리 완료된 총 프레임 수",
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
        try:
            from ibkvit_core.griptape_model_manager import ensure_all_models_ready
            ensure_all_models_ready(auto_download=True)
        except Exception:
            pass

        input_video = str(self.get_parameter_value("input_video") or "").strip()
        if not input_video or not Path(input_video).exists():
            raise ValueError(f"Valid input_video path is required: '{input_video}'")

        screen_type = str(self.get_parameter_value("screen_type") or "green").lower()
        seed_str = str(self.get_parameter_value("seed_coords") or "").strip()
        box_str = str(self.get_parameter_value("box_coords") or "").strip()
        auto_polarity = bool(self.get_parameter_value("auto_detect_polarity") if self.get_parameter_value("auto_detect_polarity") is not None else True)
        invert_m = bool(self.get_parameter_value("invert_matte") or False)
        ts_val = self.get_parameter_value("temporal_smoothing")
        temporal_a = float(ts_val) if ts_val is not None else 0.0
        res_opt = str(self.get_parameter_value("output_resolution") or "native").lower()
        export_fmt = str(self.get_parameter_value("export_format") or "exr").lower()
        max_frames = int(self.get_parameter_value("max_frames") or 0)
        out_dir_param = str(self.get_parameter_value("output_dir") or "").strip()
        enable_adaptive_blur = bool(self.get_parameter_value("enable_adaptive_blur") if self.get_parameter_value("enable_adaptive_blur") is not None else True)
        use_vit_refine = bool(self.get_parameter_value("use_vitmatte_refinement") if self.get_parameter_value("use_vitmatte_refinement") is not None else True)

        seed_points = parse_coords(seed_str)
        box_coords = parse_box(box_str)

        info = VideoIO.get_video_info(input_video)
        fps = info["fps"]
        in_w = info["width"]
        in_h = info["height"]
        total_frames = info["frame_count"]
        limit = max_frames if max_frames > 0 else None

        frames = VideoIO.read_frames(input_video, max_frames=limit)
        num_frames = len(frames)
        is_4k = res_opt == "4k"

        # Screen type dynamic detection from input video
        if num_frames > 0:
            detected_screen = MatteFusionEngine.auto_detect_screen_type(frames[0])
            if screen_type in ("auto", "") or (screen_type == "green" and detected_screen == "blue"):
                screen_type = detected_screen

        # Dynamic Output Directory: automatically inside source video's folder
        base_out_dir = Path(VideoIO.resolve_output_dir(input_video, custom_output_dir=out_dir_param, subfolder_suffix="masks_4k"))

        stab_seq_dir = base_out_dir / "stabilized" / export_fmt
        raw_seq_dir = base_out_dir / "raw" / export_fmt
        stab_seq_dir.mkdir(parents=True, exist_ok=True)
        raw_seq_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Engines
        ibk_eng = IBKEngine(screen_type=screen_type)
        vit_eng = ViTEngine()
        fusion_eng = MatteFusionEngine(
            FusionConfig(
                temporal_smoothing_alpha=temporal_a,
                black_clip=0.01,
                white_clip=0.99,
                use_core_fill=True,
                auto_detect_polarity=auto_polarity,
                invert_matte=invert_m,
            )
        )

        # 2. ViT Branch: Track and spatio-temporally de-jitter inner core & outer background envelope
        # Eliminates jitter on inner core (100% white) and background (100% black) without touching hair edges
        stab_cores, stab_envs, raw_cores, raw_envs = vit_eng.track_and_stabilize_stream(
            frame_sequence=frames,
            seed_points=seed_points,
            box_coords=box_coords,
            screen_type=screen_type,
            temporal_factor=temporal_a,
            enable_adaptive_blur=enable_adaptive_blur,
        )

        stab_mattes: List[np.ndarray] = []
        raw_mattes: List[np.ndarray] = []
        stab_previews: List[np.ndarray] = []
        raw_previews: List[np.ndarray] = []

        # 3. Main Frame Processing Loop
        for idx, frame in enumerate(frames):
            # A. IBK Branch: Extracts pristine optical edge transmission matte (hair, motion blur, transparency)
            ibk_res = ibk_eng.execute_keying(rgb_image=frame)
            current_edge = ibk_res.alpha

            # B. ViT Cores & Envelopes for this frame
            c_stab = stab_cores[idx] if idx < len(stab_cores) else None
            env_stab = stab_envs[idx] if idx < len(stab_envs) else None
            c_raw = raw_cores[idx] if idx < len(raw_cores) else None
            env_raw = raw_envs[idx] if idx < len(raw_envs) else None

            # B-2. Optional ViTMatte Neural Edge Refinement on Transition Zone
            if use_vit_refine:
                coarse_list = getattr(vit_eng, "last_coarse_masks", None)
                if coarse_list and idx < len(coarse_list):
                    cm = coarse_list[idx]
                elif env_raw is not None:
                    cm = env_raw
                else:
                    cm = frame
                vit_res = vit_eng.extract_vit_matte(
                    rgb_image=frame,
                    coarse_mask=cm,
                    enable_adaptive_blur=enable_adaptive_blur,
                    enable_roi_crop=True,
                    screen_type=screen_type,
                )
                # Complementary Fusion in transition zone:
                # ViTMatte provides the semantic boundary without background noise,
                # while IBK preserves pristine optical light transmission on fine hair strands.
                current_edge = np.clip(0.5 * ibk_res.alpha + 0.5 * vit_res.alpha, 0.0, 1.0)

            # C. Non-destructive Matte Fusion:
            # - Inner core is 100% pure white (solid, no holes, de-jittered by ViT)
            # - Outer background is 100% pure black (no screen noise, de-jittered by ViT)
            # - Transition zone is pristine hybrid edge transmission
            # - No critical/aggressive temporal edge hacks in fusion!
            m_stab, m_raw = fusion_eng.process_frame_dual(
                core_matte=c_stab,
                edge_matte=current_edge,
                envelope_matte=env_stab,
                raw_core_matte=c_raw,
                raw_envelope_matte=env_raw,
                rgb_guide=frame,
                screen_type=screen_type,
                scale_to_4k=is_4k,
            )

            stab_mattes.append(m_stab)
            raw_mattes.append(m_raw)

            # Previews (lightweight 1080p for UI inspection)
            s_gray = (m_stab * 255.0).astype(np.uint8)
            r_gray = (m_raw * 255.0).astype(np.uint8)
            if is_4k:
                s_gray = cv2.resize(s_gray, (1920, 1080), interpolation=cv2.INTER_AREA)
                r_gray = cv2.resize(r_gray, (1920, 1080), interpolation=cv2.INTER_AREA)

            stab_previews.append(np.stack([s_gray, s_gray, s_gray], axis=-1))
            raw_previews.append(np.stack([r_gray, r_gray, r_gray], axis=-1))

        vit_eng.release_memory()

        # 4. Export Sequences: Both Stabilized and Raw in 4K UHD
        ImageSequenceIO.export_sequence(
            matte_sequence=stab_mattes,
            output_dir=stab_seq_dir,
            prefix="alpha_stabilized_4k",
            format_type=export_fmt,
            rgb_plates=frames,
            resolution="native",
        )
        ImageSequenceIO.export_sequence(
            matte_sequence=raw_mattes,
            output_dir=raw_seq_dir,
            prefix="alpha_raw_4k",
            format_type=export_fmt,
            rgb_plates=frames,
            resolution="native",
        )

        stab_video = str(base_out_dir / "stabilized_preview.mp4")
        raw_video = str(base_out_dir / "raw_preview.mp4")
        red_video = str(base_out_dir / "red_overlay_preview.mp4")

        VideoIO.write_video(stab_previews, stab_video, fps=fps)
        VideoIO.write_video(raw_previews, raw_video, fps=fps)
        VideoIO.create_red_overlay_video(
            frames=frames,
            masks=stab_mattes,
            output_path=red_video,
            fps=fps,
        )

        self.set_parameter_value("stabilized_sequence_dir", str(stab_seq_dir))
        self.set_parameter_value("raw_sequence_dir", str(raw_seq_dir))
        self.set_parameter_value("stabilized_video_path", stab_video)
        self.set_parameter_value("raw_video_path", raw_video)
        self.set_parameter_value("red_overlay_video_path", red_video)
        self.set_parameter_value("frame_count", num_frames)
        res_display = "4K UHD" if is_4k else f"{in_w}x{in_h}"
        self.set_parameter_value(
            "status",
            f"Completed: {num_frames}/{total_frames} frames ({res_display} @ {fps:.2f}fps, Screen: {screen_type}) in {base_out_dir}",
        )
