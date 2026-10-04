"""
nodes/ibk_keyer_node.py
Node 02: IBK Keyer Node for Griptape Nodes Desktop.
Emulates Nuke IBKGizmo: pulls fine transmission alpha matte (hair, motion blur, semi-transparency).
Focuses strictly on high-precision alpha mask extraction with pure black background and pure white core.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

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
from ibkvit_core.matte_fusion import MatteFusionEngine
from ibkvit_core.griptape_compat import DataNode, Parameter, ParameterMode


class IBKKeyerNode(DataNode):
    """
    Node 02: IBK Keyer (IBKGizmo)
    원본 영상과 Clean Plate를 비교하여 미세 머리카락, 모션 블러, 반투명 디테일을 보존하는
    IBK 알파 마스크를 4K 해상도로 정밀 추출합니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # ── Inputs ───────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="input_video",
                type="str",
                default_value="",
                tooltip="입력 원본 비디오 파일 절대 경로",
                display_name="Input Video Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="clean_plate_dir",
                type="str",
                default_value="",
                tooltip="Clean Plate 폴더 경로 (비워두면 내부에서 자동 생성)",
                display_name="Clean Plate Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="screen_type",
                type="str",
                default_value="auto",
                tooltip="스크린 타입 ('auto': 영상 자동 감지, 'green', 'blue')",
                display_name="Screen Type",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="red_weight",
                type="float",
                default_value=0.5,
                tooltip="적색 채널 가중치 (기본 0.5)",
                display_name="Red Weight",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="blue_weight",
                type="float",
                default_value=0.5,
                tooltip="청색 채널 가중치 (기본 0.5)",
                display_name="Blue Weight",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="gamma",
                type="float",
                default_value=1.0,
                tooltip="알파 마스크 감마 커브 조절 (1.0 = 선형)",
                display_name="Matte Gamma",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="black_clip",
                type="float",
                default_value=0.01,
                tooltip="배경 노이즈 완벽 제거를 위한 퓨어 블랙 클립",
                display_name="Black Clip (Pure Black)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="white_clip",
                type="float",
                default_value=0.99,
                tooltip="전경 솔리드 고정을 위한 퓨어 화이트 클립",
                display_name="White Clip (Pure White)",
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
                tooltip="출력 결과 저장 디렉토리 (비워두면 원본 비디오 폴더 내 자동 생성)",
                display_name="Output Directory",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ── Outputs ──────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="alpha_matte_dir",
                type="str",
                tooltip="추출된 4K 고화질 IBK 알파 마스크(16-bit PNG) 폴더",
                display_name="Alpha Matte Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="matte_preview_video",
                type="str",
                tooltip="알파 마스크 흑백 검수 비디오 파일 경로",
                display_name="Matte Preview Video Path",
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
                tooltip="작업 완료 상태",
                display_name="Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        input_video = str(self.get_parameter_value("input_video") or "").strip()
        if not input_video or not Path(input_video).exists():
            raise ValueError(f"Valid input_video path is required: '{input_video}'")

        clean_dir = str(self.get_parameter_value("clean_plate_dir") or "").strip()
        screen_type = str(self.get_parameter_value("screen_type") or "green").lower()
        red_w = float(self.get_parameter_value("red_weight") or 0.5)
        blue_w = float(self.get_parameter_value("blue_weight") or 0.5)
        gamma = float(self.get_parameter_value("gamma") or 1.0)
        black_clip = float(self.get_parameter_value("black_clip") or 0.01)
        white_clip = float(self.get_parameter_value("white_clip") or 0.99)
        res_opt = str(self.get_parameter_value("output_resolution") or "4k").lower()
        max_frames = int(self.get_parameter_value("max_frames") or 0)
        out_dir = str(self.get_parameter_value("output_dir") or "").strip()

        out_dir = VideoIO.resolve_output_dir(input_video, custom_output_dir=out_dir, subfolder_suffix="ibk_keyer")
        out_path = Path(out_dir)
        m_path = out_path / "mattes"
        m_path.mkdir(parents=True, exist_ok=True)

        info = VideoIO.get_video_info(input_video)
        fps = info["fps"]
        limit = max_frames if max_frames > 0 else None

        clean_files = []
        if clean_dir and Path(clean_dir).exists():
            clean_files = sorted(list(Path(clean_dir).glob("*.png")) + list(Path(clean_dir).glob("*.exr")))

        # Synchronize edge mask frame count with clean plate sequence if multi-frame sequence is provided
        if max_frames == 0 and len(clean_files) > 1:
            limit = len(clean_files)

        frames = VideoIO.read_frames(input_video, max_frames=limit)

        from ibkvit_core.matte_fusion import MatteFusionEngine
        if len(frames) > 0:
            detected_screen = MatteFusionEngine.auto_detect_screen_type(frames[0])
            if screen_type in ("auto", "") or (screen_type == "green" and detected_screen == "blue"):
                screen_type = detected_screen

        engine = IBKEngine(screen_type=screen_type, red_weight=red_w, blue_weight=blue_w)

        clean_frames = []
        if clean_files:
            for cf in clean_files[: len(frames)]:
                c_bgr = cv2.imread(str(cf))
                if c_bgr is not None:
                    clean_frames.append(cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB))

        mattes = []
        matte_previews = []
        is_4k = res_opt == "4k"

        for idx, frame in enumerate(frames):
            # If clean_plate sequence is provided, match by index; if static 1-frame plate, hold across all frames!
            cp = clean_frames[min(idx, len(clean_frames) - 1)] if clean_frames else None

            res = engine.execute_keying(
                rgb_image=frame,
                clean_plate=cp,
                gamma=gamma,
                black_clip=black_clip,
                white_clip=white_clip,
            )

            # 4K resolution upscaling if requested
            if is_4k:
                alpha_out = MatteFusionEngine.upscale_to_4k(res.alpha)
            else:
                alpha_out = res.alpha

            mattes.append(alpha_out)

            # Save 16-bit PNG matte
            ImageSequenceIO.save_png_16bit(alpha_out, m_path / f"alpha.{idx:05d}.png")

            m_gray = (alpha_out * 255.0).astype(np.uint8)
            # Create preview in HD to keep video size manageable
            m_small = cv2.resize(m_gray, (1920, 1080), interpolation=cv2.INTER_AREA) if is_4k else m_gray
            matte_previews.append(np.stack([m_small, m_small, m_small], axis=-1))

        matte_video = str(out_path / "matte_preview.mp4")
        VideoIO.write_video(matte_previews, matte_video, fps=fps)

        self.set_parameter_value("alpha_matte_dir", str(m_path))
        self.set_parameter_value("matte_preview_video", matte_video)
        self.set_parameter_value("frame_count", len(mattes))
        self.set_parameter_value("status", f"Keyed {len(mattes)} frames successfully in {res_opt.upper()}.")


# Compositing Architecture alias
CompMatteKeyerNode = IBKKeyerNode

__all__ = ["IBKKeyerNode", "CompMatteKeyerNode"]

