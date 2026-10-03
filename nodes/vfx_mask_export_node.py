"""
nodes/vfx_mask_export_node.py
Node 05: VFX Mask & Sequence Exporter Node for Griptape Nodes Desktop.
Exports industry-standard OpenEXR sequences (32-bit float / RGBA), 16-bit PNGs, and QA preview overlays
in 4K UHD resolution.
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

from ibkvit_core.io_utils import ImageSequenceIO, VideoIO
from nodes.griptape_compat import DataNode, Parameter, ParameterMode


class VFXMaskExportNode(DataNode):
    """
    Node 05: VFX Mask & Sequence Exporter
    완성된 알파 마스크를 4K UHD(3840x2160) 해상도로 Nuke, Fusion, After Effects, DaVinci Resolve 등
    프로 VFX 툴과 100% 호환되는 OpenEXR(단일 채널/프리멀티플라이드 RGBA) 또는 16-bit 무손실 PNG 시퀀스로 내보냅니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # ── Inputs ───────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="matte_dir",
                type="str",
                default_value="",
                tooltip="입력 알파 마스크 시퀀스 폴더 경로",
                display_name="Matte Sequence Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="input_video",
                type="str",
                default_value="",
                tooltip="선택적: RGBA 합성 또는 검수 오버레이용 원본 비디오 경로",
                display_name="Input Video Path (Optional)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="export_format",
                type="str",
                default_value="exr",
                tooltip="내보내기 포맷 ('exr': 단일 채널 32-bit Float EXR, 'png16': 단일 채널 16-bit PNG, 'png8': 단일 채널 8-bit PNG)",
                display_name="Export Format",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_resolution",
                type="str",
                default_value="4k",
                tooltip="내보내기 해상도 ('4k': 3840x2160 UHD, 'native': 원본 해상도)",
                display_name="Output Resolution",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_dir",
                type="str",
                default_value="",
                tooltip="저장 대상 폴더 (비워두면 원본 비디오 또는 마스크 경로 하위 폴더에 자동 생성)",
                display_name="Output Directory",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="sequence_prefix",
                type="str",
                default_value="alpha_matte_4k",
                tooltip="시퀀스 파일명 접두사 (예: alpha_matte_4k.00000.exr)",
                display_name="Sequence Prefix",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ── Outputs ──────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="exported_dir",
                type="str",
                tooltip="내보내기 완료된 4K 시퀀스 폴더 경로",
                display_name="Exported Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="red_overlay_video_path",
                type="str",
                tooltip="빨간색 검수 오버레이 비디오 파일 경로",
                display_name="Red Overlay Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_count",
                type="int",
                default_value=0,
                tooltip="내보내기 완료된 총 프레임 수",
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
        matte_dir = str(self.get_parameter_value("matte_dir") or "").strip()
        if not matte_dir or not Path(matte_dir).exists():
            raise ValueError(f"Valid matte_dir is required: '{matte_dir}'")

        input_video = str(self.get_parameter_value("input_video") or "").strip()
        export_fmt = str(self.get_parameter_value("export_format") or "exr").lower()
        res_opt = str(self.get_parameter_value("output_resolution") or "4k").lower()
        prefix = str(self.get_parameter_value("sequence_prefix") or "alpha_matte_4k")
        out_dir = str(self.get_parameter_value("output_dir") or "").strip()

        if not out_dir:
            if input_video and Path(input_video).exists():
                out_dir = VideoIO.resolve_output_dir(input_video, subfolder_suffix="vfx_export_4k")
            else:
                out_dir = str(Path(matte_dir).parent / "vfx_export_4k")

        out_path = Path(out_dir)
        seq_out_dir = out_path / export_fmt
        seq_out_dir.mkdir(parents=True, exist_ok=True)

        matte_files = sorted(
            list(Path(matte_dir).glob("*.png")) + list(Path(matte_dir).glob("*.exr"))
        )
        if not matte_files:
            raise ValueError(f"No matte files found in '{matte_dir}'")

        mattes = []
        for mf in matte_files:
            img = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.dtype == np.uint16:
                    m = img.astype(np.float32) / 65535.0
                elif img.dtype == np.uint8:
                    m = img.astype(np.float32) / 255.0
                else:
                    m = img.astype(np.float32)
                if m.ndim == 3:
                    m = m[:, :, 0]
                mattes.append(m)

        rgb_plates = None
        fps = 24.0
        if input_video and Path(input_video).exists():
            rgb_plates = VideoIO.read_frames(input_video, max_frames=len(mattes))
            fps = VideoIO.get_video_info(input_video)["fps"]

        # Export image sequence in 4K UHD
        written_files = ImageSequenceIO.export_sequence(
            matte_sequence=mattes,
            output_dir=seq_out_dir,
            prefix=prefix,
            format_type=export_fmt,
            rgb_plates=rgb_plates,
            resolution=res_opt,
        )

        # Generate Red Overlay Video if RGB plates exist
        red_video = ""
        if rgb_plates and len(rgb_plates) == len(mattes):
            red_video = str(out_path / f"{prefix}_red_overlay.mp4")
            VideoIO.create_red_overlay_video(
                frames=rgb_plates,
                masks=mattes,
                output_path=red_video,
                fps=fps,
            )

        self.set_parameter_value("exported_dir", str(seq_out_dir))
        self.set_parameter_value("red_overlay_video_path", red_video)
        self.set_parameter_value("frame_count", len(written_files))
        self.set_parameter_value(
            "status", f"Successfully exported {len(written_files)} frames in {export_fmt.upper()} ({res_opt.upper()})."
        )
