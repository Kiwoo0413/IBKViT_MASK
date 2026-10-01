"""
nodes/ibk_clean_plate_node.py
Node 01: IBK Clean Plate Generator for Griptape Nodes Desktop.
Generates backing screen clean plates (IBKColour) from video or image footage.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from core.ibk_engine import IBKEngine, ScreenType
from core.io_utils import ImageSequenceIO, VideoIO
from nodes.griptape_compat import DataNode, Parameter, ParameterMode


class IBKCleanPlateNode(DataNode):
    """
    Node 01: IBK Clean Plate Generator (IBKColour)
    입력 비디오 또는 이미지에서 전경 피사체를 제거하고,
    배경 스크린(그린/블루/커스텀)의 자연스러운 조명 그라데이션을 복원한 Clean Plate를 생성합니다.
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
                name="screen_type",
                type="str",
                default_value="auto",
                tooltip="스크린 타입 ('auto': 영상 자동 감지, 'green', 'blue', 'custom')",
                display_name="Screen Type",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="patch_size",
                type="int",
                default_value=5,
                tooltip="스크린 확장 패치 크기 (작을수록 정밀, 클수록 넓은 영역 채움)",
                display_name="Patch Size",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="blur_radius",
                type="int",
                default_value=15,
                tooltip="클린 플레이트 조명 스무딩 블러 반경",
                display_name="Blur Radius",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="iterations",
                type="int",
                default_value=4,
                tooltip="다단계 피라미드 인페인팅 반복 횟수",
                display_name="Iterations",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="max_frames",
                type="int",
                default_value=0,
                tooltip="처리할 최대 프레임 수 (0 = 전체 프레임)",
                display_name="Max Frames",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_dir",
                type="str",
                default_value="",
                tooltip="클린 플레이트 시퀀스 저장 디렉토리 (비워두면 원본 비디오 폴더 내 자동 생성)",
                display_name="Output Directory",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ── Outputs ──────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="clean_plate_dir",
                type="str",
                tooltip="생성된 Clean Plate 시퀀스 폴더 경로",
                display_name="Clean Plate Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="clean_plate_video",
                type="str",
                tooltip="생성된 Clean Plate 미리보기 MP4 비디오 경로",
                display_name="Clean Plate Video Path",
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
                tooltip="실행 상태 요약",
                display_name="Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        input_video = str(self.get_parameter_value("input_video") or "").strip()
        if not input_video or not Path(input_video).exists():
            raise ValueError(f"Valid input_video path is required. Given: '{input_video}'")

        screen_type = str(self.get_parameter_value("screen_type") or "green").lower()
        patch_size = int(self.get_parameter_value("patch_size") or 5)
        blur_radius = int(self.get_parameter_value("blur_radius") or 15)
        iterations = int(self.get_parameter_value("iterations") or 4)
        max_frames = int(self.get_parameter_value("max_frames") or 0)
        out_dir = str(self.get_parameter_value("output_dir") or "").strip()

        out_dir = VideoIO.resolve_output_dir(input_video, custom_output_dir=out_dir, subfolder_suffix="clean_plate")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        info = VideoIO.get_video_info(input_video)
        fps = info["fps"]
        limit = max_frames if max_frames > 0 else None

        frames = VideoIO.read_frames(input_video, max_frames=limit)

        from core.matte_fusion import MatteFusionEngine
        if len(frames) > 0:
            detected_screen = MatteFusionEngine.auto_detect_screen_type(frames[0])
            if screen_type in ("auto", "") or (screen_type == "green" and detected_screen == "blue"):
                screen_type = detected_screen

        engine = IBKEngine(screen_type=screen_type)
        clean_frames = []

        import cv2

        for idx, frame in enumerate(frames):
            cp = engine.generate_clean_plate(
                frame,
                patch_size=patch_size,
                blur_radius=blur_radius,
                iterations=iterations,
            )
            clean_frames.append(cp)
            cv2.imwrite(str(out_path / f"clean_plate.{idx:05d}.png"), cv2.cvtColor(cp, cv2.COLOR_RGB2BGR))

        video_preview = str(out_path / "clean_plate_preview.mp4")
        VideoIO.write_video(clean_frames, video_preview, fps=fps)

        total_video_frames = info.get("frame_count", len(frames))
        self.set_parameter_value("clean_plate_dir", str(out_path))
        self.set_parameter_value("clean_plate_video", video_preview)
        self.set_parameter_value("frame_count", len(clean_frames))
        self.set_parameter_value("status", f"Generated {len(clean_frames)}/{total_video_frames} clean plate frames.")
