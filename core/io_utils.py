"""
core/io_utils.py
VFX I/O Utilities for Video Frames, OpenEXR Sequences, PNGs, and Preview Overlays.
Supports native OpenEXR (32-bit float / 16-bit float), 16-bit PNG, 4K UHD resolution scaling,
dynamic output folder resolution at video source location, and dual (stabilized & raw) sequence exports.
"""

from __future__ import annotations

import os
import sys

# Enable OpenCV OpenEXR support BEFORE importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np


class VideoIO:
    """Video decoding, metadata extraction, dynamic pathing, and encoding utilities."""

    @staticmethod
    def resolve_output_dir(
        input_video_path: str | Path,
        custom_output_dir: Optional[str | Path] = None,
        subfolder_suffix: str = "masks_4k",
    ) -> Path:
        """
        Dynamically resolve output directory.
        If custom_output_dir is provided and non-empty, uses it.
        Otherwise, automatically creates the mask folder directly inside the
        parent folder where the source video resides:
            e.g. "D:/videos/shot01/clip.mp4" -> "D:/videos/shot01/clip_masks_4k/"
        """
        if custom_output_dir and str(custom_output_dir).strip():
            out_p = Path(custom_output_dir).resolve()
        else:
            vid_p = Path(input_video_path).resolve()
            out_p = vid_p.parent / f"{vid_p.stem}_{subfolder_suffix}"

        out_p.mkdir(parents=True, exist_ok=True)
        return out_p

    @staticmethod
    def get_video_info(video_path: str | Path) -> dict:
        """Extract metadata (FPS, frame count, width, height) from video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        }

    @staticmethod
    def read_frames(
        video_path: str | Path,
        max_frames: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """
        Read all frames from video into a list of RGB uint8 images.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to read video: {video_path}")

        frames: List[np.ndarray] = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_resolution is not None:
                w, h = target_resolution
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

            frames.append(rgb)
            count += 1
            if max_frames and count >= max_frames:
                break

        cap.release()
        return frames

    @staticmethod
    def write_video(
        frames: List[np.ndarray],
        output_path: str | Path,
        fps: float = 24.0,
        codec: str = "mp4v",
    ) -> str:
        """Encode RGB frame list into video."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not frames:
            raise ValueError("No frames provided to write_video")

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()
        return str(output_path)

    @staticmethod
    def create_red_overlay_video(
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        output_path: str | Path,
        fps: float = 24.0,
        overlay_color: Tuple[int, int, int] = (255, 30, 30),
        alpha_weight: float = 0.5,
    ) -> str:
        """
        Create red/tinted mask inspection preview video for visual QA.
        Automatically aligns dimensions between video frame and mask.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not frames or not masks:
            return ""

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        color_arr = np.array(overlay_color, dtype=np.float32)

        for frame, mask in zip(frames, masks):
            frame_f = frame.astype(np.float32)

            if mask.shape[:2] != (h, w):
                m_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
            else:
                m_resized = mask

            m = np.clip(m_resized, 0.0, 1.0)[:, :, None]
            overlay = frame_f * (1.0 - m * alpha_weight) + color_arr * (m * alpha_weight)
            preview_rgb = np.clip(overlay, 0, 255).astype(np.uint8)
            preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
            writer.write(preview_bgr)

        writer.release()
        return str(output_path)


class ImageSequenceIO:
    """Industry standard image sequence export (OpenEXR, PNG, TIFF) with 4K support."""

    @staticmethod
    def save_exr(
        image: np.ndarray,
        filepath: str | Path,
        is_alpha_only: bool = True,
    ) -> None:
        """
        Save single frame as 32-bit floating point OpenEXR.
        Supports native OpenEXR module, OpenCV imgcodecs, or imageio fallback.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        img_f32 = image.astype(np.float32)
        h, w = img_f32.shape[:2]

        # 1. Native OpenEXR module
        try:
            import OpenEXR
            import Imath

            header = OpenEXR.Header(w, h)
            float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

            if (is_alpha_only or img_f32.ndim == 2) and img_f32.ndim == 2:
                header["channels"] = {"A": float_chan}
                exr_out = OpenEXR.OutputFile(str(filepath), header)
                exr_out.writePixels({"A": img_f32.tobytes()})
                exr_out.close()
                if filepath.exists():
                    return
            elif img_f32.ndim == 3 and img_f32.shape[-1] == 3:
                header["channels"] = {"R": float_chan, "G": float_chan, "B": float_chan}
                r = img_f32[:, :, 0].tobytes()
                g = img_f32[:, :, 1].tobytes()
                b = img_f32[:, :, 2].tobytes()
                exr_out = OpenEXR.OutputFile(str(filepath), header)
                exr_out.writePixels({"R": r, "G": g, "B": b})
                exr_out.close()
                if filepath.exists():
                    return
            elif img_f32.ndim == 3 and img_f32.shape[-1] == 4:
                header["channels"] = {"R": float_chan, "G": float_chan, "B": float_chan, "A": float_chan}
                r = img_f32[:, :, 0].tobytes()
                g = img_f32[:, :, 1].tobytes()
                b = img_f32[:, :, 2].tobytes()
                a = img_f32[:, :, 3].tobytes()
                exr_out = OpenEXR.OutputFile(str(filepath), header)
                exr_out.writePixels({"R": r, "G": g, "B": b, "A": a})
                exr_out.close()
                if filepath.exists():
                    return
        except Exception:
            pass

        # 2. Try OpenCV cv2.imwrite
        try:
            if is_alpha_only and img_f32.ndim == 2:
                success = cv2.imwrite(str(filepath), img_f32)
            elif img_f32.shape[-1] == 3:
                bgr = cv2.cvtColor(img_f32, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(str(filepath), bgr)
            elif img_f32.shape[-1] == 4:
                bgra = cv2.cvtColor(img_f32, cv2.COLOR_RGBA2BGRA)
                success = cv2.imwrite(str(filepath), bgra)
            else:
                success = cv2.imwrite(str(filepath), img_f32)

            if success and filepath.exists():
                return
        except Exception:
            pass

        # 3. Fallback to imageio
        try:
            import imageio.v3 as iio
            iio.imwrite(filepath, img_f32)
            if filepath.exists():
                return
        except Exception:
            pass

        # 4. Fallback to 16-bit PNG
        fallback_png = filepath.with_suffix(".png")
        ImageSequenceIO.save_png_16bit(image, fallback_png)

    @staticmethod
    def save_png_16bit(
        matte: np.ndarray,
        filepath: str | Path,
    ) -> None:
        """Save alpha matte as lossless 16-bit PNG."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        m = np.clip(matte, 0.0, 1.0)
        m_16 = (m * 65535.0).astype(np.uint16)
        cv2.imwrite(str(filepath), m_16)

    @staticmethod
    def export_sequence(
        matte_sequence: List[np.ndarray],
        output_dir: str | Path,
        prefix: str = "matte",
        format_type: str = "exr",
        rgb_plates: Optional[List[np.ndarray]] = None,
        resolution: str = "4k",
    ) -> List[str]:
        """
        Export list of mattes as an image sequence in 4K or native resolution.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        file_paths: List[str] = []
        fmt = format_type.lower()
        use_4k = resolution.lower() == "4k"

        for idx, m in enumerate(matte_sequence):
            filename = f"{prefix}.{idx:05d}"

            if use_4k:
                h, w = m.shape[:2]
                if w != 3840:
                    scale = 3840.0 / float(w)
                    target_h = int(round(h * scale))
                    if target_h % 2 != 0:
                        target_h += 1
                    m_scaled = cv2.resize(m, (3840, target_h), interpolation=cv2.INTER_LANCZOS4)
                    m_scaled[m_scaled > 0.99] = 1.0
                    m_scaled[m_scaled < 0.01] = 0.0
                    m_out = np.clip(m_scaled, 0.0, 1.0).astype(np.float32)
                else:
                    m_out = m
            else:
                m_out = m

            if fmt == "exr":
                fpath = out_path / f"{filename}.exr"
                ImageSequenceIO.save_exr(m_out, fpath, is_alpha_only=True)
            elif fmt == "rgba_exr" and rgb_plates is not None and idx < len(rgb_plates):
                fpath = out_path / f"{filename}.exr"
                rgb = rgb_plates[idx].astype(np.float32) / 255.0
                if rgb.shape[:2] != m_out.shape[:2]:
                    rgb = cv2.resize(rgb, (m_out.shape[1], m_out.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                alpha = np.clip(m_out, 0.0, 1.0)[:, :, None]
                rgba = np.concatenate([rgb * alpha, alpha], axis=-1)
                ImageSequenceIO.save_exr(rgba, fpath, is_alpha_only=False)
            elif fmt == "png16":
                fpath = out_path / f"{filename}.png"
                ImageSequenceIO.save_png_16bit(m_out, fpath)
            else:  # png8
                fpath = out_path / f"{filename}.png"
                m_8 = (np.clip(m_out, 0.0, 1.0) * 255.0).astype(np.uint8)
                cv2.imwrite(str(fpath), m_8)

            file_paths.append(str(fpath))

        return file_paths
