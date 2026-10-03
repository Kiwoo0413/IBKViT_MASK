"""
examples/run_ibk_vit_pipeline.py
Standalone CLI runner for VFX IBK Keying & ViT Masking Toolkit.
Extracts 4K UHD alpha matte sequences with pure-white core and pure-black background.
Can be run directly via:
    python examples/run_ibk_vit_pipeline.py --input "path/to/video.mp4" --screen green --resolution 4k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ibkvit_core.ibk_engine import IBKEngine, ScreenType
from ibkvit_core.io_utils import ImageSequenceIO, VideoIO
from ibkvit_core.matte_fusion import FusionConfig, MatteFusionEngine
from ibkvit_core.vit_engine import ViTEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="VFX IBK Keying & ViT 4K Masking Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video file path")
    parser.add_argument(
        "--screen", "-s", type=str, default="green", choices=["green", "blue", "custom"], help="Backing screen color"
    )
    parser.add_argument(
        "--seed", type=str, default="", help="Seed coordinate 'x,y' for foreground subject"
    )
    parser.add_argument(
        "--format", "-f", type=str, default="exr", choices=["exr", "rgba_exr", "png16", "png8"], help="Export format"
    )
    parser.add_argument(
        "--resolution", "-r", type=str, default="4k", choices=["4k", "native"], help="Export resolution (4k UHD: 3840x2160 or native)"
    )
    parser.add_argument("--output", "-o", type=str, default="", help="Output directory")
    parser.add_argument("--max-frames", "-m", type=int, default=0, help="Max frames to process (0 = all)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input video '{input_path}' not found.")
        sys.exit(1)

    out_dir = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_vfx_4k_matte"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"================================================================")
    print(f"🎬 VFX IBK Keying & ViT 4K Masking Pipeline")
    print(f"Input:      {input_path}")
    print(f"Screen:     {args.screen.upper()}")
    print(f"Resolution: {args.resolution.upper()}")
    print(f"Format:     {args.format.upper()}")
    print(f"Output:     {out_dir}")
    print(f"================================================================")

    # Read video
    info = VideoIO.get_video_info(input_path)
    limit = args.max_frames if args.max_frames > 0 else None
    print(f"Video Info: {info['width']}x{info['height']} @ {info['fps']:.2f}fps ({info['frame_count']} frames)")

    frames = VideoIO.read_frames(input_path, max_frames=limit)
    print(f"Loaded {len(frames)} frames.")

    # Initialize engines with pure-white core and 4K settings
    ibk = IBKEngine(screen_type=args.screen)
    vit = ViTEngine()
    fusion = MatteFusionEngine(
        FusionConfig(
            temporal_smoothing_alpha=0.2,
            black_clip=0.01,
            white_clip=0.99,
            use_core_fill=True,
        )
    )

    # Parse seed point
    seed_points = None
    if args.seed:
        parts = [float(p.strip()) for p in args.seed.split(",") if p.strip()]
        if len(parts) >= 2:
            seed_points = [(parts[0], parts[1])]

    print("Step 1: Tracking object with adaptive spatio-temporal tracker...")
    coarse_masks = vit.track_video_frames(frames, seed_points=seed_points)

    print("Step 2: Pulling IBK transmission mattes and fusing pure-white core...")
    fused_mattes = []
    is_4k = args.resolution.lower() == "4k"

    for idx, (frame, coarse_m) in enumerate(zip(frames, coarse_masks)):
        # IBK Keying
        ibk_res = ibk.execute_keying(frame)

        # ViT pure-white core
        vit_res = vit.extract_vit_matte(frame, coarse_mask=coarse_m)
        core_m = vit_res.core_mask.astype(float) / 255.0

        # Fusion with pure white core lock, pure black background, and 4K scaling
        final_m = fusion.process_frame(
            core_matte=core_m,
            edge_matte=ibk_res.alpha,
            rgb_guide=frame,
            scale_to_4k=is_4k,
        )
        fused_mattes.append(final_m)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(frames):
            print(f"  Processed {idx + 1}/{len(frames)} frames...")

    vit.release_memory()

    print(f"Step 3: Exporting image sequence ({args.format.upper()} @ {args.resolution.upper()})...")
    seq_dir = out_dir / args.format
    written = ImageSequenceIO.export_sequence(
        matte_sequence=fused_mattes,
        output_dir=seq_dir,
        prefix="alpha_matte_4k",
        format_type=args.format,
        rgb_plates=frames,
        resolution="native",  # Already scaled to 4K above
    )
    print(f"Exported {len(written)} files to: {seq_dir}")

    # Export preview videos
    alpha_video = out_dir / "alpha_matte_preview.mp4"
    red_video = out_dir / "red_overlay_preview.mp4"

    alpha_previews = []
    for m in fused_mattes:
        m_gray = (m * 255.0).astype("uint8")
        if is_4k:
            import cv2
            m_gray = cv2.resize(m_gray, (1920, 1080), interpolation=cv2.INTER_AREA)
        alpha_previews.append(np.stack([m_gray, m_gray, m_gray], axis=-1))

    VideoIO.write_video(alpha_previews, alpha_video, fps=info["fps"])
    VideoIO.create_red_overlay_video(frames, fused_mattes, red_video, fps=info["fps"])

    print(f"✨ Completed successfully!")
    print(f"  - 4K Alpha Sequence: {seq_dir}")
    print(f"  - Alpha Preview:     {alpha_video}")
    print(f"  - Red Overlay QA:    {red_video}")


if __name__ == "__main__":
    main()
