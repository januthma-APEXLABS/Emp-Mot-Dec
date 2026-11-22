"""Render the master H.264 tracked video from classified CSV telemetry (`Track_ID`)
and save dynamically sampled visual verification grids for any video duration.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config  # noqa: F401
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DEFAULT_INPUT_VIDEO,
    PLOTS_OUTPUT_DIR,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
)
from src.tracking.pipeline import render_video_from_telemetry_csv


def save_verification_grid(
    video_path: Path,
    out_plot_path: Path,
    frame_numbers: Optional[Sequence[int]] = None,
    suptitle: str = "Scale-Invariant Multi-Employee Tracking & 6-Class Categorization Verification",
) -> None:
    """Extract 6 representative frames dynamically across any video duration (or explicit frames if given)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open rendered video: {video_path}")

    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    if frame_numbers is None:
        quantiles = (0.0, 0.12, 0.32, 0.55, 0.78, 0.94)
        sampled = [max(1, min(total_frames, int(round(q * (total_frames - 1))) + 1)) for q in quantiles]
    else:
        sampled = [max(1, min(total_frames, int(f))) for f in frame_numbers]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2))
    axes_flat = axes.flatten()

    for ax, f_num in zip(axes_flat, sampled):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(f_num - 1, 0))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
        t_sec = (f_num - 1) / max(fps, 1.0)
        pct = (f_num / float(total_frames)) * 100.0
        ax.set_title(f"{video_path.name} • {t_sec:.1f}s ({pct:.0f}% of stream)", fontsize=10.5, fontweight="bold")
        ax.axis("off")

    cap.release()
    plt.suptitle(suptitle, fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved visual verification grid to: {out_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render master H.264 tracked video from classified CSV.")
    parser.add_argument("--video", type=str, default=str(DEFAULT_INPUT_VIDEO))
    parser.add_argument(
        "--csv",
        type=str,
        default=str(TELEMETRY_OUTPUT_DIR / "temporal_features.csv"),
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    vid_path = Path(args.video)
    out_vid = Path(args.output_video) if args.output_video else (VIDEOS_OUTPUT_DIR / f"{vid_path.stem}_tracked.mp4")

    res = render_video_from_telemetry_csv(
        video_path=vid_path,
        landmarks_csv_path=Path(args.csv),
        output_video_path=out_vid,
        smoother_name="Kalman",
    )
    print(
        f"Rendered H.264 video: {res['output_video_path']} "
        f"({res['frames_written']} frames in {res['elapsed_seconds']}s)"
    )

    save_verification_grid(
        video_path=out_vid,
        out_plot_path=PLOTS_OUTPUT_DIR / f"{vid_path.stem}_verification.png",
    )
    save_verification_grid(
        video_path=out_vid,
        out_plot_path=PLOTS_OUTPUT_DIR / "reid_6class_verification.png",
    )


if __name__ == "__main__":
    main()
