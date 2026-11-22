"""Centralized project paths and keypoint configuration."""

import os
from pathlib import Path
from typing import List

# Ensure Ultralytics and Matplotlib write settings/caches to /tmp to avoid macOS permission issues
if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mpl"


def _ensure_lap_shim() -> None:
    """Provide a SciPy-backed `lap.lapjv` compatibility module when `lap` C-extension is absent,
    enabling Ultralytics `model.track(..., persist=True)` ByteTrack/BoT-SORT offline.
    """
    import sys
    import types

    if "lap" in sys.modules:
        return
    try:
        import lap  # noqa: F401
        return
    except ImportError:
        pass

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def lapjv(cost_matrix: np.ndarray, extend_cost: bool = False, cost_limit: float = np.inf, return_cost: bool = True):
        cmat = np.asarray(cost_matrix, dtype=np.float64)
        n_rows, n_cols = cmat.shape
        if n_rows == 0 or n_cols == 0:
            x = np.full(n_rows, -1, dtype=np.int32)
            y = np.full(n_cols, -1, dtype=np.int32)
            return (0.0, x, y) if return_cost else (x, y)

        if np.isfinite(cost_limit):
            big = float(max(np.max(cmat[np.isfinite(cmat)], initial=1.0), abs(cost_limit)) * 1e6 + 1e6)
            work = np.where(np.isfinite(cmat), cmat, big)
            slack = np.full((n_rows, n_rows), big, dtype=np.float64)
            np.fill_diagonal(slack, float(cost_limit))
            aug = np.hstack([work, slack])
            r_ind, c_ind = linear_sum_assignment(aug)
            x = np.full(n_rows, -1, dtype=np.int32)
            y = np.full(n_cols, -1, dtype=np.int32)
            total_cost = 0.0
            for r, c in zip(r_ind, c_ind):
                if c < n_cols and cmat[r, c] <= cost_limit:
                    x[r] = int(c)
                    y[c] = int(r)
                    total_cost += float(cmat[r, c])
            return (total_cost, x, y) if return_cost else (x, y)

        if not extend_cost and n_rows != n_cols:
            raise ValueError("Non-square matrix requires extend_cost=True")

        r_ind, c_ind = linear_sum_assignment(cmat)
        x = np.full(n_rows, -1, dtype=np.int32)
        y = np.full(n_cols, -1, dtype=np.int32)
        total_cost = 0.0
        for r, c in zip(r_ind, c_ind):
            x[r] = int(c)
            y[c] = int(r)
            total_cost += float(cmat[r, c])
        return (total_cost, x, y) if return_cost else (x, y)

    shim = types.ModuleType("lap")
    shim.__version__ = "0.5.12-scipy-shim"
    shim.lapjv = lapjv  # type: ignore[attr-defined]
    sys.modules["lap"] = shim


_ensure_lap_shim()

# Base project root: employee-motion-tracker/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model weights directory
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_YOLO_MODEL = MODELS_DIR / "best_yolov8n_pose.pt"
BACKUP_V1_YOLO_MODEL = MODELS_DIR / "best_yolov8n_pose_v1_backup.pt"

# Input & fine-tuning data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FINE_TUNE_DATASET_DIR = DATA_DIR / "fine_tune_pose"
DEFAULT_INPUT_VIDEO = RAW_VIDEOS_DIR / "Input_1.mp4"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIDEOS_OUTPUT_DIR = OUTPUTS_DIR / "videos"
TELEMETRY_OUTPUT_DIR = OUTPUTS_DIR / "telemetry"
PLOTS_OUTPUT_DIR = OUTPUTS_DIR / "plots"
REPORTS_OUTPUT_DIR = OUTPUTS_DIR / "reports"

# Supported video extensions
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# YOLOv8-Pose COCO 17-keypoint indices (kpt_shape = [17, 3] -> x, y, confidence)
KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Target landmarks extracted for full posture, activity, secondary behavior, and ergonomic math
TARGET_LANDMARKS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

# Full head, arm, and torso skeleton connections for visualization
SKELETON_EDGES = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)


def ensure_directories() -> None:
    """Ensure all required data, model, and output directories exist."""
    for directory in (
        MODELS_DIR,
        RAW_VIDEOS_DIR,
        VIDEOS_OUTPUT_DIR,
        TELEMETRY_OUTPUT_DIR,
        PLOTS_OUTPUT_DIR,
        REPORTS_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def discover_raw_videos(directory: Path = RAW_VIDEOS_DIR) -> List[Path]:
    """Discover all supported video files in the specified directory."""
    if not directory.exists():
        return []
    videos = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS and not p.name.startswith(".")
    ]
    return sorted(videos)
