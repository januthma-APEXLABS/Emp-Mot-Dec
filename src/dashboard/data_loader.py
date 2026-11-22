"""Polars data ingestion, filtering, KPI math, Gantt segment compression, and heatmap binning
for Milestone 3: Executive Dashboard & Reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl

from src.config import TELEMETRY_OUTPUT_DIR

POSTURE_ORDER = ("Sitting", "Standing", "Walking", "Away")
ACTIVITY_ORDER = ("Working", "Idle", "Walking", "Away")


def format_duration_mmss(seconds: float) -> str:
    """Format duration in seconds into crisp human-readable 'Xm YYs' or 'YY.Ys'."""
    sec = max(float(seconds), 0.0)
    if sec < 60.0:
        return f"{sec:.1f}s"
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins}m {rem:04.1f}s"


def load_telemetry_dataframe(csv_path: Optional[Path] = None) -> pl.DataFrame:
    """Load finalized `temporal_features.csv` (or fallback `Input_1_classified.csv`)."""
    if csv_path is None:
        primary = TELEMETRY_OUTPUT_DIR / "temporal_features.csv"
        fallback = TELEMETRY_OUTPUT_DIR / "Input_1_classified.csv"
        target = primary if primary.exists() else fallback
    else:
        target = Path(csv_path)

    if not target.exists():
        raise FileNotFoundError(f"Telemetry CSV not found: {target}")

    df = pl.read_csv(target, infer_schema_length=25000)
    required = {"video_name", "frame", "timestamp_sec", "fps", "presence", "posture_state", "activity_state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Telemetry CSV missing required columns: {sorted(missing)}")

    return df.sort(["video_name", "frame", "track_id"])


def load_calibrated_thresholds(json_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load calibrated mathematical thresholds JSON if available."""
    target = Path(json_path) if json_path is not None else (TELEMETRY_OUTPUT_DIR / "calibrated_thresholds.json")
    if not target.exists():
        return {}
    with open(target, "r") as f:
        return json.load(f)


def get_available_videos(df: pl.DataFrame) -> List[str]:
    if df.is_empty():
        return []
    return sorted(df["video_name"].drop_nulls().unique().to_list())


def get_available_employees(df: pl.DataFrame, video_name: Optional[str] = None) -> List[int]:
    sub = df.filter(pl.col("video_name") == video_name) if video_name else df
    if sub.is_empty() or "track_id" not in sub.columns:
        return []
    ids = sub["track_id"].drop_nulls().unique().sort().to_list()
    return [int(x) for x in ids]


def filter_telemetry(
    df: pl.DataFrame,
    video_name: Optional[str] = None,
    employee_id: Optional[int] = None,
    time_range: Optional[Tuple[float, float]] = None,
) -> pl.DataFrame:
    """Filter telemetry by video, optional persistent `employee_id`, and timestamp window.

    When `employee_id` is specified, builds a complete 1-row-per-frame timeline for that
    specific employee so frames where that employee is out of the room are accurately
    represented as `Away`.
    """
    sub = df
    if video_name is not None:
        sub = sub.filter(pl.col("video_name") == video_name)

    if time_range is not None:
        t_lo, t_hi = float(time_range[0]), float(time_range[1])
        sub = sub.filter((pl.col("timestamp_sec") >= t_lo - 1e-6) & (pl.col("timestamp_sec") <= t_hi + 1e-6))

    if sub.is_empty() or employee_id is None:
        return sub

    emp_int = int(employee_id)
    # Build canonical frame spine per video so non-present frames for `emp_int` become 'Away'
    frame_spine = (
        sub.select(["video_name", "frame", "timestamp_sec", "fps", "frame_width", "frame_height"])
        .unique(subset=["video_name", "frame"], keep="first")
        .sort(["video_name", "frame"])
    )
    emp_rows = sub.filter(pl.col("track_id") == emp_int).drop(
        ["timestamp_sec", "fps", "frame_width", "frame_height"]
    )

    joined = frame_spine.join(emp_rows, on=["video_name", "frame"], how="left").with_columns(
        [
            pl.col("presence").fill_null(0),
            pl.col("presence_state").fill_null("Away"),
            pl.col("posture_state").fill_null("Away"),
            pl.col("activity_state").fill_null("Away"),
            pl.col("motion_state").fill_null("Away"),
            pl.col("tracking_status").fill_null("away"),
            pl.col("camera_zone").fill_null("-"),
        ]
    )
    return joined.sort(["video_name", "frame"])


def compute_dashboard_kpis(
    df: pl.DataFrame,
    sparkline_points: int = 36,
) -> Dict[str, Any]:
    """Compute executive KPIs, posture & utilization percentages, and sparkline series."""
    if df.is_empty():
        empty_b = {k: {"count": 0, "seconds": 0.0, "pct": 0.0} for k in POSTURE_ORDER}
        empty_a = {k: {"count": 0, "seconds": 0.0, "pct": 0.0} for k in ACTIVITY_ORDER}
        return {
            "total_rows": 0,
            "unique_frames": 0,
            "fps": 30.0,
            "video_duration_sec": 0.0,
            "observed_person_sec": 0.0,
            "unique_employees": 0,
            "employee_ids": [],
            "posture_breakdown": empty_b,
            "activity_breakdown": empty_a,
            "utilization_3way": {
                "Working": {"count": 0, "seconds": 0.0, "pct": 0.0},
                "Idle": {"count": 0, "seconds": 0.0, "pct": 0.0},
                "Away": {"count": 0, "seconds": 0.0, "pct": 0.0},
            },
            "sitting_pct": 0.0,
            "standing_pct": 0.0,
            "walking_pct": 0.0,
            "away_pct": 0.0,
            "working_pct": 0.0,
            "working_sec": 0.0,
            "idle_pct": 0.0,
            "idle_sec": 0.0,
            "mean_wrist_vel_px_s": 0.0,
            "peak_wrist_vel_px_s": 0.0,
            "posture_transitions": 0,
            "sparkline_x": [0.0, 1.0],
            "working_sparkline": [0.0, 0.0],
            "idle_sparkline": [0.0, 0.0],
        }

    fps = float(df["fps"].drop_nulls().median() or 30.0)
    dt = 1.0 / max(fps, 1.0)
    n_rows = len(df)
    uniq_frames = int(df.select(["video_name", "frame"]).unique().height)
    video_duration_sec = round(uniq_frames * dt, 2)
    observed_person_sec = round(n_rows * dt, 2)

    emp_ids = get_available_employees(df)

    # 1. Posture breakdown (Sitting, Standing, Walking, Away)
    p_counts_raw = dict(df.group_by("posture_state").len().iter_rows())
    posture_breakdown: Dict[str, Dict[str, float]] = {}
    for cls in POSTURE_ORDER:
        cnt = int(p_counts_raw.get(cls, 0))
        posture_breakdown[cls] = {
            "count": cnt,
            "seconds": round(cnt * dt, 2),
            "pct": round((cnt / n_rows) * 100.0, 1),
        }

    # 2. Activity breakdown (Working, Idle, Walking, Away)
    a_counts_raw = dict(df.group_by("activity_state").len().iter_rows())
    activity_breakdown: Dict[str, Dict[str, float]] = {}
    for cls in ACTIVITY_ORDER:
        cnt = int(a_counts_raw.get(cls, 0))
        activity_breakdown[cls] = {
            "count": cnt,
            "seconds": round(cnt * dt, 2),
            "pct": round((cnt / n_rows) * 100.0, 1),
        }

    # 3. 3-Way Utilization view (Active Working/Transit vs Idle vs Away, plus strict Working)
    work_cnt = int(a_counts_raw.get("Working", 0))
    walk_cnt = int(a_counts_raw.get("Walking", 0))
    idle_cnt = int(a_counts_raw.get("Idle", 0))
    away_cnt = int(a_counts_raw.get("Away", 0))
    denom_3way = max(work_cnt + idle_cnt + away_cnt, 1)
    utilization_3way = {
        "Working": {
            "count": work_cnt,
            "seconds": round(work_cnt * dt, 2),
            "pct": round((work_cnt / n_rows) * 100.0, 1),
            "share_ex_walk_pct": round((work_cnt / denom_3way) * 100.0, 1),
        },
        "Idle": {
            "count": idle_cnt,
            "seconds": round(idle_cnt * dt, 2),
            "pct": round((idle_cnt / n_rows) * 100.0, 1),
            "share_ex_walk_pct": round((idle_cnt / denom_3way) * 100.0, 1),
        },
        "Away": {
            "count": away_cnt,
            "seconds": round(away_cnt * dt, 2),
            "pct": round((away_cnt / n_rows) * 100.0, 1),
            "share_ex_walk_pct": round((away_cnt / denom_3way) * 100.0, 1),
        },
    }
    _ = walk_cnt

    # 4. Wrist velocity stats on present rows
    if "rolling_30f_wrist_vel_px_s" in df.columns:
        vel_series = df.filter(pl.col("presence") == 1)["rolling_30f_wrist_vel_px_s"].drop_nulls()
        vel_np = vel_series.to_numpy()
        vel_np = vel_np[np.isfinite(vel_np)]
    else:
        vel_np = np.array([], dtype=np.float64)

    mean_vel = round(float(np.mean(vel_np)), 2) if len(vel_np) > 0 else 0.0
    peak_vel = round(float(np.percentile(vel_np, 99.0)), 2) if len(vel_np) > 0 else 0.0

    # 5. Count posture transitions per employee
    transitions = 0
    present_tracks = df.filter(pl.col("track_id").is_not_null()).sort(["video_name", "track_id", "frame"])
    for _, grp in present_tracks.group_by(["video_name", "track_id"], maintain_order=True):
        states = grp["posture_state"].to_list()
        transitions += sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])

    # 6. Smooth cumulative running-share sparklines across timeline bins
    t_arr = df["timestamp_sec"].to_numpy()
    act_arr = df["activity_state"].to_numpy()
    t_min, t_max = float(np.min(t_arr)), float(np.max(t_arr))
    num_pts = max(int(sparkline_points), 6)

    if t_max - t_min <= 1e-6:
        spark_x = [0.0, 1.0]
        work_spark = [activity_breakdown["Working"]["pct"]] * 2
        idle_spark = [activity_breakdown["Idle"]["pct"]] * 2
    else:
        edges = np.linspace(t_min, t_max, num_pts + 1)
        spark_x = [round(float(v), 2) for v in edges[1:]]
        work_spark = []
        idle_spark = []
        is_work = (act_arr == "Working").astype(np.float64)
        is_idle = (act_arr == "Idle").astype(np.float64)
        for b_idx in range(1, len(edges)):
            mask = t_arr <= edges[b_idx] + 1e-6
            cnt_so_far = max(int(np.sum(mask)), 1)
            work_spark.append(round(float(np.sum(is_work[mask]) / cnt_so_far) * 100.0, 2))
            idle_spark.append(round(float(np.sum(is_idle[mask]) / cnt_so_far) * 100.0, 2))

    return {
        "total_rows": n_rows,
        "unique_frames": uniq_frames,
        "fps": fps,
        "video_duration_sec": video_duration_sec,
        "observed_person_sec": observed_person_sec,
        "unique_employees": len(emp_ids),
        "employee_ids": emp_ids,
        "posture_breakdown": posture_breakdown,
        "activity_breakdown": activity_breakdown,
        "utilization_3way": utilization_3way,
        "sitting_pct": posture_breakdown["Sitting"]["pct"],
        "standing_pct": posture_breakdown["Standing"]["pct"],
        "walking_pct": posture_breakdown["Walking"]["pct"],
        "away_pct": posture_breakdown["Away"]["pct"],
        "working_pct": activity_breakdown["Working"]["pct"],
        "working_sec": activity_breakdown["Working"]["seconds"],
        "idle_pct": activity_breakdown["Idle"]["pct"],
        "idle_sec": activity_breakdown["Idle"]["seconds"],
        "mean_wrist_vel_px_s": mean_vel,
        "peak_wrist_vel_px_s": peak_vel,
        "posture_transitions": transitions,
        "sparkline_x": spark_x,
        "working_sparkline": work_spark,
        "idle_sparkline": idle_spark,
    }


def extract_state_segments(
    df: pl.DataFrame,
    state_col: str = "posture_state",
) -> List[Dict[str, Any]]:
    """Compress frame-by-frame labels into contiguous [start_sec, end_sec] Gantt intervals."""
    if df.is_empty() or state_col not in df.columns:
        return []

    fps = float(df["fps"].drop_nulls().median() or 30.0)
    dt = 1.0 / max(fps, 1.0)

    segments: List[Dict[str, Any]] = []

    # Group by employee track if present; also include room-level Away intervals
    emp_ids = get_available_employees(df)
    if not emp_ids:
        emp_ids = []

    for emp_id in emp_ids:
        emp_timeline = filter_telemetry(df, employee_id=emp_id)
        vids = emp_timeline["video_name"].to_list()
        frames = emp_timeline["frame"].to_numpy()
        times = emp_timeline["timestamp_sec"].to_numpy()
        states = emp_timeline[state_col].to_list()
        motions = emp_timeline["motion_state"].to_list() if "motion_state" in emp_timeline.columns else states
        n = len(states)
        if n == 0:
            continue

        s_idx = 0
        for i in range(1, n + 1):
            is_gap = i < n and (vids[i] != vids[i - 1] or abs(int(frames[i]) - int(frames[i - 1])) > 2)
            if i == n or states[i] != states[s_idx] or is_gap:
                t_start = float(times[s_idx])
                t_end = float(times[i - 1]) + dt
                dur = max(t_end - t_start, dt)
                segments.append(
                    {
                        "track_label": f"Emp #{emp_id}",
                        "track_id": emp_id,
                        "state": states[s_idx],
                        "motion_detail": motions[s_idx],
                        "start_sec": round(t_start, 3),
                        "end_sec": round(t_end, 3),
                        "duration_sec": round(dur, 2),
                        "start_frame": int(frames[s_idx]),
                        "end_frame": int(frames[i - 1]),
                    }
                )
                s_idx = i

    if not segments:
        # Only Away rows exist
        times = df["timestamp_sec"].to_numpy()
        frames = df["frame"].to_numpy()
        t_start = float(times[0])
        t_end = float(times[-1]) + dt
        segments.append(
            {
                "track_label": "Workstation",
                "track_id": 0,
                "state": "Away",
                "motion_detail": "Away",
                "start_sec": round(t_start, 3),
                "end_sec": round(t_end, 3),
                "duration_sec": round(max(t_end - t_start, dt), 2),
                "start_frame": int(frames[0]),
                "end_frame": int(frames[-1]),
            }
        )

    return segments


def build_velocity_heatmap_data(
    df: pl.DataFrame,
    bin_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Aggregate rolling wrist velocity into a 2D [channel_or_employee x time_bin] heatmap matrix."""
    if df.is_empty():
        return {"x_centers": [0.0], "y_labels": ["Dominant Wrist"], "z_matrix": [[0.0]], "hover_text": [[""]]}

    t_all = df["timestamp_sec"].to_numpy()
    t_min = float(np.min(t_all))
    t_max = float(np.max(t_all))
    bin_w = max(float(bin_seconds), 0.5)
    num_bins = max(int(np.ceil((t_max - t_min) / bin_w)), 1)
    edges = np.linspace(t_min, max(t_max, t_min + bin_w), num_bins + 1)
    centers = [round(0.5 * float(edges[i] + edges[i + 1]), 1) for i in range(num_bins)]

    emp_ids = get_available_employees(df)
    y_labels: List[str] = []
    z_matrix: List[List[float]] = []
    hover_matrix: List[List[str]] = []

    if len(emp_ids) <= 1:
        # Single-employee detailed view: Dominant Wrist, Right Wrist, Left Wrist
        channel_specs = [
            ("Dominant Wrist (30f)", "rolling_30f_wrist_vel_px_s"),
            ("Right Wrist (30f)", "rolling_30f_right_wrist_vel_px_s"),
            ("Left Wrist (30f)", "rolling_30f_left_wrist_vel_px_s"),
        ]
        present = df.filter(pl.col("presence") == 1)
        p_t = present["timestamp_sec"].to_numpy() if not present.is_empty() else np.array([])
        for label, col_name in channel_specs:
            if col_name not in df.columns:
                continue
            vals = np.nan_to_num(present[col_name].to_numpy(), nan=0.0) if not present.is_empty() else np.array([])
            row_z: List[float] = []
            row_h: List[str] = []
            for b in range(num_bins):
                lo, hi = edges[b], edges[b + 1]
                mask = (p_t >= lo) & (p_t < hi if b < num_bins - 1 else p_t <= hi + 1e-6)
                v = float(np.mean(vals[mask])) if np.any(mask) else 0.0
                v_max = float(np.max(vals[mask])) if np.any(mask) else 0.0
                row_z.append(round(v, 2))
                row_h.append(
                    f"<b>{label}</b><br>Window: {lo:.1f}s – {hi:.1f}s<br>"
                    f"Mean Speed: {v:.1f} px/s<br>Peak Speed: {v_max:.1f} px/s"
                )
            y_labels.append(label)
            z_matrix.append(row_z)
            hover_matrix.append(row_h)
    else:
        # Multi-employee + channel matrix: each employee's Dominant, Right, and Left wrist channels
        for emp_id in reversed(emp_ids):
            emp_df = df.filter((pl.col("track_id") == emp_id) & (pl.col("presence") == 1))
            p_t = emp_df["timestamp_sec"].to_numpy() if not emp_df.is_empty() else np.array([])
            for ch_short, col_name in [
                ("Right Wrist", "rolling_30f_right_wrist_vel_px_s"),
                ("Left Wrist", "rolling_30f_left_wrist_vel_px_s"),
                ("Peak Wrist", "rolling_30f_wrist_vel_px_s"),
            ]:
                if col_name not in df.columns:
                    continue
                label = f"Emp #{emp_id} • {ch_short}"
                vals = np.nan_to_num(emp_df[col_name].to_numpy(), nan=0.0) if not emp_df.is_empty() else np.array([])
                row_z = []
                row_h = []
                for b in range(num_bins):
                    lo, hi = edges[b], edges[b + 1]
                    mask = (p_t >= lo) & (p_t < hi if b < num_bins - 1 else p_t <= hi + 1e-6)
                    v = float(np.mean(vals[mask])) if np.any(mask) else 0.0
                    v_max = float(np.max(vals[mask])) if np.any(mask) else 0.0
                    row_z.append(round(v, 2))
                    row_h.append(
                        f"<b>{label}</b><br>Window: {lo:.1f}s – {hi:.1f}s<br>"
                        f"Mean Speed: {v:.1f} px/s<br>Peak Speed: {v_max:.1f} px/s"
                    )
                y_labels.append(label)
                z_matrix.append(row_z)
                hover_matrix.append(row_h)

    return {
        "x_centers": centers,
        "bin_seconds": bin_w,
        "y_labels": y_labels,
        "z_matrix": z_matrix,
        "hover_text": hover_matrix,
    }


def build_employee_summary_table(df: pl.DataFrame) -> pl.DataFrame:
    """Build an executive per-employee + overall KPI comparison table ready for UI display & CSV export."""
    rows: List[Dict[str, Any]] = []

    overall = compute_dashboard_kpis(df)
    rows.append(
        {
            "Scope": "All Observations (Aggregate)",
            "Active_Seconds": overall["observed_person_sec"],
            "Duration_Formatted": format_duration_mmss(overall["observed_person_sec"]),
            "Working_Pct": overall["working_pct"],
            "Idle_Pct": overall["idle_pct"],
            "Sitting_Pct": overall["sitting_pct"],
            "Standing_Pct": overall["standing_pct"],
            "Walking_Pct": overall["walking_pct"],
            "Away_Pct": overall["away_pct"],
            "Mean_Wrist_Speed_px_s": overall["mean_wrist_vel_px_s"],
            "Peak_Wrist_Speed_px_s": overall["peak_wrist_vel_px_s"],
            "Posture_Transitions": overall["posture_transitions"],
        }
    )

    for emp_id in get_available_employees(df):
        emp_full = filter_telemetry(df, employee_id=emp_id)
        emp_present_only = df.filter(pl.col("track_id") == emp_id)
        k_full = compute_dashboard_kpis(emp_full)
        k_pres = compute_dashboard_kpis(emp_present_only)
        rows.append(
            {
                "Scope": f"Emp #{emp_id}",
                "Active_Seconds": k_pres["observed_person_sec"],
                "Duration_Formatted": format_duration_mmss(k_pres["observed_person_sec"]),
                "Working_Pct": k_full["working_pct"],
                "Idle_Pct": k_full["idle_pct"],
                "Sitting_Pct": k_full["sitting_pct"],
                "Standing_Pct": k_full["standing_pct"],
                "Walking_Pct": k_full["walking_pct"],
                "Away_Pct": k_full["away_pct"],
                "Mean_Wrist_Speed_px_s": k_pres["mean_wrist_vel_px_s"],
                "Peak_Wrist_Speed_px_s": k_pres["peak_wrist_vel_px_s"],
                "Posture_Transitions": k_pres["posture_transitions"],
            }
        )

    return pl.DataFrame(rows)
