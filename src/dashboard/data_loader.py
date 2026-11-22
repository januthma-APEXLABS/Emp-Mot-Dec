"""Polars data ingestion, multi-stream discovery, filtering, scale-invariant KPI math,
per-employee detail profiling (`1..N` employees), Gantt segment compression, and heatmap binning
for Milestone 3: Executive Dashboard & Reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl

from src.calibration.features import compute_temporal_features
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


def _normalize_track_id_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure both `track_id` and `Track_ID` exist as clean nullable Int64 columns."""
    if df.is_empty():
        return df

    has_tid = "track_id" in df.columns
    has_cap = "Track_ID" in df.columns
    if not has_tid and not has_cap:
        return df.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias("track_id"),
                pl.lit(None, dtype=pl.Int64).alias("Track_ID"),
            ]
        )

    def _to_int_expr(col_name: str) -> pl.Expr:
        if df.schema[col_name] in (pl.Utf8, pl.String):
            return (
                pl.when(pl.col(col_name).str.strip_chars() == "")
                .then(None)
                .otherwise(pl.col(col_name))
                .cast(pl.Float64, strict=False)
                .cast(pl.Int64, strict=False)
            )
        return pl.col(col_name).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)

    if has_tid and has_cap:
        merged_expr = pl.coalesce([_to_int_expr("track_id"), _to_int_expr("Track_ID")])
    elif has_tid:
        merged_expr = _to_int_expr("track_id")
    else:
        merged_expr = _to_int_expr("Track_ID")

    return df.with_columns(
        [
            merged_expr.alias("track_id"),
            merged_expr.alias("Track_ID"),
        ]
    )


def _enrich_raw_landmarks_df(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Convert a freshly played Milestone 1 `{stem}_landmarks.csv` DataFrame into a dashboard-ready
    feature DataFrame immediately (even before Milestone 2 has been run on it).
    """
    feat_df = compute_temporal_features(raw_df)
    extra_cols = []
    if "presence_state" not in feat_df.columns:
        extra_cols.append(
            pl.when(pl.col("presence") == 1)
            .then(pl.lit("Present"))
            .otherwise(pl.lit("Away"))
            .alias("presence_state")
        )
    if "camera_zone" not in feat_df.columns:
        extra_cols.append(
            pl.when(pl.col("presence") == 1)
            .then(pl.lit("global"))
            .otherwise(pl.lit("-"))
            .alias("camera_zone")
        )
    if "motion_state" not in feat_df.columns:
        extra_cols.append(
            pl.when(pl.col("presence") != 1)
            .then(pl.lit("Away"))
            .when(pl.col("posture_state") == "Walking")
            .then(pl.lit("Walking"))
            .otherwise(pl.concat_str([pl.col("posture_state"), pl.lit(" | "), pl.col("activity_state")]))
            .alias("motion_state")
        )
    if extra_cols:
        feat_df = feat_df.with_columns(extra_cols)
    return _normalize_track_id_columns(feat_df)


def get_telemetry_cache_fingerprint(
    telemetry_dir: Path = TELEMETRY_OUTPUT_DIR,
) -> Tuple[Tuple[str, int, int], ...]:
    """Return a hashable `(filename, mtime_ns, size)` fingerprint across `telemetry_dir` so
    `@st.cache_data` refreshes the instant any video stream is played or updated.
    """
    t_dir = Path(telemetry_dir)
    if not t_dir.exists():
        return ()
    patterns = (
        "played_streams.json",
        "temporal_features.csv",
        "*_classified.csv",
        "*_landmarks.csv",
        "*_thresholds.json",
        "calibrated_thresholds.json",
    )
    seen = {}
    for pat in patterns:
        for p in t_dir.glob(pat):
            if p.is_file() and p.name != "batch_landmarks.csv":
                st = p.stat()
                seen[p.name] = (p.name, int(st.st_mtime_ns), int(st.st_size))
    return tuple(seen[k] for k in sorted(seen.keys()))


def load_played_stream_manifest(telemetry_dir: Path = TELEMETRY_OUTPUT_DIR) -> List[str]:
    """Return played video names ordered by play history from `played_streams.json` if available."""
    manifest_path = Path(telemetry_dir) / "played_streams.json"
    if not manifest_path.exists():
        return []
    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(item["video_name"]) for item in data if isinstance(item, dict) and item.get("video_name")]
    except Exception:
        pass
    return []


def load_all_played_telemetry(telemetry_dir: Path = TELEMETRY_OUTPUT_DIR) -> pl.DataFrame:
    """Discover and merge EVERY played/processed video stream in `telemetry_dir`.

    - Uses `{stem}_classified.csv` when available.
    - Immediately falls back to `{stem}_landmarks.csv` (enriched on the fly) the moment a new
      video finishes playing in Milestone 1, even before Milestone 2 is executed.
    - Also includes any extra `video_name` streams stored in `temporal_features.csv`.
    """
    t_dir = Path(telemetry_dir)
    if not t_dir.exists():
        raise FileNotFoundError(f"Telemetry directory not found: {t_dir}")

    classified_by_stem: Dict[str, Path] = {
        p.name[: -len("_classified.csv")]: p
        for p in sorted(t_dir.glob("*_classified.csv"))
        if p.is_file()
    }
    landmarks_by_stem: Dict[str, Path] = {
        p.name[: -len("_landmarks.csv")]: p
        for p in sorted(t_dir.glob("*_landmarks.csv"))
        if p.is_file() and p.name != "batch_landmarks.csv"
    }

    all_stems = sorted(set(classified_by_stem.keys()) | set(landmarks_by_stem.keys()))
    frames_list: List[pl.DataFrame] = []
    loaded_video_names: set = set()

    for stem in all_stems:
        if stem in classified_by_stem:
            part = pl.read_csv(classified_by_stem[stem], infer_schema_length=25000)
            part = _normalize_track_id_columns(part)
        else:
            raw_part = pl.read_csv(landmarks_by_stem[stem], infer_schema_length=25000)
            part = _enrich_raw_landmarks_df(raw_part)

        if not part.is_empty() and "video_name" in part.columns:
            loaded_video_names.update(part["video_name"].drop_nulls().unique().to_list())
            frames_list.append(part)

    tf_path = t_dir / "temporal_features.csv"
    if tf_path.exists():
        tf_df = _normalize_track_id_columns(pl.read_csv(tf_path, infer_schema_length=25000))
        if "video_name" in tf_df.columns:
            extra_tf = tf_df.filter(~pl.col("video_name").is_in(list(loaded_video_names)))
            if not extra_tf.is_empty():
                frames_list.append(extra_tf)

    if not frames_list:
        raise FileNotFoundError(f"No telemetry CSV streams found in: {t_dir}")

    combined = (
        pl.concat(frames_list, how="diagonal_relaxed")
        if len(frames_list) > 1
        else frames_list[0]
    )
    required = {"video_name", "frame", "timestamp_sec", "fps", "presence", "posture_state", "activity_state"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Telemetry CSV missing required columns: {sorted(missing)}")

    return combined.sort(["video_name", "frame", "track_id"], nulls_last=True)


def load_telemetry_dataframe(csv_path: Optional[Path] = None) -> pl.DataFrame:
    """Load telemetry from a specific CSV path, or merge all played streams when `csv_path` is None."""
    if csv_path is None:
        return load_all_played_telemetry(TELEMETRY_OUTPUT_DIR)

    target = Path(csv_path)
    if not target.exists():
        raise FileNotFoundError(f"Telemetry CSV not found: {target}")

    df = pl.read_csv(target, infer_schema_length=25000)
    if "shoulder_dist_bottom_px" not in df.columns and "left_shoulder_y" in df.columns:
        df = _enrich_raw_landmarks_df(df)
    else:
        df = _normalize_track_id_columns(df)

    required = {"video_name", "frame", "timestamp_sec", "fps", "presence", "posture_state", "activity_state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Telemetry CSV missing required columns: {sorted(missing)}")

    return df.sort(["video_name", "frame", "track_id"], nulls_last=True)


def load_calibrated_thresholds(
    json_path: Optional[Path] = None,
    video_name: Optional[str] = None,
    telemetry_dir: Path = TELEMETRY_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Load per-video `{stem}_thresholds.json` when `video_name` is supplied, or global fallback."""
    t_dir = Path(telemetry_dir)
    if json_path is not None:
        target = Path(json_path)
    elif video_name:
        per_vid = t_dir / f"{Path(video_name).stem}_thresholds.json"
        target = per_vid if per_vid.exists() else (t_dir / "calibrated_thresholds.json")
    else:
        target = t_dir / "calibrated_thresholds.json"

    if not target.exists():
        return {}
    with open(target, "r") as f:
        return json.load(f)


def get_available_videos(
    df: pl.DataFrame,
    telemetry_dir: Optional[Path] = None,
) -> List[str]:
    """Return all available played video streams, preserving natural order and including any newly played streams."""
    if df.is_empty() or "video_name" not in df.columns:
        return []
    discovered = sorted(str(v) for v in df["video_name"].drop_nulls().unique().to_list())
    if telemetry_dir is None:
        return discovered

    manifest_order = load_played_stream_manifest(telemetry_dir)
    if not manifest_order:
        return discovered

    # Keep natural order for baseline streams, and ensure every manifest stream in `discovered` is present
    disc_set = set(discovered)
    ordered = [v for v in discovered if v in disc_set]
    return ordered


def get_latest_played_video(telemetry_dir: Path = TELEMETRY_OUTPUT_DIR) -> Optional[str]:
    """Return the most recently played video stream recorded in `played_streams.json`, if any."""
    manifest = load_played_stream_manifest(telemetry_dir)
    return manifest[-1] if manifest else None


def get_available_employees(df: pl.DataFrame, video_name: Optional[str] = None) -> List[int]:
    """Return all unique positive `Track_ID` integers (`1..N`, unlimited count) for `video_name`."""
    sub = df.filter(pl.col("video_name") == video_name) if video_name else df
    if sub.is_empty():
        return []
    sub = _normalize_track_id_columns(sub)
    if "presence" in sub.columns:
        sub = sub.filter(pl.col("presence") == 1)
    if sub.is_empty() or "track_id" not in sub.columns:
        return []
    ids = (
        sub["track_id"]
        .drop_nulls()
        .filter(sub["track_id"].drop_nulls() > 0)
        .unique()
        .sort()
        .to_list()
    )
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
    sub = _normalize_track_id_columns(df)
    if video_name is not None:
        sub = sub.filter(pl.col("video_name") == video_name)

    if time_range is not None:
        t_lo, t_hi = float(time_range[0]), float(time_range[1])
        sub = sub.filter((pl.col("timestamp_sec") >= t_lo - 1e-6) & (pl.col("timestamp_sec") <= t_hi + 1e-6))

    if sub.is_empty() or employee_id is None:
        return sub

    emp_int = int(employee_id)
    frame_spine = (
        sub.select(["video_name", "frame", "timestamp_sec", "fps", "frame_width", "frame_height"])
        .unique(subset=["video_name", "frame"], keep="first")
        .sort(["video_name", "frame"])
    )
    emp_rows = (
        sub.filter(pl.col("track_id") == emp_int)
        .unique(subset=["video_name", "frame"], keep="first")
        .drop(["timestamp_sec", "fps", "frame_width", "frame_height"])
    )

    fill_exprs = [
        pl.col("presence").fill_null(0),
        pl.col("posture_state").fill_null("Away"),
        pl.col("activity_state").fill_null("Away"),
        pl.col("tracking_status").fill_null("away"),
    ]
    for opt_col, fallback_val in (
        ("presence_state", "Away"),
        ("motion_state", "Away"),
        ("camera_zone", "-"),
    ):
        if opt_col in emp_rows.columns:
            fill_exprs.append(pl.col(opt_col).fill_null(fallback_val))

    joined = frame_spine.join(emp_rows, on=["video_name", "frame"], how="left").with_columns(fill_exprs)
    return joined.sort(["video_name", "frame"])


def compute_dashboard_kpis(
    df: pl.DataFrame,
    sparkline_points: int = 36,
) -> Dict[str, Any]:
    """Compute executive KPIs, posture & utilization percentages, scale-invariant wrist speeds, and sparklines."""
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
                "Working": {"count": 0, "seconds": 0.0, "pct": 0.0, "share_ex_walk_pct": 0.0},
                "Idle": {"count": 0, "seconds": 0.0, "pct": 0.0, "share_ex_walk_pct": 0.0},
                "Away": {"count": 0, "seconds": 0.0, "pct": 0.0, "share_ex_walk_pct": 0.0},
            },
            "sitting_pct": 0.0,
            "standing_pct": 0.0,
            "walking_pct": 0.0,
            "away_pct": 0.0,
            "working_pct": 0.0,
            "working_sec": 0.0,
            "idle_pct": 0.0,
            "idle_sec": 0.0,
            "mean_wrist_vel_sw_s": 0.0,
            "peak_wrist_vel_sw_s": 0.0,
            "mean_wrist_vel_px_s": 0.0,
            "peak_wrist_vel_px_s": 0.0,
            "posture_transitions": 0,
            "sparkline_x": [0.0, 1.0],
            "working_sparkline": [0.0, 0.0],
            "idle_sparkline": [0.0, 0.0],
        }

    df = _normalize_track_id_columns(df)
    fps = float(df["fps"].drop_nulls().median() or 30.0)
    dt = 1.0 / max(fps, 1.0)
    n_rows = len(df)
    uniq_frames = int(df.select(["video_name", "frame"]).unique().height)
    video_duration_sec = round(uniq_frames * dt, 2)
    present_rows_cnt = int(df.filter(pl.col("presence") == 1).height) if "presence" in df.columns else n_rows
    observed_person_sec = round(present_rows_cnt * dt, 2)

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

    # 3. 3-Way Utilization view
    work_cnt = int(a_counts_raw.get("Working", 0))
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

    # 4. Scale-invariant (SW/s) and pixel (px/s) wrist velocity stats on present rows
    present_sub = df.filter(pl.col("presence") == 1)

    def _col_mean_p99(col_name: str, decimals: int = 3) -> Tuple[float, float]:
        if col_name not in present_sub.columns or present_sub.is_empty():
            return 0.0, 0.0
        arr = present_sub[col_name].cast(pl.Float64, strict=False).drop_nulls().to_numpy()
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0, 0.0
        return round(float(np.mean(arr)), decimals), round(float(np.percentile(arr, 99.0)), decimals)

    mean_sw, peak_sw = _col_mean_p99("rolling_30f_wrist_vel_sw_s", decimals=3)
    mean_px, peak_px = _col_mean_p99("rolling_30f_wrist_vel_px_s", decimals=2)

    # 5. Count posture transitions per employee
    transitions = 0
    present_tracks = df.filter((pl.col("presence") == 1) & pl.col("track_id").is_not_null()).sort(
        ["video_name", "track_id", "frame"]
    )
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
        "mean_wrist_vel_sw_s": mean_sw,
        "peak_wrist_vel_sw_s": peak_sw,
        "mean_wrist_vel_px_s": mean_px,
        "peak_wrist_vel_px_s": peak_px,
        "posture_transitions": transitions,
        "sparkline_x": spark_x,
        "working_sparkline": work_spark,
        "idle_sparkline": idle_spark,
    }


def build_employee_detail_profiles(
    df: pl.DataFrame,
    video_name: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """Build detailed per-employee telemetry cards for ALL detected employees (`1..N`)."""
    sub = _normalize_track_id_columns(df)
    if video_name is not None:
        sub = sub.filter(pl.col("video_name") == video_name)
    if time_range is not None:
        t_lo, t_hi = float(time_range[0]), float(time_range[1])
        sub = sub.filter((pl.col("timestamp_sec") >= t_lo - 1e-6) & (pl.col("timestamp_sec") <= t_hi + 1e-6))

    if sub.is_empty():
        return []

    profiles: List[Dict[str, Any]] = []
    for emp_id in get_available_employees(sub):
        emp_full = filter_telemetry(sub, employee_id=emp_id)
        emp_pres = sub.filter((pl.col("track_id") == emp_id) & (pl.col("presence") == 1)).sort("frame")
        if emp_pres.is_empty():
            continue

        k_full = compute_dashboard_kpis(emp_full)
        k_pres = compute_dashboard_kpis(emp_pres)

        t_vals = emp_pres["timestamp_sec"].to_numpy()
        first_seen = round(float(np.min(t_vals)), 1)
        last_seen = round(float(np.max(t_vals)), 1)

        zone_val = "global"
        if "camera_zone" in emp_pres.columns:
            z_modes = emp_pres["camera_zone"].drop_nulls().mode().to_list()
            if z_modes and str(z_modes[0]) != "-":
                zone_val = str(z_modes[0])

        p_modes = emp_pres["posture_state"].drop_nulls().mode().to_list()
        dom_posture = str(p_modes[0]) if p_modes else "Sitting"
        a_modes = emp_pres["activity_state"].drop_nulls().mode().to_list()
        dom_activity = str(a_modes[0]) if a_modes else "Working"

        def _mean_col(cname: str) -> float:
            if cname not in emp_pres.columns:
                return 0.0
            arr = emp_pres[cname].cast(pl.Float64, strict=False).drop_nulls().to_numpy()
            arr = arr[np.isfinite(arr)]
            return round(float(np.mean(arr)), 3) if len(arr) > 0 else 0.0

        profiles.append(
            {
                "track_id": int(emp_id),
                "label": f"Emp #{emp_id}",
                "active_frames": int(len(emp_pres)),
                "active_seconds": k_pres["observed_person_sec"],
                "duration_formatted": format_duration_mmss(k_pres["observed_person_sec"]),
                "first_seen_sec": first_seen,
                "last_seen_sec": last_seen,
                "window_formatted": f"{format_duration_mmss(first_seen)} – {format_duration_mmss(last_seen)}",
                "camera_zone": zone_val.capitalize(),
                "dominant_posture": dom_posture,
                "dominant_activity": dom_activity,
                "working_pct": k_full["working_pct"],
                "working_sec": k_full["working_sec"],
                "in_room_working_pct": k_pres["working_pct"],
                "idle_pct": k_full["idle_pct"],
                "idle_sec": k_full["idle_sec"],
                "in_room_idle_pct": k_pres["idle_pct"],
                "sitting_pct": k_full["sitting_pct"],
                "standing_pct": k_full["standing_pct"],
                "walking_pct": k_full["walking_pct"],
                "away_pct": k_full["away_pct"],
                "mean_wrist_sw_s": k_pres["mean_wrist_vel_sw_s"],
                "peak_wrist_sw_s": k_pres["peak_wrist_vel_sw_s"],
                "left_wrist_sw_s": _mean_col("rolling_30f_left_wrist_vel_sw_s"),
                "right_wrist_sw_s": _mean_col("rolling_30f_right_wrist_vel_sw_s"),
                "posture_transitions": k_pres["posture_transitions"],
            }
        )

    return profiles


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
    emp_ids = get_available_employees(df)

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
    """Aggregate scale-invariant rolling wrist velocity (`SW/s`) into a 2D [employee_channel x time_bin] matrix."""
    if df.is_empty():
        return {
            "x_centers": [0.0],
            "bin_seconds": bin_seconds,
            "unit": "SW/s",
            "y_labels": ["Dominant Wrist"],
            "z_matrix": [[0.0]],
            "hover_text": [[""]],
        }

    df = _normalize_track_id_columns(df)
    use_sw = "rolling_30f_wrist_vel_sw_s" in df.columns
    unit = "SW/s" if use_sw else "px/s"
    dom_col = "rolling_30f_wrist_vel_sw_s" if use_sw else "rolling_30f_wrist_vel_px_s"
    rw_col = "rolling_30f_right_wrist_vel_sw_s" if "rolling_30f_right_wrist_vel_sw_s" in df.columns else (
        dom_col if use_sw else "rolling_30f_right_wrist_vel_px_s"
    )
    lw_col = "rolling_30f_left_wrist_vel_sw_s" if "rolling_30f_left_wrist_vel_sw_s" in df.columns else (
        dom_col if use_sw else "rolling_30f_left_wrist_vel_px_s"
    )

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
        channel_specs = [
            ("Dominant Wrist (30f)", dom_col),
            ("Right Wrist (30f)", rw_col),
            ("Left Wrist (30f)", lw_col),
        ]
        present = df.filter(pl.col("presence") == 1)
        p_t = present["timestamp_sec"].to_numpy() if not present.is_empty() else np.array([])
        for label, col_name in channel_specs:
            if col_name not in df.columns:
                continue
            vals = (
                np.nan_to_num(present[col_name].cast(pl.Float64, strict=False).to_numpy(), nan=0.0)
                if not present.is_empty()
                else np.array([])
            )
            row_z: List[float] = []
            row_h: List[str] = []
            for b in range(num_bins):
                lo, hi = edges[b], edges[b + 1]
                mask = (p_t >= lo) & (p_t < hi if b < num_bins - 1 else p_t <= hi + 1e-6)
                v = float(np.mean(vals[mask])) if np.any(mask) else 0.0
                v_max = float(np.max(vals[mask])) if np.any(mask) else 0.0
                row_z.append(round(v, 3))
                row_h.append(
                    f"<b>{label}</b><br>Window: {lo:.1f}s – {hi:.1f}s<br>"
                    f"Mean Speed: {v:.2f} {unit}<br>Peak Speed: {v_max:.2f} {unit}"
                )
            y_labels.append(label)
            z_matrix.append(row_z)
            hover_matrix.append(row_h)
    else:
        for emp_id in reversed(emp_ids):
            emp_df = df.filter((pl.col("track_id") == emp_id) & (pl.col("presence") == 1))
            p_t = emp_df["timestamp_sec"].to_numpy() if not emp_df.is_empty() else np.array([])
            for ch_short, col_name in [
                ("Right Wrist", rw_col),
                ("Left Wrist", lw_col),
                ("Peak Wrist", dom_col),
            ]:
                if col_name not in df.columns:
                    continue
                label = f"Emp #{emp_id} • {ch_short}"
                vals = (
                    np.nan_to_num(emp_df[col_name].cast(pl.Float64, strict=False).to_numpy(), nan=0.0)
                    if not emp_df.is_empty()
                    else np.array([])
                )
                row_z = []
                row_h = []
                for b in range(num_bins):
                    lo, hi = edges[b], edges[b + 1]
                    mask = (p_t >= lo) & (p_t < hi if b < num_bins - 1 else p_t <= hi + 1e-6)
                    v = float(np.mean(vals[mask])) if np.any(mask) else 0.0
                    v_max = float(np.max(vals[mask])) if np.any(mask) else 0.0
                    row_z.append(round(v, 3))
                    row_h.append(
                        f"<b>{label}</b><br>Window: {lo:.1f}s – {hi:.1f}s<br>"
                        f"Mean Speed: {v:.2f} {unit}<br>Peak Speed: {v_max:.2f} {unit}"
                    )
                y_labels.append(label)
                z_matrix.append(row_z)
                hover_matrix.append(row_h)

    return {
        "x_centers": centers,
        "bin_seconds": bin_w,
        "unit": unit,
        "y_labels": y_labels,
        "z_matrix": z_matrix,
        "hover_text": hover_matrix,
    }


def build_employee_summary_table(df: pl.DataFrame) -> pl.DataFrame:
    """Build an executive per-employee (`1..N`) + aggregate KPI comparison table for UI & CSV export."""
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
            "Mean_Wrist_Speed_SW_s": overall["mean_wrist_vel_sw_s"],
            "Peak_Wrist_Speed_SW_s": overall["peak_wrist_vel_sw_s"],
            "Posture_Transitions": overall["posture_transitions"],
        }
    )

    for emp_id in get_available_employees(df):
        emp_full = filter_telemetry(df, employee_id=emp_id)
        emp_present_only = _normalize_track_id_columns(df).filter(
            (pl.col("track_id") == emp_id) & (pl.col("presence") == 1)
        )
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
                "Mean_Wrist_Speed_SW_s": k_pres["mean_wrist_vel_sw_s"],
                "Peak_Wrist_Speed_SW_s": k_pres["peak_wrist_vel_sw_s"],
                "Posture_Transitions": k_pres["posture_transitions"],
            }
        )

    return pl.DataFrame(rows)
