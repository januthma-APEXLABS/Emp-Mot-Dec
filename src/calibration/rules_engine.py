"""Scale-invariant behavioral rules engine with dictionary-based per-employee state
(`employee_states = {track_id: ...}`), Schmitt-trigger hysteresis, and minimum-dwell debouncing
across the 6 core motion classes: Sitting, Standing, Walking, Away, Idle, Working.

Zero hardcoded frame numbers and zero absolute pixel thresholds: every threshold is evaluated
strictly in dimensionless biological shoulder-width (`SW`, `SW/s`) or `[0, 1]` coordinates.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.calibration.calibrator import (
    CalibratedThresholds,
    calibrate_thresholds_from_features,
    generate_calibration_plots,
    kde_valley_or_otsu_1d,
)
from src.calibration.features import load_and_extract_features
from src.calibration.online_engine import HysteresisStateMachine
from src.config import PLOTS_OUTPUT_DIR, TELEMETRY_OUTPUT_DIR, ensure_directories

SIX_MOTION_CLASSES = ("Sitting", "Standing", "Walking", "Away", "Idle", "Working")


def _suppress_short_runs(states: List[str], min_dwell_frames: int) -> List[str]:
    """Suppress contiguous state runs shorter than `min_dwell_frames`."""
    n = len(states)
    out = list(states)
    dwell = max(int(min_dwell_frames), 1)
    if dwell <= 1 or n <= dwell:
        return out

    changed = True
    while changed:
        changed = False
        runs: List[Tuple[str, int, int]] = []
        run_start = 0
        for i in range(1, n + 1):
            if i == n or out[i] != out[run_start]:
                runs.append((out[run_start], run_start, i))
                run_start = i

        if len(runs) <= 1:
            break

        shortest_idx = -1
        shortest_len = dwell
        for r_idx, (_, s, e) in enumerate(runs):
            r_len = e - s
            if r_len < shortest_len:
                shortest_len = r_len
                shortest_idx = r_idx

        if shortest_idx != -1:
            _, s, e = runs[shortest_idx]
            replacement = runs[shortest_idx - 1][0] if shortest_idx > 0 else runs[shortest_idx + 1][0]
            for k in range(s, e):
                out[k] = replacement
            changed = True
    return out


def apply_hysteresis_and_dwell(
    values: np.ndarray,
    low_thresh: float,
    high_thresh: float,
    low_label: str,
    high_label: str,
    min_dwell_frames: int = 15,
) -> Tuple[List[str], int]:
    """Classify a 1D dimensionless temporal signal with Schmitt-trigger hysteresis
    and minimum-dwell short-run debouncing (`min_dwell_frames`).
    """
    n = len(values)
    if n == 0:
        return [], 0

    mid_thresh = 0.5 * (low_thresh + high_thresh)
    clean_vals = np.where(np.isfinite(values), values, mid_thresh)

    raw_states = [high_label if v > mid_thresh else low_label for v in clean_vals]
    raw_flips = sum(1 for i in range(1, n) if raw_states[i] != raw_states[i - 1])

    init_win = clean_vals[: min(max(int(min_dwell_frames), 1), n)]
    curr_state = high_label if float(np.median(init_win)) >= mid_thresh else low_label

    states = [curr_state] * n
    for i, val in enumerate(clean_vals):
        if curr_state == low_label and val >= high_thresh:
            curr_state = high_label
        elif curr_state == high_label and val <= low_thresh:
            curr_state = low_label
        states[i] = curr_state

    states = _suppress_short_runs(states, min_dwell_frames)
    final_flips = sum(1 for i in range(1, n) if states[i] != states[i - 1])
    return states, max(0, raw_flips - final_flips)


def _dilate_boolean_mask(mask: np.ndarray, half_width: int) -> np.ndarray:
    """Dilate True pulses by +/- `half_width` frames."""
    n = len(mask)
    if n == 0 or half_width <= 0:
        return mask.copy()
    out = np.zeros(n, dtype=bool)
    for idx in np.flatnonzero(mask):
        out[max(0, idx - half_width) : min(n, idx + half_width + 1)] = True
    return out


class BehaviorRulesEngine:
    """Classifies multi-employee telemetry into the 6 core classes using per-employee dictionary
    state machines (`self.employee_states`) and strictly dimensionless normalized features.
    """

    def __init__(self, thresholds: CalibratedThresholds) -> None:
        self.thresholds = thresholds
        self.employee_states: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def classify_dataframe(self, feature_df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict]:
        """Classify every row into Posture and Activity with zero pixel thresholds."""
        if feature_df.is_empty():
            return feature_df, {}

        if "track_id" not in feature_df.columns and "Track_ID" in feature_df.columns:
            feature_df = feature_df.with_columns(pl.col("Track_ID").alias("track_id"))

        required_defaults = {
            "avg_shoulder_x_norm": 0.5,
            "shoulder_dist_bottom_norm": 0.3,
            "spine_to_shoulder_ratio": 1.2,
            "bbox_to_shoulder_ratio": 2.5,
            "bbox_aspect_h_w": 1.6,
            "rolling_30f_wrist_vel_sw_s": 0.0,
            "torso_horiz_vel_sw_s": 0.0,
            "torso_horiz_span_30f_sw": 0.0,
            "torso_2d_vel_sw_s": 0.0,
            "torso_2d_span_30f_sw": 0.0,
            "min_wrist_to_ear_norm": np.nan,
            "ear_wrist_rel_std_norm": 1.0,
            "min_wrist_to_face_norm": np.nan,
            "face_wrist_vert_ptp_norm": 0.0,
            "head_pitch_ratio": np.nan,
            "head_forward_sym_norm": np.nan,
            "head_gaze_std_norm": np.nan,
        }
        missing_cols = [
            pl.lit(val).alias(c) for c, val in required_defaults.items() if c not in feature_df.columns
        ]
        if missing_cols:
            feature_df = feature_df.with_columns(missing_cols)

        df_indexed = feature_df.with_row_index("_row_id")
        n_rows = len(df_indexed)

        presence_state = ["Away"] * n_rows
        camera_zone = ["-"] * n_rows
        posture_state = ["Away"] * n_rows
        activity_state = ["Away"] * n_rows
        motion_state = ["Away"] * n_rows

        posture_flickers_suppressed = 0
        activity_flickers_suppressed = 0
        self.employee_states.clear()

        present_df = df_indexed.filter((pl.col("presence") == 1) & pl.col("track_id").is_not_null()).sort(
            ["video_name", "track_id", "frame"]
        )

        dwell = max(int(self.thresholds.min_dwell_frames), 1)
        walk_dwell = min(8, max(dwell // 2, 3))

        for (vid_name, tid_val), group in present_df.group_by(["video_name", "track_id"], maintain_order=True):
            emp_key = (str(vid_name), int(tid_val))
            r_ids = group["_row_id"].to_numpy()
            m = len(r_ids)
            fps_med = float(group["fps"].drop_nulls().median() or self.thresholds.fps or 30.0)
            idle_confirm_frames = max(int(round(self.thresholds.idle_confirm_seconds * fps_med)), dwell)

            sh_x_norm = group["avg_shoulder_x_norm"].to_numpy()
            dist_bot_norm = group["shoulder_dist_bottom_norm"].to_numpy()
            spine_ratio = np.nan_to_num(group["spine_to_shoulder_ratio"].to_numpy(), nan=1.0)
            bbox_ratio = np.nan_to_num(group["bbox_to_shoulder_ratio"].to_numpy(), nan=2.5)
            bbox_aspect = np.nan_to_num(group["bbox_aspect_h_w"].to_numpy(), nan=1.6)
            horiz_vel_sw = np.nan_to_num(group["torso_horiz_vel_sw_s"].to_numpy(), nan=0.0)
            horiz_span_sw = np.nan_to_num(group["torso_horiz_span_30f_sw"].to_numpy(), nan=0.0)
            t2d_vel_sw = np.nan_to_num(group["torso_2d_vel_sw_s"].to_numpy(), nan=0.0)
            t2d_span_sw = np.nan_to_num(group["torso_2d_span_30f_sw"].to_numpy(), nan=0.0)
            wrist_vel_sw = np.nan_to_num(group["rolling_30f_wrist_vel_sw_s"].to_numpy(), nan=0.0)

            ear_dist = group["min_wrist_to_ear_norm"].to_numpy()
            ear_std = np.nan_to_num(group["ear_wrist_rel_std_norm"].to_numpy(), nan=1.0)
            face_dist = group["min_wrist_to_face_norm"].to_numpy()
            face_ptp = np.nan_to_num(group["face_wrist_vert_ptp_norm"].to_numpy(), nan=0.0)

            pitch_r = group["head_pitch_ratio"].to_numpy()
            fwd_sym = group["head_forward_sym_norm"].to_numpy()
            gaze_std = group["head_gaze_std_norm"].to_numpy()

            # Select perspective zone in normalized [0, 1] X space
            if (
                self.thresholds.use_perspective_zones
                and self.thresholds.perspective_split_x_norm is not None
                and np.any(np.isfinite(sh_x_norm))
            ):
                stat_x = sh_x_norm[np.isfinite(sh_x_norm) & (horiz_vel_sw < 0.38)]
                med_x_norm = float(np.median(stat_x)) if len(stat_x) >= 10 else float(np.nanmedian(sh_x_norm))
                matched_zone = "global"
                for z_name, z_cfg_cand in self.thresholds.posture_zones.items():
                    if z_name == "global":
                        continue
                    if z_cfg_cand.x_min_norm <= med_x_norm <= z_cfg_cand.x_max_norm:
                        matched_zone = z_name
                        break
            else:
                matched_zone = "global"

            z_cfg = self.thresholds.posture_zones.get(matched_zone, self.thresholds.posture_zones["global"])

            # Per-Employee Desk Posture Calibration (handles multi-row open offices + any camera mount!)
            clean_dist = dist_bot_norm[np.isfinite(dist_bot_norm)]
            stat_dist = dist_bot_norm[np.isfinite(dist_bot_norm) & (horiz_vel_sw < 0.50)]
            emp_pool = stat_dist if len(stat_dist) >= 15 else clean_dist
            med_bbox_r = float(np.median(bbox_ratio[np.isfinite(bbox_ratio)])) if len(bbox_ratio) > 0 else 2.5
            med_aspect = float(np.median(bbox_aspect[np.isfinite(bbox_aspect)])) if len(bbox_aspect) > 0 else 1.6

            if len(emp_pool) >= 20:
                ep10, ep50, ep95 = np.percentile(emp_pool, [10.0, 50.0, 95.0])
                emp_spread = float(ep95 - ep10)
            else:
                ep50 = float(np.median(emp_pool)) if len(emp_pool) > 0 else z_cfg.threshold_norm
                emp_spread = 0.0

            if emp_spread >= 0.055:
                # Employee transitions between Sitting & Standing at their own workstation depth
                emp_t, low_pk, high_pk, _ = kde_valley_or_otsu_1d(emp_pool, prefer_upper_valley=True)
                if (high_pk - low_pk) < 0.045:
                    emp_t = float(ep50 + 0.055)
                emp_enter_stand = emp_t * 1.035
                emp_exit_sit = emp_t * 0.965
            else:
                # Unimodal elevation for this employee: determine whether always-Standing or always-Sitting
                is_upright_standing = (
                    (med_bbox_r >= 3.48 and med_aspect >= 2.25)
                    or (
                        ep50 >= z_cfg.enter_standing_norm + 0.04
                        and med_bbox_r >= 3.15
                        and med_aspect >= 2.05
                    )
                )
                if is_upright_standing:
                    emp_t = max(ep50 - 0.06, 0.12)
                else:
                    emp_t = min(ep50 + 0.065, 0.96)
                emp_enter_stand = emp_t * 1.03
                emp_exit_sit = emp_t * 0.97

            # Perspective close-camera biological ratio check (dimensionless spine_to_shoulder & bbox_to_shoulder)
            dist_norm_eff = dist_bot_norm.copy()
            l_hc = np.nan_to_num(group["left_hip_conf"].to_numpy(), nan=0.0) if "left_hip_conf" in group.columns else np.ones(m)
            r_hc = np.nan_to_num(group["right_hip_conf"].to_numpy(), nan=0.0) if "right_hip_conf" in group.columns else np.ones(m)
            valid_hips = (l_hc >= 0.30) & (r_hc >= 0.30) & np.isfinite(spine_ratio)
            seated_sp_r = spine_ratio[valid_hips & np.isfinite(dist_bot_norm) & (dist_bot_norm <= emp_exit_sit)]
            if len(seated_sp_r) >= 20:
                seated_sp_med = float(np.median(seated_sp_r))
                closeup_upright = (
                    valid_hips
                    & (spine_ratio >= 1.85 * seated_sp_med)
                    & (dist_bot_norm >= 0.90 * emp_t)
                )
                dist_norm_eff = np.where(
                    closeup_upright,
                    np.maximum(dist_norm_eff, emp_enter_stand + 0.025),
                    dist_norm_eff,
                )

            # Full-body upright bounding box aspect reinforcement (e.g., walking/standing in foreground aisle)
            full_upright_mask = (bbox_ratio >= 3.55) & (bbox_aspect >= 2.25)
            dist_norm_eff = np.where(
                full_upright_mask,
                np.maximum(dist_norm_eff, emp_enter_stand + 0.02),
                dist_norm_eff,
            )

            # Initialize per-employee state dictionary entry with debounced FSMs
            p_base, p_sup = apply_hysteresis_and_dwell(
                values=dist_norm_eff,
                low_thresh=emp_exit_sit,
                high_thresh=emp_enter_stand,
                low_label="Sitting",
                high_label="Standing",
                min_dwell_frames=dwell,
            )
            posture_flickers_suppressed += p_sup

            walk_hi = self.thresholds.walking_horiz_vel_threshold_sw_s
            walk_lo = 0.80 * walk_hi
            eff_loco_vel = np.maximum(horiz_vel_sw, np.where(horiz_span_sw >= 0.35, t2d_vel_sw, 0.0))
            w_vel_labels, _ = apply_hysteresis_and_dwell(
                values=eff_loco_vel,
                low_thresh=walk_lo,
                high_thresh=walk_hi,
                low_label="Stationary",
                high_label="Walking",
                min_dwell_frames=walk_dwell,
            )

            posture_fsm = HysteresisStateMachine(initial_state=p_base[0], min_dwell_frames=walk_dwell)
            posture_fsm.initialized = True
            p_final: List[str] = []
            for i in range(m):
                is_walk = (
                    (
                        p_base[i] == "Standing"
                        and w_vel_labels[i] == "Walking"
                        and (horiz_span_sw[i] >= 0.42 or t2d_span_sw[i] >= 1.55)
                    )
                    or (horiz_vel_sw[i] >= 1.25 and horiz_span_sw[i] >= 1.35)
                )
                cand_p = "Walking" if is_walk else p_base[i]
                p_final.append(posture_fsm.step(cand_p))
            posture_flickers_suppressed += posture_fsm.flickers_suppressed

            # --- 2. Scale-Invariant Workstation Activity: Working vs Idle ---
            not_walking = np.array([p != "Walking" for p in p_final], dtype=bool)

            a_wrist, a_sup = apply_hysteresis_and_dwell(
                values=wrist_vel_sw,
                low_thresh=self.thresholds.exit_to_idle_vel_sw_s,
                high_thresh=self.thresholds.enter_working_vel_sw_s,
                low_label="Idle",
                high_label="Working",
                min_dwell_frames=dwell,
            )

            # Phone Call & Screen Focus streaks + Drinking cyclical transition
            drink_sip_peak = (
                not_walking
                & np.isfinite(face_dist)
                & (face_dist <= self.thresholds.drinking_face_dist_norm_threshold)
                & (face_ptp >= 0.55)
            )
            drink_dilated = _dilate_boolean_mask(drink_sip_peak, half_width=10) & not_walking

            phone_req = min(max(int(round(0.75 * fps_med)), 12), m)
            focus_req = min(dwell, m)

            # Determine initial activity state over first `dwell` frames so tracks starting idle begin Idle
            init_act_slice = a_wrist[: min(dwell, m)]
            init_act = "Working" if sum(1 for x in init_act_slice if x == "Working") > (len(init_act_slice) // 2) else "Idle"
            act_fsm = HysteresisStateMachine(
                initial_state=init_act,
                min_dwell_frames=dwell,
                custom_dwell_overrides={("Working", "Idle"): idle_confirm_frames},
            )
            act_fsm.initialized = True

            phone_streak = 0
            focus_streak = 0
            drink_streak = 0
            confirmed_work: List[str] = []

            for i in range(m):
                if (
                    not_walking[i]
                    and np.isfinite(ear_dist[i])
                    and ear_dist[i] <= self.thresholds.phone_ear_dist_norm_threshold
                    and ear_std[i] <= 0.09
                ):
                    phone_streak += 1
                else:
                    phone_streak = 0
                is_phone = phone_streak >= phone_req

                if drink_dilated[i] and not is_phone:
                    drink_streak += 1
                else:
                    drink_streak = 0
                is_drink = drink_streak >= min(8, m)

                tilted_down = np.isfinite(pitch_r[i]) and pitch_r[i] >= 0.22 and np.nan_to_num(gaze_std[i], nan=0.0) <= 0.09
                locked_forward = (
                    np.isfinite(pitch_r[i])
                    and pitch_r[i] >= 0.17
                    and np.isfinite(fwd_sym[i])
                    and fwd_sym[i] <= 0.14
                    and np.isfinite(gaze_std[i])
                    and gaze_std[i] <= 0.05
                )
                if not_walking[i] and not is_phone and not is_drink and a_wrist[i] == "Idle" and (tilted_down or locked_forward):
                    focus_streak += 1
                else:
                    focus_streak = 0
                is_focus = focus_streak >= focus_req

                if is_drink:
                    prop_act = "Idle"
                elif a_wrist[i] == "Working" or is_phone or is_focus:
                    prop_act = "Working"
                else:
                    prop_act = "Idle"

                confirmed_work.append(
                    act_fsm.step(prop_act, immediate_override=(is_drink or is_phone or is_focus))
                )

            activity_flickers_suppressed += a_sup + act_fsm.flickers_suppressed

            self.employee_states[emp_key] = {
                "Track_ID": int(tid_val),
                "camera_zone": matched_zone,
                "frames_observed": m,
                "posture_fsm": posture_fsm,
                "activity_fsm": act_fsm,
            }

            for i, rid in enumerate(r_ids):
                r_int = int(rid)
                p_lbl = p_final[i]
                presence_state[r_int] = "Present"
                camera_zone[r_int] = matched_zone
                posture_state[r_int] = p_lbl

                if p_lbl == "Walking":
                    activity_state[r_int] = "Walking"
                    motion_state[r_int] = "Walking"
                else:
                    a_lbl = confirmed_work[i]
                    activity_state[r_int] = a_lbl
                    motion_state[r_int] = f"{p_lbl} | {a_lbl}"

        classified = (
            df_indexed.with_columns(
                [
                    pl.col("track_id").alias("Track_ID"),
                    pl.Series("presence_state", presence_state),
                    pl.Series("camera_zone", camera_zone),
                    pl.Series("posture_state", posture_state),
                    pl.Series("activity_state", activity_state),
                    pl.Series("motion_state", motion_state),
                ]
            )
            .sort(["video_name", "frame", "track_id"], nulls_last=True)
            .drop("_row_id")
        )

        def _dist_dict(col_name: str) -> Dict[str, int]:
            rows = classified.group_by(col_name).len().sort("len", descending=True).to_dicts()
            return {str(d[col_name]): int(d["len"]) for d in rows}

        p_dist = _dist_dict("posture_state")
        a_dist = _dist_dict("activity_state")
        six_class_counts = {
            "Sitting": p_dist.get("Sitting", 0),
            "Standing": p_dist.get("Standing", 0),
            "Walking": p_dist.get("Walking", 0),
            "Away": p_dist.get("Away", 0),
            "Working": a_dist.get("Working", 0),
            "Idle": a_dist.get("Idle", 0),
        }

        metrics = {
            "total_rows": int(len(classified)),
            "unique_videos": int(classified["video_name"].n_unique()),
            "unique_tracks": int(classified.filter(pl.col("presence") == 1)["track_id"].n_unique()),
            "posture_distribution_rows": p_dist,
            "activity_distribution_rows": a_dist,
            "motion_state_distribution_rows": _dist_dict("motion_state"),
            "six_class_summary": six_class_counts,
            "posture_flickers_suppressed": int(posture_flickers_suppressed),
            "activity_flickers_suppressed": int(activity_flickers_suppressed),
        }
        return classified, metrics


def generate_state_timeline_plot(
    classified_df: pl.DataFrame,
    plots_dir: Path,
) -> Path:
    """Render state timeline chart for each tracked employee across the 6 classes."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "milestone2_state_timelines.png"

    present = classified_df.filter(pl.col("presence") == 1)
    if present.is_empty():
        return out_path

    top_tracks = (
        present.group_by("track_id")
        .len()
        .sort("len", descending=True)
        .head(4)["track_id"]
        .to_list()
    )

    n_tracks = len(top_tracks)
    fig, axes = plt.subplots(n_tracks, 1, figsize=(14, 3.2 * max(n_tracks, 1)), sharex=False)
    if n_tracks == 1:
        axes = [axes]

    posture_map = {"Sitting": 0.0, "Standing": 0.5, "Walking": 1.0}
    activity_map = {"Idle": 0.0, "Working": 1.0, "Walking": 0.5}

    for ax, tid in zip(axes, top_tracks):
        sub = present.filter(pl.col("track_id") == tid).sort("frame")
        t_sec = sub["timestamp_sec"].to_numpy()
        p_vals = np.array([posture_map.get(s, 0.0) for s in sub["posture_state"].to_list()])
        a_vals = np.array([activity_map.get(s, 0.0) for s in sub["activity_state"].to_list()])
        zone = sub["camera_zone"][0]

        ax.step(t_sec, p_vals + 1.35, where="post", color="#8e44ad", linewidth=2.0, label="Posture (Sitting / Standing / Walking)")
        ax.step(t_sec, a_vals, where="post", color="#27ae60", linewidth=2.0, label="Activity (Idle / Working)")
        ax.set_yticks([0.0, 1.0, 1.35, 1.85, 2.35])
        ax.set_yticklabels(["Idle", "Working", "Sitting", "Standing", "Walking"])
        ax.set_ylim(-0.25, 2.65)
        ax.set_title(f"Emp #{tid} ({zone.capitalize()} Zone, {len(sub)} frames) — 6-Class Stabilized Timeline")
        ax.set_xlabel("Timestamp (seconds)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_motion_categories_plot(
    classified_df: pl.DataFrame,
    plots_dir: Path,
) -> Path:
    """Render 2-panel 6-class summary chart: Spatial Posture/Presence vs Workstation Activity."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "motion_categories_breakdown.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    total_rows = max(len(classified_df), 1)

    p_counts = {str(d["posture_state"]): int(d["len"]) for d in classified_df.group_by("posture_state").len().to_dicts()}
    p_cats = ["Sitting", "Standing", "Walking", "Away"]
    p_vals = [p_counts.get(c, 0) for c in p_cats]
    p_colors = ["#27ae60", "#2980b9", "#8e44ad", "#95a5a6"]
    bars1 = ax1.barh(np.arange(len(p_cats)), p_vals, color=p_colors, alpha=0.88)
    ax1.set_yticks(np.arange(len(p_cats)))
    ax1.set_yticklabels(p_cats, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_title("Spatial Track (Sitting / Standing / Walking / Away)", fontweight="bold")
    ax1.set_xlabel("Telemetry Rows (Frames)")
    ax1.set_xlim(0, max(max(p_vals, default=1) * 1.25, 10))
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35)
    for bar, v in zip(bars1, p_vals):
        ax1.text(bar.get_width() + max(p_vals, default=1) * 0.02, bar.get_y() + bar.get_height() / 2.0, f"{v} ({100.0 * v / total_rows:.1f}%)", va="center", fontsize=9)

    a_counts = {str(d["activity_state"]): int(d["len"]) for d in classified_df.group_by("activity_state").len().to_dicts()}
    a_cats = ["Working", "Idle", "Walking", "Away"]
    a_vals = [a_counts.get(c, 0) for c in a_cats]
    a_colors = ["#16a085", "#e67e22", "#8e44ad", "#95a5a6"]
    bars2 = ax2.barh(np.arange(len(a_cats)), a_vals, color=a_colors, alpha=0.88)
    ax2.set_yticks(np.arange(len(a_cats)))
    ax2.set_yticklabels(a_cats, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_title("Activity Track (Working / Idle / Walking / Away)", fontweight="bold")
    ax2.set_xlabel("Telemetry Rows (Frames)")
    ax2.set_xlim(0, max(max(a_vals, default=1) * 1.25, 10))
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35)
    for bar, v in zip(bars2, a_vals):
        ax2.text(bar.get_width() + max(a_vals, default=1) * 0.02, bar.get_y() + bar.get_height() / 2.0, f"{v} ({100.0 * v / total_rows:.1f}%)", va="center", fontsize=9)

    plt.suptitle("6-Class Employee Motion Summary (Sitting, Standing, Walking, Away, Idle, Working)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_milestone2_pipeline(
    input_csv_path: Path = TELEMETRY_OUTPUT_DIR / "batch_landmarks.csv",
    output_telemetry_dir: Path = TELEMETRY_OUTPUT_DIR,
    output_plots_dir: Path = PLOTS_OUTPUT_DIR,
    wrist_window_frames: int = 30,
    min_dwell_frames: int = 15,
    thresholds_override_path: Optional[Path] = None,
    enable_perspective_zones: bool = True,
) -> Dict:
    """End-to-end scale-invariant 6-class calibration, classification, and plot pipeline."""
    ensure_directories()
    output_telemetry_dir.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    feature_df = load_and_extract_features(input_csv_path, wrist_window_frames=wrist_window_frames)

    video_names = sorted(feature_df["video_name"].drop_nulls().unique().to_list()) if "video_name" in feature_df.columns else []
    per_video_thresholds_dict: Dict[str, Dict] = {}

    if thresholds_override_path is not None and thresholds_override_path.exists():
        thresholds = CalibratedThresholds.load_json(thresholds_override_path)
        engine = BehaviorRulesEngine(thresholds=thresholds)
        classified_df, classification_metrics = engine.classify_dataframe(feature_df)
    elif len(video_names) > 1:
        # Calibrate & classify each camera/video stream independently so distinct camera angles never pollute each other
        classified_parts: List[pl.DataFrame] = []
        total_p_sup = 0
        total_a_sup = 0
        primary_thresholds: Optional[CalibratedThresholds] = None

        for v_name in video_names:
            v_feat = feature_df.filter(pl.col("video_name") == v_name)
            v_present = v_feat.filter(pl.col("presence") == 1)
            if v_present.is_empty():
                v_thresh = calibrate_thresholds_from_features(
                    feature_df=feature_df,
                    wrist_window_frames=wrist_window_frames,
                    min_dwell_frames=min_dwell_frames,
                    enable_perspective_zones=enable_perspective_zones,
                )
            else:
                v_thresh = calibrate_thresholds_from_features(
                    feature_df=v_feat,
                    wrist_window_frames=wrist_window_frames,
                    min_dwell_frames=min_dwell_frames,
                    enable_perspective_zones=enable_perspective_zones,
                )
            if primary_thresholds is None:
                primary_thresholds = v_thresh
            v_thresh.save_json(output_telemetry_dir / f"{Path(v_name).stem}_thresholds.json")
            per_video_thresholds_dict[v_name] = v_thresh.to_dict()

            v_engine = BehaviorRulesEngine(thresholds=v_thresh)
            v_cls, v_met = v_engine.classify_dataframe(v_feat)
            classified_parts.append(v_cls)
            total_p_sup += int(v_met.get("posture_flickers_suppressed", 0))
            total_a_sup += int(v_met.get("activity_flickers_suppressed", 0))

        thresholds = primary_thresholds or calibrate_thresholds_from_features(
            feature_df=feature_df,
            wrist_window_frames=wrist_window_frames,
            min_dwell_frames=min_dwell_frames,
            enable_perspective_zones=enable_perspective_zones,
        )
        classified_df = pl.concat(classified_parts, how="vertical_relaxed").sort(
            ["video_name", "frame", "track_id"], nulls_last=True
        )

        def _agg_dist(col_name: str) -> Dict[str, int]:
            rows = classified_df.group_by(col_name).len().sort("len", descending=True).to_dicts()
            return {str(d[col_name]): int(d["len"]) for d in rows}

        p_dist = _agg_dist("posture_state")
        a_dist = _agg_dist("activity_state")
        classification_metrics = {
            "total_rows": int(len(classified_df)),
            "unique_videos": int(classified_df["video_name"].n_unique()),
            "unique_tracks": int(
                classified_df.filter(pl.col("presence") == 1)
                .select(["video_name", "track_id"])
                .unique()
                .height
            ),
            "posture_distribution_rows": p_dist,
            "activity_distribution_rows": a_dist,
            "motion_state_distribution_rows": _agg_dist("motion_state"),
            "six_class_summary": {
                "Sitting": p_dist.get("Sitting", 0),
                "Standing": p_dist.get("Standing", 0),
                "Walking": p_dist.get("Walking", 0),
                "Away": p_dist.get("Away", 0),
                "Working": a_dist.get("Working", 0),
                "Idle": a_dist.get("Idle", 0),
            },
            "posture_flickers_suppressed": total_p_sup,
            "activity_flickers_suppressed": total_a_sup,
        }
    else:
        thresholds = calibrate_thresholds_from_features(
            feature_df=feature_df,
            wrist_window_frames=wrist_window_frames,
            min_dwell_frames=min_dwell_frames,
            enable_perspective_zones=enable_perspective_zones,
        )
        engine = BehaviorRulesEngine(thresholds=thresholds)
        classified_df, classification_metrics = engine.classify_dataframe(feature_df)

    thresholds_path = output_telemetry_dir / "calibrated_thresholds.json"
    thresholds.save_json(thresholds_path)

    temporal_csv_path = output_telemetry_dir / "temporal_features.csv"
    classified_df.write_csv(temporal_csv_path, float_precision=4)

    per_video_paths: List[str] = []
    for vid_tuple, v_df in classified_df.group_by("video_name", maintain_order=True):
        vid_name = str(vid_tuple[0])
        vid_stem = Path(vid_name).stem
        v_out = output_telemetry_dir / f"{vid_stem}_classified.csv"
        v_df.write_csv(v_out, float_precision=4)
        per_video_paths.append(str(v_out))

    plot_paths = generate_calibration_plots(
        classified_df=classified_df,
        thresholds=thresholds,
        plots_dir=output_plots_dir,
    )
    timeline_plot_path = generate_state_timeline_plot(
        classified_df=classified_df,
        plots_dir=output_plots_dir,
    )
    categories_plot_path = generate_motion_categories_plot(
        classified_df=classified_df,
        plots_dir=output_plots_dir,
    )

    summary = {
        "input_csv_path": str(input_csv_path),
        "temporal_features_csv_path": str(temporal_csv_path),
        "per_video_classified_csvs": per_video_paths,
        "calibrated_thresholds_json_path": str(thresholds_path),
        "calibrated_thresholds": thresholds.to_dict(),
        "classification_metrics": classification_metrics,
        "plots": {
            "posture_calibration": str(plot_paths["posture_calibration_plot"]),
            "wrist_velocity_calibration": str(plot_paths["wrist_velocity_calibration_plot"]),
            "state_timelines": str(timeline_plot_path),
            "motion_categories_breakdown": str(categories_plot_path),
        },
    }

    summary_json_path = output_telemetry_dir / "milestone2_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
