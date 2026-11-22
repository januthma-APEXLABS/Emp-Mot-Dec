"""Scale-invariant mathematical feature engineering using per-employee dictionary state
and `collections.deque(maxlen=30)` rolling windows.
"""

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Tuple, Union
import numpy as np
import polars as pl

from src.config import TARGET_LANDMARKS


def _sliding_ptp_1d(arr: np.ndarray, half_window: int) -> np.ndarray:
    """Centered sliding peak-to-peak range (max - min) on normalized arrays."""
    m = len(arr)
    if m == 0:
        return np.array([], dtype=np.float64)
    hw = max(int(half_window), 1)
    out = np.zeros(m, dtype=np.float64)
    for i in range(m):
        lo = max(0, i - hw)
        hi = min(m, i + hw + 1)
        seg = arr[lo:hi]
        valid = seg[np.isfinite(seg)]
        if len(valid) >= 2:
            out[i] = float(np.max(valid) - np.min(valid))
    return out


def compute_temporal_features(
    df: pl.DataFrame,
    wrist_window_frames: int = 30,
    max_frame_gap: int = 3,
    min_wrist_conf: float = 0.10,
) -> pl.DataFrame:
    """Compute scale-invariant posture, `deque(maxlen=30)` rolling wrist kinematics,
    and head/ear/face relationships per `Track_ID`.
    """
    if df.is_empty():
        return df

    if "track_id" not in df.columns and "Track_ID" in df.columns:
        df = df.with_columns(pl.col("Track_ID").alias("track_id"))

    missing_exprs = []
    for j_name in TARGET_LANDMARKS:
        for suf, default_val in (("x", None), ("y", None), ("conf", 0.0)):
            col_name = f"{j_name}_{suf}"
            if col_name not in df.columns:
                missing_exprs.append(pl.lit(default_val, dtype=pl.Float64).alias(col_name))
    if missing_exprs:
        df = df.with_columns(missing_exprs)

    float_cols = [
        "timestamp_sec",
        "fps",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_conf",
        "wrist_pixel_motion_ratio",
        "lk_wrist_flow_sw_s",
    ] + [f"{j_name}_{suf}" for j_name in TARGET_LANDMARKS for suf in ("x", "y", "conf")]

    cast_exprs = []
    for col in float_cols:
        if col in df.columns:
            if df.schema[col] in (pl.Utf8, pl.String):
                cast_exprs.append(
                    pl.when(pl.col(col).str.strip_chars() == "")
                    .then(None)
                    .otherwise(pl.col(col))
                    .cast(pl.Float64, strict=False)
                    .alias(col)
                )
            else:
                cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))

    for int_col in ("frame", "frame_width", "frame_height", "presence", "track_id"):
        if int_col in df.columns:
            if df.schema[int_col] in (pl.Utf8, pl.String):
                cast_exprs.append(
                    pl.when(pl.col(int_col).str.strip_chars() == "")
                    .then(None)
                    .otherwise(pl.col(int_col))
                    .cast(pl.Int64, strict=False)
                    .alias(int_col)
                )
            else:
                cast_exprs.append(pl.col(int_col).cast(pl.Int64, strict=False).alias(int_col))

    df_clean = df.with_columns(cast_exprs).with_row_index("_orig_row_idx")

    avg_sh_x = (pl.col("left_shoulder_x") + pl.col("right_shoulder_x")) / 2.0
    avg_sh_y = (pl.col("left_shoulder_y") + pl.col("right_shoulder_y")) / 2.0
    avg_hip_x = (pl.col("left_hip_x") + pl.col("right_hip_x")) / 2.0
    avg_hip_y = (pl.col("left_hip_y") + pl.col("right_hip_y")) / 2.0

    frame_w_f = pl.max_horizontal(pl.col("frame_width").cast(pl.Float64), pl.lit(1.0))
    frame_h_f = pl.max_horizontal(pl.col("frame_height").cast(pl.Float64), pl.lit(1.0))
    sh_dist_bot_px = frame_h_f - avg_sh_y
    sh_dist_bot_norm = sh_dist_bot_px / frame_h_f
    hip_dist_bot_px = frame_h_f - avg_hip_y

    spine_dx = (avg_sh_x - avg_hip_x).abs()
    spine_dy = avg_hip_y - avg_sh_y
    spine_len_px = ((avg_hip_x - avg_sh_x) ** 2 + (avg_hip_y - avg_sh_y) ** 2).sqrt()
    spine_len_norm = spine_len_px / frame_h_f

    sh_width_px = (
        (pl.col("left_shoulder_x") - pl.col("right_shoulder_x")) ** 2
        + (pl.col("left_shoulder_y") - pl.col("right_shoulder_y")) ** 2
    ).sqrt()

    df_geom = df_clean.with_columns(
        [
            pl.col("track_id").alias("Track_ID"),
            pl.when(pl.col("presence") == 1).then(avg_sh_x).otherwise(None).alias("avg_shoulder_x"),
            pl.when(pl.col("presence") == 1).then(avg_sh_x / frame_w_f).otherwise(None).alias("avg_shoulder_x_norm"),
            pl.when(pl.col("presence") == 1).then(avg_sh_y).otherwise(None).alias("avg_shoulder_y"),
            pl.when(pl.col("presence") == 1).then(avg_hip_x).otherwise(None).alias("avg_hip_x"),
            pl.when(pl.col("presence") == 1).then(avg_hip_y).otherwise(None).alias("avg_hip_y"),
            pl.when(pl.col("presence") == 1).then(sh_dist_bot_px).otherwise(None).alias("shoulder_dist_bottom_px"),
            pl.when(pl.col("presence") == 1).then(sh_dist_bot_norm).otherwise(None).alias("shoulder_dist_bottom_norm"),
            pl.when(pl.col("presence") == 1).then(hip_dist_bot_px).otherwise(None).alias("hip_dist_bottom_px"),
            pl.when(pl.col("presence") == 1).then(spine_len_px).otherwise(None).alias("spine_length_px"),
            pl.when(pl.col("presence") == 1).then(spine_len_norm).otherwise(None).alias("spine_length_norm"),
            pl.when(pl.col("presence") == 1).then(sh_width_px).otherwise(None).alias("shoulder_width_px"),
            pl.when(pl.col("presence") == 1)
            .then((spine_dx / pl.max_horizontal(spine_dy, pl.lit(1e-3))).arctan() * (180.0 / np.pi))
            .otherwise(None)
            .alias("spine_angle_deg"),
        ]
    )

    n_rows = len(df_geom)
    bio_scale_arr = np.full(n_rows, np.nan, dtype=np.float64)
    spine_to_sh_arr = np.full(n_rows, np.nan, dtype=np.float64)
    bbox_to_sh_arr = np.full(n_rows, np.nan, dtype=np.float64)
    bbox_aspect_arr = np.full(n_rows, np.nan, dtype=np.float64)

    lw_vel_sw_s = np.full(n_rows, np.nan, dtype=np.float64)
    rw_vel_sw_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_lw_sw_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_rw_sw_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_wrist_sw_s = np.full(n_rows, np.nan, dtype=np.float64)
    wrist_var_30f_sw = np.full(n_rows, np.nan, dtype=np.float64)
    torso_horiz_vel_sw_s = np.full(n_rows, np.nan, dtype=np.float64)
    torso_horiz_span_sw = np.full(n_rows, np.nan, dtype=np.float64)
    torso_2d_vel_sw_s = np.full(n_rows, np.nan, dtype=np.float64)
    torso_2d_span_sw = np.full(n_rows, np.nan, dtype=np.float64)

    lw_vel_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    rw_vel_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_lw_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_rw_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_wrist_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_wrist_px_frame = np.full(n_rows, np.nan, dtype=np.float64)
    roll_wrist_norm_s = np.full(n_rows, np.nan, dtype=np.float64)
    torso_horiz_vel_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    torso_horiz_span_24f_px = np.full(n_rows, np.nan, dtype=np.float64)

    bilateral_wrist_ratio = np.full(n_rows, np.nan, dtype=np.float64)
    dom_wrist_rel_std_norm = np.full(n_rows, np.nan, dtype=np.float64)
    min_wrist_to_ear_norm = np.full(n_rows, np.nan, dtype=np.float64)
    ear_wrist_rel_std_norm = np.full(n_rows, np.nan, dtype=np.float64)
    min_wrist_to_face_norm = np.full(n_rows, np.nan, dtype=np.float64)
    face_wrist_vert_ptp_norm = np.full(n_rows, np.nan, dtype=np.float64)
    head_pitch_ratio = np.full(n_rows, np.nan, dtype=np.float64)
    head_forward_sym_norm = np.full(n_rows, np.nan, dtype=np.float64)
    head_gaze_std_norm = np.full(n_rows, np.nan, dtype=np.float64)
    torso_compression_ratio = np.full(n_rows, np.nan, dtype=np.float64)
    both_wrists_above_sh = np.zeros(n_rows, dtype=np.int64)

    present_df = df_geom.filter((pl.col("presence") == 1) & pl.col("track_id").is_not_null()).sort(
        ["video_name", "track_id", "frame"]
    )

    window = max(int(wrist_window_frames), 2)
    employee_states: Dict[Tuple[str, int], Dict[str, Deque[Any]]] = {}

    for (vid_name, tid_val), group in present_df.group_by(["video_name", "track_id"], maintain_order=True):
        emp_key = (str(vid_name), int(tid_val))
        employee_states[emp_key] = {
            "sh_scale_win": deque(maxlen=window),
            "lw_vel_win": deque(maxlen=window),
            "rw_vel_win": deque(maxlen=window),
            "dom_vel_win": deque(maxlen=window),
            "lw_rel_win": deque(maxlen=window),
            "rw_rel_win": deque(maxlen=window),
            "lw_short_win": deque(maxlen=max(window // 2, 12)),
            "rw_short_win": deque(maxlen=max(window // 2, 12)),
            "nose_rel_win": deque(maxlen=window),
            "torso_x_win": deque(maxlen=window),
        }
        emp_mem = employee_states[emp_key]

        idxs = group["_orig_row_idx"].to_numpy()
        frames = group["frame"].to_numpy()
        fps_arr = np.nan_to_num(group["fps"].to_numpy(), nan=30.0)
        fps_arr = np.where(fps_arr > 0, fps_arr, 30.0)
        h_arr = np.maximum(group["frame_height"].to_numpy().astype(np.float64), 1.0)

        bx1 = np.nan_to_num(group["bbox_x1"].to_numpy(), nan=0.0)
        by1 = np.nan_to_num(group["bbox_y1"].to_numpy(), nan=0.0)
        bx2 = np.nan_to_num(group["bbox_x2"].to_numpy(), nan=0.0)
        by2 = np.nan_to_num(group["bbox_y2"].to_numpy(), nan=0.0)
        bh_arr = np.maximum(by2 - by1, 1e-3)
        bw_arr = np.maximum(bx2 - bx1, 1e-3)

        sh_x = np.nan_to_num(group["avg_shoulder_x"].to_numpy(), nan=0.0)
        sh_y = np.nan_to_num(group["avg_shoulder_y"].to_numpy(), nan=0.0)
        hip_x = np.nan_to_num(group["avg_hip_x"].to_numpy(), nan=sh_x)
        hip_y = np.nan_to_num(group["avg_hip_y"].to_numpy(), nan=sh_y + 0.22 * h_arr)
        l_hc = np.nan_to_num(group["left_hip_conf"].to_numpy(), nan=0.0)
        r_hc = np.nan_to_num(group["right_hip_conf"].to_numpy(), nan=0.0)
        has_hips = (l_hc >= 0.12) & (r_hc >= 0.12)

        l_sh_c = np.nan_to_num(group["left_shoulder_conf"].to_numpy(), nan=0.0)
        r_sh_c = np.nan_to_num(group["right_shoulder_conf"].to_numpy(), nan=0.0)
        l_sh_y = np.nan_to_num(group["left_shoulder_y"].to_numpy(), nan=sh_y)
        r_sh_y = np.nan_to_num(group["right_shoulder_y"].to_numpy(), nan=sh_y)
        sh_w = np.nan_to_num(group["shoulder_width_px"].to_numpy(), nan=0.0)
        sp_len = np.nan_to_num(group["spine_length_px"].to_numpy(), nan=0.0)

        m = len(idxs)
        scale_s = np.zeros(m, dtype=np.float64)
        for k in range(m):
            if l_sh_c[k] >= 0.25 and r_sh_c[k] >= 0.25 and sh_w[k] > 0.0:
                emp_mem["sh_scale_win"].append(float(sh_w[k]))
            med_sh = float(np.median(emp_mem["sh_scale_win"])) if emp_mem["sh_scale_win"] else float(sh_w[k])
            scale_s[k] = max(med_sh, float(sh_w[k]), 0.68 * float(sp_len[k]), 0.22 * float(bh_arr[k]), 0.30 * float(bw_arr[k]), 1e-3)

        torso_x = np.where(has_hips, 0.5 * (sh_x + hip_x), sh_x)
        torso_y = np.where(has_hips, 0.5 * (sh_y + hip_y), sh_y)
        horiz_v_sw_s = np.zeros(m, dtype=np.float64)
        horiz_span_sw = np.zeros(m, dtype=np.float64)
        t2d_v_sw_s = np.zeros(m, dtype=np.float64)
        t2d_span_sw = np.zeros(m, dtype=np.float64)

        for k in range(m):
            emp_mem["torso_x_win"].append((int(frames[k]), float(torso_x[k]), float(torso_y[k])))
            tx_vals = [p[1] for p in emp_mem["torso_x_win"]]
            ty_vals = [p[2] for p in emp_mem["torso_x_win"]]
            horiz_span_sw[k] = (max(tx_vals) - min(tx_vals)) / scale_s[k]
            t2d_span_sw[k] = float(np.hypot(max(tx_vals) - min(tx_vals), max(ty_vals) - min(ty_vals))) / scale_s[k]
            if len(emp_mem["torso_x_win"]) >= 2:
                k_back = min(15, len(emp_mem["torso_x_win"]) - 1)
                f_old, tx_old, ty_old = emp_mem["torso_x_win"][-1 - k_back]
                f_span = int(frames[k]) - f_old
                if 1 <= f_span <= k_back * max_frame_gap:
                    horiz_v_sw_s[k] = ((abs(float(torso_x[k]) - tx_old) / scale_s[k]) / float(f_span)) * fps_arr[k]
                    t2d_v_sw_s[k] = (
                        (float(np.hypot(float(torso_x[k]) - tx_old, float(torso_y[k]) - ty_old)) / scale_s[k])
                        / float(f_span)
                    ) * fps_arr[k]
                else:
                    f_prev, tx_prev, ty_prev = emp_mem["torso_x_win"][-2]
                    f_step = int(frames[k]) - f_prev
                    if 1 <= f_step <= max_frame_gap:
                        horiz_v_sw_s[k] = ((abs(float(torso_x[k]) - tx_prev) / scale_s[k]) / float(f_step)) * fps_arr[k]
                        t2d_v_sw_s[k] = (
                            (float(np.hypot(float(torso_x[k]) - tx_prev, float(torso_y[k]) - ty_prev)) / scale_s[k])
                            / float(f_step)
                        ) * fps_arr[k]
        if m > 1 and horiz_v_sw_s[0] == 0.0:
            horiz_v_sw_s[0] = horiz_v_sw_s[1]
            t2d_v_sw_s[0] = t2d_v_sw_s[1]

        lwx = np.nan_to_num(group["left_wrist_x"].to_numpy(), nan=0.0)
        lwy = np.nan_to_num(group["left_wrist_y"].to_numpy(), nan=0.0)
        lwc = np.nan_to_num(group["left_wrist_conf"].to_numpy(), nan=0.0)

        rwx = np.nan_to_num(group["right_wrist_x"].to_numpy(), nan=0.0)
        rwy = np.nan_to_num(group["right_wrist_y"].to_numpy(), nan=0.0)
        rwc = np.nan_to_num(group["right_wrist_conf"].to_numpy(), nan=0.0)

        walk_blend = np.clip((np.maximum(horiz_v_sw_s, t2d_v_sw_s) - 0.30) / 0.30, 0.0, 1.0)
        inst_lv_sw_s = np.zeros(m, dtype=np.float64)
        inst_rv_sw_s = np.zeros(m, dtype=np.float64)

        if m > 1:
            df_step = np.diff(frames)
            valid_gap = (df_step >= 1) & (df_step <= max_frame_gap)
            valid_l = valid_gap & (lwc[1:] >= min_wrist_conf) & (lwc[:-1] >= min_wrist_conf)
            valid_r = valid_gap & (rwc[1:] >= min_wrist_conf) & (rwc[:-1] >= min_wrist_conf)

            dl_raw_sw = np.hypot(np.diff(lwx), np.diff(lwy)) / scale_s[1:]
            dr_raw_sw = np.hypot(np.diff(rwx), np.diff(rwy)) / scale_s[1:]

            dl_rel_sw = np.hypot(np.diff((lwx - sh_x) / scale_s), np.diff((lwy - sh_y) / scale_s))
            dr_rel_sw = np.hypot(np.diff((rwx - sh_x) / scale_s), np.diff((rwy - sh_y) / scale_s))

            wb = walk_blend[1:]
            dl_sw = (1.0 - wb) * dl_raw_sw + wb * dl_rel_sw
            dr_sw = (1.0 - wb) * dr_raw_sw + wb * dr_rel_sw

            inst_lv_sw_s[1:] = np.where(valid_l, (dl_sw / np.maximum(df_step, 1)) * fps_arr[1:], 0.0)
            inst_rv_sw_s[1:] = np.where(valid_r, (dr_sw / np.maximum(df_step, 1)) * fps_arr[1:], 0.0)
            inst_lv_sw_s[0] = inst_lv_sw_s[1]
            inst_rv_sw_s[0] = inst_rv_sw_s[1]

        has_px_cols = "wrist_pixel_motion_ratio" in group.columns and "lk_wrist_flow_sw_s" in group.columns
        if has_px_cols:
            px_mot_raw = group["wrist_pixel_motion_ratio"].to_numpy()
            lk_flow_raw = group["lk_wrist_flow_sw_s"].to_numpy()
            valid_px = np.isfinite(px_mot_raw) & np.isfinite(lk_flow_raw)
            if np.any(valid_px):
                px_mot_c = np.nan_to_num(px_mot_raw, nan=0.0)
                lk_flow_c = np.nan_to_num(lk_flow_raw, nan=0.0)
                pixel_gate = np.where(
                    valid_px,
                    np.clip(np.maximum(px_mot_c / 0.026, lk_flow_c / 0.13), 0.0, 1.0),
                    1.0,
                )
                # First frame of video has no prior frame diff -> retain raw step
                if m > 1 and px_mot_c[0] == 0.0 and lk_flow_c[0] == 0.0:
                    pixel_gate[0] = pixel_gate[1]
                inst_lv_sw_s = inst_lv_sw_s * pixel_gate
                inst_rv_sw_s = inst_rv_sw_s * pixel_gate
                px_boost = np.maximum(0.85 * lk_flow_c, np.where(px_mot_c >= 0.042, 3.0 * px_mot_c, 0.0))
                inst_dom_sw_s = np.maximum(
                    np.maximum(inst_lv_sw_s, inst_rv_sw_s),
                    np.where(valid_px, pixel_gate * px_boost, 0.0),
                )
            else:
                inst_dom_sw_s = np.maximum(inst_lv_sw_s, inst_rv_sw_s)
        else:
            inst_dom_sw_s = np.maximum(inst_lv_sw_s, inst_rv_sw_s)

        nx = np.nan_to_num(group["nose_x"].to_numpy(), nan=0.0)
        ny = np.nan_to_num(group["nose_y"].to_numpy(), nan=0.0)
        nc = np.nan_to_num(group["nose_conf"].to_numpy(), nan=0.0)

        rl_sw = np.zeros(m, dtype=np.float64)
        rr_sw = np.zeros(m, dtype=np.float64)
        rd_sw = np.zeros(m, dtype=np.float64)
        lw_rel_std = np.zeros(m, dtype=np.float64)
        rw_rel_std = np.zeros(m, dtype=np.float64)
        lw_short_std = np.zeros(m, dtype=np.float64)
        rw_short_std = np.zeros(m, dtype=np.float64)
        w_var_sw = np.zeros(m, dtype=np.float64)
        gaze_std = np.full(m, np.nan, dtype=np.float64)

        for k in range(m):
            emp_mem["lw_vel_win"].append(float(inst_lv_sw_s[k]))
            emp_mem["rw_vel_win"].append(float(inst_rv_sw_s[k]))
            emp_mem["dom_vel_win"].append(float(inst_dom_sw_s[k]))

            lw_rx = float((lwx[k] - sh_x[k]) / scale_s[k])
            lw_ry = float((lwy[k] - sh_y[k]) / scale_s[k])
            rw_rx = float((rwx[k] - sh_x[k]) / scale_s[k])
            rw_ry = float((rwy[k] - sh_y[k]) / scale_s[k])

            emp_mem["lw_rel_win"].append((lw_rx, lw_ry))
            emp_mem["rw_rel_win"].append((rw_rx, rw_ry))
            emp_mem["lw_short_win"].append((lw_rx, lw_ry))
            emp_mem["rw_short_win"].append((rw_rx, rw_ry))

            rl_sw[k] = float(np.mean(emp_mem["lw_vel_win"]))
            rr_sw[k] = float(np.mean(emp_mem["rw_vel_win"]))
            rd_sw[k] = float(np.mean(emp_mem["dom_vel_win"]))

            if len(emp_mem["lw_rel_win"]) >= 2:
                lx_a = np.array([p[0] for p in emp_mem["lw_rel_win"]], dtype=np.float64)
                ly_a = np.array([p[1] for p in emp_mem["lw_rel_win"]], dtype=np.float64)
                rx_a = np.array([p[0] for p in emp_mem["rw_rel_win"]], dtype=np.float64)
                ry_a = np.array([p[1] for p in emp_mem["rw_rel_win"]], dtype=np.float64)
                l_v = float(np.var(lx_a) + np.var(ly_a))
                r_v = float(np.var(rx_a) + np.var(ry_a))
                lw_rel_std[k] = float(np.sqrt(l_v))
                rw_rel_std[k] = float(np.sqrt(r_v))
                w_var_sw[k] = max(l_v, r_v)

            if len(emp_mem["lw_short_win"]) >= 2:
                lsx = np.array([p[0] for p in emp_mem["lw_short_win"]], dtype=np.float64)
                lsy = np.array([p[1] for p in emp_mem["lw_short_win"]], dtype=np.float64)
                rsx = np.array([p[0] for p in emp_mem["rw_short_win"]], dtype=np.float64)
                rsy = np.array([p[1] for p in emp_mem["rw_short_win"]], dtype=np.float64)
                lw_short_std[k] = float(np.sqrt(np.var(lsx) + np.var(lsy)))
                rw_short_std[k] = float(np.sqrt(np.var(rsx) + np.var(rsy)))

            if nc[k] >= 0.18:
                emp_mem["nose_rel_win"].append(
                    (float((nx[k] - sh_x[k]) / scale_s[k]), float((ny[k] - sh_y[k]) / scale_s[k]))
                )
                if len(emp_mem["nose_rel_win"]) >= 2:
                    nxa = np.array([p[0] for p in emp_mem["nose_rel_win"]], dtype=np.float64)
                    nya = np.array([p[1] for p in emp_mem["nose_rel_win"]], dtype=np.float64)
                    gaze_std[k] = float(np.sqrt(np.var(nxa) + np.var(nya)))
                else:
                    gaze_std[k] = 0.0

        bi_ratio = np.minimum(rl_sw, rr_sw) / np.maximum(np.maximum(rl_sw, rr_sw), 1e-4)
        dom_rel_std = np.where(rl_sw >= rr_sw, lw_rel_std, rw_rel_std)

        lex = np.nan_to_num(group["left_eye_x"].to_numpy(), nan=0.0)
        ley = np.nan_to_num(group["left_eye_y"].to_numpy(), nan=0.0)
        lec = np.nan_to_num(group["left_eye_conf"].to_numpy(), nan=0.0)

        rex = np.nan_to_num(group["right_eye_x"].to_numpy(), nan=0.0)
        rey = np.nan_to_num(group["right_eye_y"].to_numpy(), nan=0.0)
        rec = np.nan_to_num(group["right_eye_conf"].to_numpy(), nan=0.0)

        leax = np.nan_to_num(group["left_ear_x"].to_numpy(), nan=0.0)
        leay = np.nan_to_num(group["left_ear_y"].to_numpy(), nan=0.0)
        leac = np.nan_to_num(group["left_ear_conf"].to_numpy(), nan=0.0)

        reax = np.nan_to_num(group["right_ear_x"].to_numpy(), nan=0.0)
        reay = np.nan_to_num(group["right_ear_y"].to_numpy(), nan=0.0)
        reac = np.nan_to_num(group["right_ear_conf"].to_numpy(), nan=0.0)

        inf_arr = np.full(m, 99.0, dtype=np.float64)
        d_lw_lear = np.where((lwc >= 0.12) & (leac >= 0.12), np.hypot(lwx - leax, lwy - leay) / scale_s, inf_arr)
        d_lw_rear = np.where((lwc >= 0.12) & (reac >= 0.12), np.hypot(lwx - reax, lwy - reay) / scale_s, inf_arr)
        d_rw_rear = np.where((rwc >= 0.12) & (reac >= 0.12), np.hypot(rwx - reax, rwy - reay) / scale_s, inf_arr)
        d_rw_lear = np.where((rwc >= 0.12) & (leac >= 0.12), np.hypot(rwx - leax, rwy - leay) / scale_s, inf_arr)

        d_lw_head = np.minimum(
            np.minimum(d_lw_lear, d_lw_rear),
            np.where((lwc >= 0.12) & (nc >= 0.15), 1.15 * np.hypot(lwx - nx, lwy - ny) / scale_s, inf_arr),
        )
        d_rw_head = np.minimum(
            np.minimum(d_rw_rear, d_rw_lear),
            np.where((rwc >= 0.12) & (nc >= 0.15), 1.15 * np.hypot(rwx - nx, rwy - ny) / scale_s, inf_arr),
        )
        min_ear_d = np.minimum(d_lw_head, d_rw_head)
        ear_w_std = np.where(d_lw_head <= d_rw_head, lw_short_std, rw_short_std)

        mouth_x = nx
        mouth_y = ny + 0.16 * scale_s
        d_lw_face = np.where((lwc >= 0.12) & (nc >= 0.15), np.hypot(lwx - mouth_x, lwy - mouth_y) / scale_s, inf_arr)
        d_rw_face = np.where((rwc >= 0.12) & (nc >= 0.15), np.hypot(rwx - mouth_x, rwy - mouth_y) / scale_s, inf_arr)
        min_face_d = np.minimum(d_lw_face, d_rw_face)

        lw_y_rel = (lwy - sh_y) / scale_s
        rw_y_rel = (rwy - sh_y) / scale_s
        lw_ptp = _sliding_ptp_1d(lw_y_rel, half_window=max(window, 30))
        rw_ptp = _sliding_ptp_1d(rw_y_rel, half_window=max(window, 30))
        face_vert_ptp = np.where(d_lw_face <= d_rw_face, lw_ptp, rw_ptp)

        has_eyes = (lec >= 0.15) & (rec >= 0.15)
        has_ears = (leac >= 0.15) & (reac >= 0.15)
        eye_mid_x = np.where(has_eyes, 0.5 * (lex + rex), sh_x)
        eye_mid_y = np.where(has_eyes, 0.5 * (ley + rey), np.where(has_ears, 0.5 * (leay + reay), ny - 0.12 * scale_s))

        pitch_r = np.where(nc >= 0.18, (ny - eye_mid_y) / scale_s, np.nan)
        fwd_sym = np.where(nc >= 0.18, np.abs(nx - eye_mid_x) / scale_s, np.nan)

        sp_ratio = sp_len / scale_s
        valid_sp = sp_ratio[np.isfinite(sp_ratio) & has_hips & (horiz_v_sw_s < 0.35)]
        if len(valid_sp) >= 10:
            upright_ref = float(np.percentile(valid_sp, 75.0))
        elif np.any(np.isfinite(sp_ratio)):
            upright_ref = float(np.nanmedian(sp_ratio))
        else:
            upright_ref = 1.0
        comp_ratio = np.where(has_hips & np.isfinite(sp_ratio), sp_ratio / max(upright_ref, 1e-3), 1.0)

        stretch_flag = (
            (lwc >= 0.15)
            & (rwc >= 0.15)
            & (lwy < l_sh_y - 0.04 * scale_s)
            & (rwy < r_sh_y - 0.04 * scale_s)
        ).astype(np.int64)

        bio_scale_arr[idxs] = scale_s
        spine_to_sh_arr[idxs] = sp_ratio
        bbox_to_sh_arr[idxs] = bh_arr / scale_s
        bbox_aspect_arr[idxs] = bh_arr / np.maximum(bw_arr, 1e-3)

        lw_vel_sw_s[idxs] = inst_lv_sw_s
        rw_vel_sw_s[idxs] = inst_rv_sw_s
        roll_lw_sw_s[idxs] = rl_sw
        roll_rw_sw_s[idxs] = rr_sw
        roll_wrist_sw_s[idxs] = rd_sw
        wrist_var_30f_sw[idxs] = w_var_sw
        torso_horiz_vel_sw_s[idxs] = horiz_v_sw_s
        torso_horiz_span_sw[idxs] = horiz_span_sw
        torso_2d_vel_sw_s[idxs] = t2d_v_sw_s
        torso_2d_span_sw[idxs] = t2d_span_sw

        lw_vel_px_s[idxs] = inst_lv_sw_s * scale_s
        rw_vel_px_s[idxs] = inst_rv_sw_s * scale_s
        roll_lw_px_s[idxs] = rl_sw * scale_s
        roll_rw_px_s[idxs] = rr_sw * scale_s
        roll_wrist_px_s[idxs] = rd_sw * scale_s
        roll_wrist_px_frame[idxs] = (rd_sw * scale_s) / fps_arr
        roll_wrist_norm_s[idxs] = (rd_sw * scale_s) / h_arr
        torso_horiz_vel_px_s[idxs] = horiz_v_sw_s * scale_s
        torso_horiz_span_24f_px[idxs] = horiz_span_sw * scale_s

        bilateral_wrist_ratio[idxs] = bi_ratio
        dom_wrist_rel_std_norm[idxs] = dom_rel_std
        min_wrist_to_ear_norm[idxs] = np.where(min_ear_d < 50.0, min_ear_d, np.nan)
        ear_wrist_rel_std_norm[idxs] = ear_w_std
        min_wrist_to_face_norm[idxs] = np.where(min_face_d < 50.0, min_face_d, np.nan)
        face_wrist_vert_ptp_norm[idxs] = face_vert_ptp
        head_pitch_ratio[idxs] = pitch_r
        head_forward_sym_norm[idxs] = fwd_sym
        head_gaze_std_norm[idxs] = gaze_std
        torso_compression_ratio[idxs] = comp_ratio
        both_wrists_above_sh[idxs] = stretch_flag

    enriched = (
        df_geom.with_columns(
            [
                pl.Series("bio_shoulder_scale_px", bio_scale_arr).fill_nan(None),
                pl.Series("spine_to_shoulder_ratio", spine_to_sh_arr).fill_nan(None),
                pl.Series("bbox_to_shoulder_ratio", bbox_to_sh_arr).fill_nan(None),
                pl.Series("bbox_aspect_h_w", bbox_aspect_arr).fill_nan(None),
                pl.Series("left_wrist_vel_sw_s", lw_vel_sw_s).fill_nan(None),
                pl.Series("right_wrist_vel_sw_s", rw_vel_sw_s).fill_nan(None),
                pl.Series("rolling_30f_left_wrist_vel_sw_s", roll_lw_sw_s).fill_nan(None),
                pl.Series("rolling_30f_right_wrist_vel_sw_s", roll_rw_sw_s).fill_nan(None),
                pl.Series("rolling_30f_wrist_vel_sw_s", roll_wrist_sw_s).fill_nan(None),
                pl.Series("wrist_spatial_var_30f_sw", wrist_var_30f_sw).fill_nan(None),
                pl.Series("torso_horiz_vel_sw_s", torso_horiz_vel_sw_s).fill_nan(None),
                pl.Series("torso_horiz_span_30f_sw", torso_horiz_span_sw).fill_nan(None),
                pl.Series("torso_2d_vel_sw_s", torso_2d_vel_sw_s).fill_nan(None),
                pl.Series("torso_2d_span_30f_sw", torso_2d_span_sw).fill_nan(None),
                pl.Series("left_wrist_vel_px_s", lw_vel_px_s).fill_nan(None),
                pl.Series("right_wrist_vel_px_s", rw_vel_px_s).fill_nan(None),
                pl.Series("rolling_30f_left_wrist_vel_px_s", roll_lw_px_s).fill_nan(None),
                pl.Series("rolling_30f_right_wrist_vel_px_s", roll_rw_px_s).fill_nan(None),
                pl.Series("rolling_30f_wrist_vel_px_s", roll_wrist_px_s).fill_nan(None),
                pl.Series("rolling_30f_wrist_vel_px_frame", roll_wrist_px_frame).fill_nan(None),
                pl.Series("rolling_30f_wrist_vel_norm_s", roll_wrist_norm_s).fill_nan(None),
                pl.Series("torso_horiz_vel_px_s", torso_horiz_vel_px_s).fill_nan(None),
                pl.Series("torso_horiz_span_24f_px", torso_horiz_span_24f_px).fill_nan(None),
                pl.Series("bilateral_wrist_ratio", bilateral_wrist_ratio).fill_nan(None),
                pl.Series("dom_wrist_rel_std_norm", dom_wrist_rel_std_norm).fill_nan(None),
                pl.Series("min_wrist_to_ear_norm", min_wrist_to_ear_norm).fill_nan(None),
                pl.Series("ear_wrist_rel_std_norm", ear_wrist_rel_std_norm).fill_nan(None),
                pl.Series("min_wrist_to_face_norm", min_wrist_to_face_norm).fill_nan(None),
                pl.Series("face_wrist_vert_ptp_norm", face_wrist_vert_ptp_norm).fill_nan(None),
                pl.Series("head_pitch_ratio", head_pitch_ratio).fill_nan(None),
                pl.Series("head_forward_sym_norm", head_forward_sym_norm).fill_nan(None),
                pl.Series("head_gaze_std_norm", head_gaze_std_norm).fill_nan(None),
                pl.Series("torso_compression_ratio", torso_compression_ratio).fill_nan(None),
                pl.Series("both_wrists_above_shoulders", both_wrists_above_sh),
            ]
        )
        .sort(["video_name", "frame", "track_id"], nulls_last=True)
        .drop("_orig_row_idx")
    )
    return enriched


def load_and_extract_features(
    csv_path: Union[str, Path],
    wrist_window_frames: int = 30,
) -> pl.DataFrame:
    """Load a Milestone 1 landmark CSV and compute all scale-invariant features."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Landmark CSV not found: {path}")
    raw_df = pl.read_csv(path, infer_schema_length=20000)
    return compute_temporal_features(raw_df, wrist_window_frames=wrist_window_frames)
