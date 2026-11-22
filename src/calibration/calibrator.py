"""Scale-invariant mathematical threshold calibration for Posture (Sitting vs Standing)
and Activity (Working vs Idle).

All classification cutoffs are calibrated in dimensionless coordinates:
  - Spatial posture in normalized vertical elevation `(H - Avg_Shoulder_Y) / H` in [0, 1]
    and normalized horizontal camera zone `Avg_Shoulder_X / W` in [0, 1].
  - Wrist velocity in biological shoulder-widths per second (`SW/s`), dividing Euclidean
    wrist movement by each employee's dynamic inter-shoulder breadth `S_i`.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import src.config  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


@dataclass
class ZonePostureThreshold:
    """Calibrated Sitting-to-Standing normalized threshold for a camera perspective zone."""
    zone_name: str
    x_min_norm: float
    x_max_norm: float
    threshold_norm: float
    enter_standing_norm: float
    exit_to_sitting_norm: float
    seated_median_norm: float
    standing_median_norm: float
    separability_eta: float
    sample_count: int
    # Derived display metadata (scaled by frame dimensions solely for chart/report labels)
    x_min_px: float = 0.0
    x_max_px: float = 1.0
    threshold_px: float = 0.0
    enter_standing_px: float = 0.0
    exit_to_sitting_px: float = 0.0
    seated_median_px: float = 0.0
    standing_median_px: float = 0.0


@dataclass
class CalibratedThresholds:
    """Complete scale-invariant calibration profile exported to JSON."""
    frame_width: int
    frame_height: int
    fps: float
    wrist_window_frames: int
    min_dwell_frames: int
    # Scale-invariant posture metrics (dimensionless [0, 1])
    global_posture_threshold_norm: float
    global_posture_separability_eta: float
    use_perspective_zones: bool
    perspective_split_x_norm: Optional[float]
    posture_zones: Dict[str, ZonePostureThreshold] = field(default_factory=dict)
    # Scale-invariant wrist velocity metrics (in biological shoulder-widths per second, SW/s)
    wrist_velocity_threshold_sw_s: float = 0.24
    enter_working_vel_sw_s: float = 0.26
    exit_to_idle_vel_sw_s: float = 0.22
    idle_mode_peak_sw_s: float = 0.05
    active_mode_peak_sw_s: float = 0.65
    velocity_separability_eta: float = 0.0
    # Scale-invariant locomotion & relational thresholds (in SW/s and SW)
    walking_horiz_vel_threshold_sw_s: float = 0.68
    idle_confirm_seconds: float = 5.0
    bilateral_typing_ratio_threshold: float = 0.35
    phone_ear_dist_norm_threshold: float = 0.58
    drinking_face_dist_norm_threshold: float = 0.48
    slouch_spine_angle_deg_threshold: float = 15.0
    # Derived display metadata for UI/CLI reporting
    global_posture_threshold_px: float = 0.0
    perspective_split_x_px: Optional[float] = None
    wrist_velocity_threshold_px_s: float = 0.0
    wrist_velocity_threshold_px_frame: float = 0.0
    wrist_velocity_threshold_norm_s: float = 0.0
    enter_working_vel_px_s: float = 0.0
    exit_to_idle_vel_px_s: float = 0.0
    idle_mode_peak_px_s: float = 0.0
    active_mode_peak_px_s: float = 0.0
    walking_horiz_vel_threshold_px_s: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "CalibratedThresholds":
        with open(path, "r") as f:
            raw = json.load(f)
        zones_raw = raw.pop("posture_zones", {})
        zones = {k: ZonePostureThreshold(**v) for k, v in zones_raw.items()}
        return cls(**raw, posture_zones=zones)


def otsu_threshold_1d(values: np.ndarray, num_bins: int = 256) -> Tuple[float, float]:
    """Compute exact 1D Otsu threshold maximizing between-class variance."""
    clean = values[np.isfinite(values)]
    if len(clean) < 4:
        val = float(np.median(clean)) if len(clean) > 0 else 0.0
        return val, 0.0

    vmin, vmax = float(np.min(clean)), float(np.max(clean))
    if vmax - vmin < 1e-6:
        return vmin, 0.0

    hist, bin_edges = np.histogram(clean, bins=num_bins, range=(vmin, vmax))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    probs = hist.astype(np.float64) / float(np.sum(hist))

    omega0 = np.cumsum(probs)
    omega1 = 1.0 - omega0
    mu_cum = np.cumsum(probs * bin_centers)
    mu_total = mu_cum[-1]

    total_var = float(np.sum(probs * (bin_centers - mu_total) ** 2))
    if total_var <= 1e-12:
        return float(np.median(clean)), 0.0

    valid = (omega0 > 1e-6) & (omega1 > 1e-6)
    sigma_b2 = np.zeros_like(omega0)
    mu0 = np.where(valid, mu_cum / np.maximum(omega0, 1e-12), 0.0)
    mu1 = np.where(valid, (mu_total - mu_cum) / np.maximum(omega1, 1e-12), 0.0)
    sigma_b2[valid] = omega0[valid] * omega1[valid] * (mu0[valid] - mu1[valid]) ** 2

    max_sigma = float(np.max(sigma_b2))
    plateau_mask = sigma_b2 >= max_sigma * (1.0 - 1e-9)
    best_thresh = float(np.mean(bin_centers[plateau_mask]))
    eta = float(min(max(max_sigma / total_var, 0.0), 1.0))
    return best_thresh, eta


def kde_valley_or_otsu_1d(
    values: np.ndarray,
    num_grid: int = 512,
    prefer_upper_valley: bool = False,
) -> Tuple[float, float, float, float]:
    """Locate bimodal KDE valley between primary modes, falling back to Otsu if unimodal."""
    clean = values[np.isfinite(values)]
    otsu_t, eta = otsu_threshold_1d(clean)
    if len(clean) < 30 or (np.max(clean) - np.min(clean)) < 1e-5:
        low_med = float(np.median(clean[clean <= otsu_t])) if np.any(clean <= otsu_t) else otsu_t
        high_med = float(np.median(clean[clean > otsu_t])) if np.any(clean > otsu_t) else otsu_t
        return otsu_t, low_med, high_med, eta

    q01, q99 = np.percentile(clean, [1.0, 99.0])
    if q99 - q01 < 1e-5:
        q01, q99 = float(np.min(clean)), float(np.max(clean))

    grid = np.linspace(q01, q99, num_grid)
    try:
        kde = gaussian_kde(clean, bw_method="scott")
        density = kde(grid)
        peaks, _ = find_peaks(density, prominence=0.06 * np.max(density), distance=max(num_grid // 5, 20))
        if len(peaks) >= 2:
            p_low = int(peaks[0])
            upper_candidates = peaks[1:]
            p_high = int(upper_candidates[np.argmax(density[upper_candidates])])
            if p_high > p_low + 4:
                if prefer_upper_valley:
                    valleys, _ = find_peaks(-density[p_low : p_high + 1])
                    valleys = p_low + valleys
                    low_cap = 0.35 * min(float(density[p_low]), float(density[p_high]))
                    valid_v = [int(v) for v in valleys if float(density[v]) <= low_cap]
                    if valid_v:
                        valley_idx = valid_v[-1]
                        return float(grid[valley_idx]), float(grid[p_low]), float(grid[p_high]), eta

                margin = max(int(0.15 * (p_high - p_low)), 1)
                s_lo = p_low + margin
                s_hi = max(p_high - margin, s_lo + 1)
                valley_rel = int(np.argmin(density[s_lo : s_hi + 1]))
                valley_idx = s_lo + valley_rel
                return float(grid[valley_idx]), float(grid[p_low]), float(grid[p_high]), eta
    except Exception:
        pass

    low_vals = clean[clean <= otsu_t]
    high_vals = clean[clean > otsu_t]
    low_med = float(np.median(low_vals)) if len(low_vals) > 0 else otsu_t
    high_med = float(np.median(high_vals)) if len(high_vals) > 0 else otsu_t
    return otsu_t, low_med, high_med, eta


def _calibrate_single_zone_posture_norm(
    dist_norm: np.ndarray,
    zone_name: str,
    x_min_norm: float,
    x_max_norm: float,
    frame_width: float,
    frame_height: float,
    hysteresis_frac: float = 0.04,
) -> ZonePostureThreshold:
    """Calibrate Sitting vs Standing purely in normalized [0, 1] elevation space."""
    clean = dist_norm[np.isfinite(dist_norm)]
    n = len(clean)
    fw = max(float(frame_width), 1.0)
    fh = max(float(frame_height), 1.0)

    if n == 0:
        default_n = 0.30
        return ZonePostureThreshold(
            zone_name=zone_name,
            x_min_norm=round(x_min_norm, 4),
            x_max_norm=round(x_max_norm, 4),
            threshold_norm=default_n,
            enter_standing_norm=round(default_n * (1.0 + hysteresis_frac), 4),
            exit_to_sitting_norm=round(default_n * (1.0 - hysteresis_frac), 4),
            seated_median_norm=round(default_n * 0.7, 4),
            standing_median_norm=round(default_n * 1.3, 4),
            separability_eta=0.0,
            sample_count=0,
            x_min_px=round(x_min_norm * fw, 1),
            x_max_px=round(x_max_norm * fw, 1),
            threshold_px=round(default_n * fh, 2),
            enter_standing_px=round(default_n * (1.0 + hysteresis_frac) * fh, 2),
            exit_to_sitting_px=round(default_n * (1.0 - hysteresis_frac) * fh, 2),
            seated_median_px=round(default_n * 0.7 * fh, 2),
            standing_median_px=round(default_n * 1.3 * fh, 2),
        )

    thresh_n, low_peak, high_peak, eta = kde_valley_or_otsu_1d(clean, prefer_upper_valley=True)
    p10, p50, p95 = np.percentile(clean, [10.0, 50.0, 95.0])
    spread_95_10 = float(p95 - p10)

    # Guard unimodal zone in dimensionless elevation fraction (> 0.055 normalized spread)
    min_excursion = 0.055
    if spread_95_10 < min_excursion or (high_peak - low_peak) < min_excursion:
        mad = float(np.median(np.abs(clean - p50)))
        thresh_n = float(p50 + max(4.5 * mad, 0.065))
        seated_med_n = float(p50)
        standing_med_n = float(thresh_n + 0.06)
    else:
        seated_vals = clean[clean <= thresh_n]
        standing_vals = clean[clean > thresh_n]
        seated_med_n = float(np.median(seated_vals)) if len(seated_vals) > 0 else float(low_peak)
        standing_med_n = float(np.median(standing_vals)) if len(standing_vals) > 0 else float(high_peak)
        min_safe_n = seated_med_n + 0.035
        if thresh_n < min_safe_n and standing_med_n > min_safe_n:
            thresh_n = 0.5 * (min_safe_n + standing_med_n)

    delta_n = thresh_n * hysteresis_frac
    enter_n = thresh_n + delta_n
    exit_n = thresh_n - delta_n

    return ZonePostureThreshold(
        zone_name=zone_name,
        x_min_norm=round(float(x_min_norm), 4),
        x_max_norm=round(float(x_max_norm), 4),
        threshold_norm=round(float(thresh_n), 4),
        enter_standing_norm=round(float(enter_n), 4),
        exit_to_sitting_norm=round(float(exit_n), 4),
        seated_median_norm=round(float(seated_med_n), 4),
        standing_median_norm=round(float(standing_med_n), 4),
        separability_eta=round(float(eta), 4),
        sample_count=int(n),
        x_min_px=round(float(x_min_norm * fw), 1),
        x_max_px=round(float(x_max_norm * fw), 1),
        threshold_px=round(float(thresh_n * fh), 2),
        enter_standing_px=round(float(enter_n * fh), 2),
        exit_to_sitting_px=round(float(exit_n * fh), 2),
        seated_median_px=round(float(seated_med_n * fh), 2),
        standing_median_px=round(float(standing_med_n * fh), 2),
    )


def calibrate_thresholds_from_features(
    feature_df: pl.DataFrame,
    wrist_window_frames: int = 30,
    min_dwell_frames: int = 15,
    enable_perspective_zones: bool = True,
) -> CalibratedThresholds:
    """Calibrate dimensionless posture (`[0, 1]`) and wrist-velocity (`SW/s`) thresholds."""
    present = feature_df.filter(pl.col("presence") == 1)
    if present.is_empty():
        raise ValueError("Cannot calibrate thresholds: no 'presence == 1' rows found in telemetry.")

    frame_w = int(present["frame_width"].drop_nulls().median() or 1920)
    frame_h = int(present["frame_height"].drop_nulls().median() or 1080)
    fps = float(present["fps"].drop_nulls().median() or 30.0)
    fw_f = max(float(frame_w), 1.0)
    fh_f = max(float(frame_h), 1.0)

    dist_norm_all = present["shoulder_dist_bottom_norm"].drop_nulls().to_numpy()
    if "avg_shoulder_x_norm" in present.columns:
        sh_x_norm_all = present["avg_shoulder_x_norm"].drop_nulls().to_numpy()
    else:
        sh_x_norm_all = present["avg_shoulder_x"].drop_nulls().to_numpy() / fw_f

    global_zone = _calibrate_single_zone_posture_norm(
        dist_norm=dist_norm_all,
        zone_name="global",
        x_min_norm=0.0,
        x_max_norm=1.0,
        frame_width=fw_f,
        frame_height=fh_f,
    )

    zones: Dict[str, ZonePostureThreshold] = {"global": global_zone}
    use_zones = False
    split_x_norm: Optional[float] = None

    if enable_perspective_zones and len(sh_x_norm_all) >= 200 and len(sh_x_norm_all) == len(dist_norm_all):
        cand_split_n, eta_x = otsu_threshold_1d(sh_x_norm_all)
        left_mask = sh_x_norm_all <= cand_split_n
        right_mask = sh_x_norm_all > cand_split_n

        if eta_x >= 0.50 and int(np.sum(left_mask)) >= 100 and int(np.sum(right_mask)) >= 100:
            left_p25 = float(np.percentile(dist_norm_all[left_mask], 25.0))
            right_p25 = float(np.percentile(dist_norm_all[right_mask], 25.0))
            if abs(left_p25 - right_p25) >= 0.06:
                use_zones = True
                split_x_norm = round(float(cand_split_n), 4)
                left_name = "background" if left_p25 > right_p25 else "foreground"
                right_name = "foreground" if left_name == "background" else "background"

                zones[left_name] = _calibrate_single_zone_posture_norm(
                    dist_norm=dist_norm_all[left_mask],
                    zone_name=left_name,
                    x_min_norm=0.0,
                    x_max_norm=split_x_norm,
                    frame_width=fw_f,
                    frame_height=fh_f,
                )
                zones[right_name] = _calibrate_single_zone_posture_norm(
                    dist_norm=dist_norm_all[right_mask],
                    zone_name=right_name,
                    x_min_norm=split_x_norm,
                    x_max_norm=1.0,
                    frame_width=fw_f,
                    frame_height=fh_f,
                )

    # --- Action 2: Scale-Invariant Wrist Velocity Calibration in Shoulder-Widths per Second (SW/s) ---
    if "rolling_30f_wrist_vel_sw_s" in present.columns:
        vel_sw_col = present["rolling_30f_wrist_vel_sw_s"].to_numpy()
    else:
        bio_s_fallback = np.maximum(present["shoulder_width_px"].fill_null(0.10 * fh_f).to_numpy(), 1e-3)
        vel_sw_col = present["rolling_30f_wrist_vel_px_s"].to_numpy() / bio_s_fallback

    if use_zones and split_x_norm is not None and len(sh_x_norm_all) == len(present):
        seated_mask = np.zeros(len(present), dtype=bool)
        for z_name, z_cfg in zones.items():
            if z_name == "global":
                continue
            in_z = (sh_x_norm_all >= z_cfg.x_min_norm) & (sh_x_norm_all <= z_cfg.x_max_norm)
            seated_mask |= in_z & (dist_norm_all <= z_cfg.threshold_norm)
    else:
        seated_mask = dist_norm_all <= global_zone.threshold_norm

    seated_vel_mask = seated_mask & np.isfinite(vel_sw_col) & (vel_sw_col >= 0.0)
    if "torso_horiz_vel_sw_s" in present.columns:
        hv_sw = np.nan_to_num(present["torso_horiz_vel_sw_s"].to_numpy(), nan=0.0)
        seated_vel_mask &= hv_sw < 0.38

    seated_vel = vel_sw_col[seated_vel_mask]
    vel_pool = seated_vel if len(seated_vel) >= 30 else vel_sw_col[np.isfinite(vel_sw_col) & (vel_sw_col >= 0.0)]

    if len(vel_pool) >= 50:
        med_v = float(np.median(vel_pool))
        p90_v = float(np.percentile(vel_pool, 90.0))
        if med_v < 0.28:
            upper_cap = max(p90_v * 1.5, 0.40)
            trimmed = vel_pool[vel_pool <= upper_cap]
            if len(trimmed) >= 30:
                vel_pool = trimmed

    if len(vel_pool) < 10:
        vel_thresh_sw_s = 0.12
        idle_peak_sw_s = 0.04
        active_peak_sw_s = 0.35
        vel_eta = 0.0
    else:
        # Calibrate in dimensionless log10(v_sw + 0.05) space
        offset = 0.05
        log_vel = np.log10(vel_pool + offset)
        log_thresh, log_low, log_high, vel_eta = kde_valley_or_otsu_1d(log_vel)
        vel_thresh_sw_s = float(10.0 ** log_thresh - offset)
        idle_peak_sw_s = max(float(10.0 ** log_low - offset), 0.0)
        active_peak_sw_s = max(float(10.0 ** log_high - offset), vel_thresh_sw_s)
        # Biological micro-jitter floor in shoulder-widths/s (~2.5% of shoulder width/s)
        vel_thresh_sw_s = max(vel_thresh_sw_s, 0.025)

    enter_sw_s = vel_thresh_sw_s * 1.08
    exit_sw_s = vel_thresh_sw_s * 0.92
    walk_sw_s = 0.68

    # Representative biological shoulder scale ONLY for human-readable report display metadata
    if "bio_shoulder_scale_px" in present.columns:
        med_bio_px = float(present["bio_shoulder_scale_px"].drop_nulls().median() or (0.11 * fh_f))
    else:
        med_bio_px = float(present["shoulder_width_px"].drop_nulls().median() or (0.11 * fh_f))

    vel_px_s_disp = vel_thresh_sw_s * med_bio_px
    split_x_px_disp = round(split_x_norm * fw_f, 1) if split_x_norm is not None else None

    return CalibratedThresholds(
        frame_width=frame_w,
        frame_height=frame_h,
        fps=round(fps, 2),
        wrist_window_frames=wrist_window_frames,
        min_dwell_frames=min_dwell_frames,
        global_posture_threshold_norm=global_zone.threshold_norm,
        global_posture_separability_eta=global_zone.separability_eta,
        use_perspective_zones=use_zones,
        perspective_split_x_norm=split_x_norm,
        posture_zones=zones,
        wrist_velocity_threshold_sw_s=round(vel_thresh_sw_s, 4),
        enter_working_vel_sw_s=round(enter_sw_s, 4),
        exit_to_idle_vel_sw_s=round(exit_sw_s, 4),
        idle_mode_peak_sw_s=round(idle_peak_sw_s, 4),
        active_mode_peak_sw_s=round(active_peak_sw_s, 4),
        velocity_separability_eta=round(vel_eta, 4),
        walking_horiz_vel_threshold_sw_s=walk_sw_s,
        idle_confirm_seconds=5.0,
        bilateral_typing_ratio_threshold=0.35,
        phone_ear_dist_norm_threshold=0.58,
        drinking_face_dist_norm_threshold=0.48,
        slouch_spine_angle_deg_threshold=15.0,
        global_posture_threshold_px=global_zone.threshold_px,
        perspective_split_x_px=split_x_px_disp,
        wrist_velocity_threshold_px_s=round(vel_px_s_disp, 2),
        wrist_velocity_threshold_px_frame=round(vel_px_s_disp / max(fps, 1.0), 4),
        wrist_velocity_threshold_norm_s=round(vel_px_s_disp / fh_f, 5),
        enter_working_vel_px_s=round(enter_sw_s * med_bio_px, 2),
        exit_to_idle_vel_px_s=round(exit_sw_s * med_bio_px, 2),
        idle_mode_peak_px_s=round(idle_peak_sw_s * med_bio_px, 2),
        active_mode_peak_px_s=round(active_peak_sw_s * med_bio_px, 2),
        walking_horiz_vel_threshold_px_s=round(walk_sw_s * med_bio_px, 2),
    )


def generate_calibration_plots(
    classified_df: pl.DataFrame,
    thresholds: CalibratedThresholds,
    plots_dir: Path,
) -> Dict[str, Path]:
    """Render scale-invariant posture and wrist-velocity calibration plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    present = classified_df.filter(pl.col("presence") == 1)

    posture_plot_path = plots_dir / "milestone2_posture_calibration.png"
    velocity_plot_path = plots_dir / "milestone2_wrist_velocity_calibration.png"

    # --- Plot 1: Normalized Posture Calibration ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    if thresholds.use_perspective_zones and "camera_zone" in present.columns:
        zone_colors = {"foreground": "#2980b9", "background": "#8e44ad"}
        for z_name in ("foreground", "background"):
            if z_name not in thresholds.posture_zones:
                continue
            z_cfg = thresholds.posture_zones[z_name]
            sub_d = present.filter(pl.col("camera_zone") == z_name)["shoulder_dist_bottom_norm"].drop_nulls().to_numpy()
            if len(sub_d) > 0:
                ax1.hist(
                    sub_d,
                    bins=45,
                    alpha=0.45,
                    color=zone_colors.get(z_name, "#3498db"),
                    label=f"{z_name.capitalize()} Zone (n={len(sub_d)})",
                    density=True,
                )
                ax1.axvline(
                    z_cfg.threshold_norm,
                    color=zone_colors.get(z_name, "#3498db"),
                    linestyle="--",
                    linewidth=2.2,
                    label=f"{z_name.capitalize()} Cutoff: {z_cfg.threshold_norm:.3f} H",
                )
    else:
        dist_vals = present["shoulder_dist_bottom_norm"].drop_nulls().to_numpy()
        ax1.hist(dist_vals, bins=50, alpha=0.6, color="#2980b9", density=True, label="All Tracked Frames")
        ax1.axvline(
            thresholds.global_posture_threshold_norm,
            color="#e74c3c",
            linestyle="--",
            linewidth=2.2,
            label=f"Calibrated Cutoff: {thresholds.global_posture_threshold_norm:.3f} H",
        )

    ax1.set_title("Scale-Invariant Posture Calibration: (H - Avg_Shoulder_Y) / H")
    ax1.set_xlabel("Normalized Vertical Shoulder Elevation [0, 1]")
    ax1.set_ylabel("Probability Density")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    x_col = "avg_shoulder_x_norm" if "avg_shoulder_x_norm" in present.columns else "avg_shoulder_x"
    sit_df = present.filter(pl.col("posture_state") == "Sitting")
    stand_df = present.filter(pl.col("posture_state") == "Standing")
    walk_df = present.filter(pl.col("posture_state") == "Walking")

    if not sit_df.is_empty():
        ax2.scatter(sit_df[x_col].to_numpy(), sit_df["shoulder_dist_bottom_norm"].to_numpy(), s=8, alpha=0.35, color="#27ae60", label=f"Sitting (n={len(sit_df)})")
    if not stand_df.is_empty():
        ax2.scatter(stand_df[x_col].to_numpy(), stand_df["shoulder_dist_bottom_norm"].to_numpy(), s=12, alpha=0.55, color="#e67e22", label=f"Standing (n={len(stand_df)})")
    if not walk_df.is_empty():
        ax2.scatter(walk_df[x_col].to_numpy(), walk_df["shoulder_dist_bottom_norm"].to_numpy(), s=14, alpha=0.65, color="#8e44ad", label=f"Walking (n={len(walk_df)})")

    if thresholds.use_perspective_zones and thresholds.perspective_split_x_norm is not None:
        ax2.axvline(thresholds.perspective_split_x_norm, color="#555555", linestyle=":", linewidth=1.5, label="Zone Split X")
        for z_name, z_cfg in thresholds.posture_zones.items():
            if z_name == "global":
                continue
            ax2.hlines(z_cfg.threshold_norm, xmin=z_cfg.x_min_norm, xmax=z_cfg.x_max_norm, colors="#c0392b", linestyles="--", linewidth=2.2)
    else:
        ax2.axhline(thresholds.global_posture_threshold_norm, color="#c0392b", linestyle="--", linewidth=2.2)

    ax2.set_title("Normalized Sitting vs. Standing Decision Boundary")
    ax2.set_xlabel("Normalized Horizontal Position Avg_Shoulder_X / W")
    ax2.set_ylabel("Normalized Elevation (H - Avg_Shoulder_Y) / H")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(posture_plot_path, dpi=150)
    plt.close(fig)

    # --- Plot 2: Scale-Invariant Wrist Velocity Calibration (SW/s) ---
    fig, (vx1, vx2) = plt.subplots(1, 2, figsize=(14, 5.5))

    vel_sw_vals = present["rolling_30f_wrist_vel_sw_s"].drop_nulls().to_numpy()
    vel_clip = np.clip(vel_sw_vals, 0.0, np.percentile(vel_sw_vals, 99.0) if len(vel_sw_vals) > 10 else 3.0)

    offset = 0.05
    log_bins = np.logspace(np.log10(offset), np.log10(max(float(np.max(vel_clip)) + offset + 0.1, 1.0)), 55)
    vx1.hist(vel_clip + offset, bins=log_bins, color="#34495e", alpha=0.65, edgecolor="white")
    vx1.set_xscale("log")

    t_sw = thresholds.wrist_velocity_threshold_sw_s
    vx1.axvline(t_sw + offset, color="#e74c3c", linestyle="--", linewidth=2.3, label=f"Speed Cutoff: {t_sw:.3f} SW/s")
    vx1.axvspan(
        thresholds.exit_to_idle_vel_sw_s + offset,
        thresholds.enter_working_vel_sw_s + offset,
        color="#e74c3c",
        alpha=0.18,
        label=f"Hysteresis [{thresholds.exit_to_idle_vel_sw_s:.3f}, {thresholds.enter_working_vel_sw_s:.3f}] SW/s",
    )
    vx1.set_title(f"Rolling {thresholds.wrist_window_frames}f Wrist Speed (Shoulder-Widths / sec)")
    vx1.set_xlabel("Rolling Wrist Velocity + 0.05 (SW/s, log scale)")
    vx1.set_ylabel("Frame Count")
    vx1.legend(fontsize=8.5)
    vx1.grid(True, which="both", linestyle="--", alpha=0.35)

    top_tracks = present.group_by("track_id").len().sort("len", descending=True)["track_id"].head(2).to_list()
    palette = ["#2980b9", "#16a085"]
    for idx, tid in enumerate(top_tracks):
        t_sub = present.filter(pl.col("track_id") == tid).sort("frame")
        vx2.plot(
            t_sub["timestamp_sec"].to_numpy(),
            t_sub["rolling_30f_wrist_vel_sw_s"].to_numpy(),
            color=palette[idx % len(palette)],
            linewidth=1.3,
            alpha=0.85,
            label=f"Emp #{tid} Wrist Speed (SW/s)",
        )

    vx2.axhline(t_sw, color="#e74c3c", linestyle="--", linewidth=2.0, label=f"Working Cutoff ({t_sw:.3f} SW/s)")
    vx2.set_ylim(0, min(max(t_sw * 6.0, 1.2), 4.0))
    vx2.set_title("Scale-Invariant 30-Frame Wrist Velocity Timeline (SW/s)")
    vx2.set_xlabel("Video Timestamp (seconds)")
    vx2.set_ylabel("Wrist Velocity (Shoulder-Widths / sec)")
    vx2.legend(fontsize=8.5, loc="upper right")
    vx2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(velocity_plot_path, dpi=150)
    plt.close(fig)

    return {
        "posture_calibration_plot": posture_plot_path,
        "wrist_velocity_calibration_plot": velocity_plot_path,
    }
