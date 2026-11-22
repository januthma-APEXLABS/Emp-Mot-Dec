"""Executive reporting engine generating CSV/JSON KPI summaries, high-resolution dark-UI
dashboard snapshots (PNG), and multi-page executive PDF reports in `outputs/reports/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.config import REPORTS_OUTPUT_DIR, ensure_directories
from src.dashboard.data_loader import (
    POSTURE_ORDER,
    build_employee_summary_table,
    build_velocity_heatmap_data,
    compute_dashboard_kpis,
    extract_state_segments,
    format_duration_mmss,
    load_calibrated_thresholds,
    load_telemetry_dataframe,
)

DARK_BG = "#0B0E14"
CARD_BG = "#151922"
CARD_BORDER = "#252B37"
TEXT_MAIN = "#F0F6FC"
TEXT_SUB = "#94A3B8"

POSTURE_HEX = {
    "Sitting": "#23478E",
    "Standing": "#5D3FD3",
    "Walking": "#1F6E9C",
    "Away": "#2CA89A",
}
STATE_HEX = {
    **POSTURE_HEX,
    "Working": "#3B82F6",
    "Idle": "#8B5CF6",
}


def _style_dark_card(ax: plt.Axes, border_color: str = CARD_BORDER) -> None:
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(1.2)


def _draw_executive_hero_figure(df: pl.DataFrame, kpis: Dict[str, Any], title_text: str) -> plt.Figure:
    """Render the flagship dark-themed Executive Dashboard figure matching the target UI mockup
    plus Ergonomic Timeline and Activity Heatmap panels.
    """
    fig = plt.figure(figsize=(16.5, 11.2), facecolor=DARK_BG)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=5,
        height_ratios=[0.42, 2.15, 1.15, 1.15],
        hspace=0.36,
        wspace=0.28,
        left=0.105,
        right=0.965,
        top=0.955,
        bottom=0.055,
    )

    # --- Row 0: Header Banner ---
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(DARK_BG)
    ax_hdr.axis("off")
    ax_hdr.text(
        0.0,
        0.62,
        title_text,
        fontsize=22,
        fontweight="bold",
        color=TEXT_MAIN,
        va="center",
    )
    emp_ids_list = kpis.get("employee_ids", [])
    emp_ids_str = f" ({', '.join(f'Emp #{eid}' for eid in emp_ids_list)})" if emp_ids_list else ""
    sub_meta = (
        f"Frames: {kpis['unique_frames']:,}  |  "
        f"Video Duration: {format_duration_mmss(kpis['video_duration_sec'])}  |  "
        f"Employees Identified: {kpis['unique_employees']}{emp_ids_str}  |  "
        f"Mean Wrist Speed: {kpis['mean_wrist_vel_px_s']:.1f} px/s"
    )
    ax_hdr.text(0.0, 0.08, sub_meta, fontsize=10.8, color=TEXT_SUB, va="center")

    # --- Row 1 Left (cols 0..2): Spatial Posture Donut Card ---
    ax_donut_card = fig.add_subplot(gs[1, :3])
    _style_dark_card(ax_donut_card)
    ax_donut_card.set_xticks([])
    ax_donut_card.set_yticks([])

    # Inset axis for crisp circular donut
    ax_pie = ax_donut_card.inset_axes([0.06, 0.14, 0.88, 0.80])
    ax_pie.set_facecolor(CARD_BG)

    pb = kpis["posture_breakdown"]
    labels = list(POSTURE_ORDER)
    vals = [max(pb[k]["count"], 0) for k in labels]
    pcts = [pb[k]["pct"] for k in labels]
    colors = [POSTURE_HEX[k] for k in labels]

    if sum(vals) == 0:
        vals = [0, 0, 0, 1]
        pcts = [0.0, 0.0, 0.0, 100.0]

    wedges, _ = ax_pie.pie(
        vals,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.44, edgecolor=CARD_BG, linewidth=3.0),
    )

    # Outside callout labels matching mockup
    total_v = sum(vals)
    cum_angle = 90.0
    for lbl, v, pct in zip(labels, vals, pcts):
        sweep = (v / total_v) * 360.0
        mid_deg = cum_angle - 0.5 * sweep
        cum_angle -= sweep
        if v <= 0:
            continue
        rad = np.deg2rad(mid_deg)
        r_lbl = 1.24
        ha = "left" if np.cos(rad) >= 0.1 else ("right" if np.cos(rad) <= -0.1 else "center")
        ax_pie.text(
            r_lbl * np.cos(rad),
            r_lbl * np.sin(rad),
            f"{lbl}\n{pct:.1f}%",
            ha=ha,
            va="center",
            fontsize=11.5,
            fontweight="semibold",
            color=TEXT_MAIN,
        )

    ax_pie.set(aspect="equal")

    # Bottom dot legend inside donut card
    leg_handles = [
        mpatches.Patch(color=POSTURE_HEX[k], label=k) for k in labels
    ]
    ax_donut_card.legend(
        handles=leg_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10.5,
        labelcolor="#CBD5E1",
        handlelength=1.0,
        handleheight=1.0,
        bbox_to_anchor=(0.5, 0.01),
    )

    # --- Row 1 Right (cols 3..4): Stacked Working & Idle KPI Cards with Sparklines ---
    sub_kpi = gs[1, 3:].subgridspec(2, 1, hspace=0.16)

    kpi_specs = [
        (
            "Working",
            kpis["working_pct"],
            kpis["working_sec"],
            kpis["working_sparkline"],
            "#4F7DF3",
            "#22D3EE",
            "#253560",
        ),
        (
            "Idle",
            kpis["idle_pct"],
            kpis["idle_sec"],
            kpis["idle_sparkline"],
            "#2CA89A",
            "#8B5CF6",
            "#1F3B44",
        ),
    ]

    for idx, (lbl, pct_val, sec_val, spark_y, line_col, pill_col, border_col) in enumerate(kpi_specs):
        ax_k = fig.add_subplot(sub_kpi[idx, 0])
        _style_dark_card(ax_k, border_color=border_col)
        ax_k.set_xticks([])
        ax_k.set_yticks([])
        ax_k.set_xlim(0, 1)
        ax_k.set_ylim(0, 1)

        ax_k.text(0.07, 0.78, lbl, fontsize=14.5, color="#B8C4D4", fontweight="medium", va="center")
        ax_k.text(0.07, 0.49, f"{pct_val:.1f}%", fontsize=31, color=TEXT_MAIN, fontweight="bold", va="center")
        ax_k.text(
            0.47,
            0.45,
            f"({format_duration_mmss(sec_val)})",
            fontsize=10.5,
            color=TEXT_SUB,
            va="center",
        )

        # Decorative status pills & progress bar matching UI mockup
        ax_k.plot([0.78, 0.83], [0.79, 0.79], color="#334155", lw=5, solid_capstyle="round")
        ax_k.plot([0.87, 0.93], [0.79, 0.79], color="#2CA89A", lw=5, solid_capstyle="round")
        ax_k.plot([0.73, 0.93], [0.51, 0.51], color="#262E3D", lw=5, solid_capstyle="round")
        fill_end = 0.73 + 0.20 * min(max(pct_val / 100.0, 0.12), 1.0)
        ax_k.plot([0.73, fill_end], [0.51, 0.51], color=pill_col if idx == 1 else line_col, lw=5, solid_capstyle="round")

        # Bottom Sparkline inset
        ax_sp = ax_k.inset_axes([0.06, 0.08, 0.88, 0.24])
        ax_sp.set_facecolor(CARD_BG)
        xs = np.linspace(0, 1, max(len(spark_y), 2))
        ys = np.array(spark_y if len(spark_y) >= 2 else [pct_val, pct_val], dtype=np.float64)
        ax_sp.plot(xs, ys, color=line_col, linewidth=2.3)
        ax_sp.fill_between(xs, ys.min() - 3.0, ys, color=line_col, alpha=0.14)
        ax_sp.set_xlim(0, 1)
        span = max(float(ys.max() - ys.min()), 5.0)
        ax_sp.set_ylim(float(ys.min() - 0.35 * span), float(ys.max() + 0.35 * span))
        ax_sp.axis("off")

    # --- Row 2: Ergonomic Posture Timeline (Gantt Ribbon) ---
    ax_tl = fig.add_subplot(gs[2, :])
    _style_dark_card(ax_tl)
    segs = extract_state_segments(df, state_col="posture_state")
    lanes = sorted({s["track_label"] for s in segs}, reverse=True)
    lane_map = {ln: idx for idx, ln in enumerate(lanes)}

    for seg in segs:
        y_idx = lane_map[seg["track_label"]]
        ax_tl.barh(
            y=y_idx,
            width=seg["duration_sec"],
            left=seg["start_sec"],
            height=0.52,
            color=STATE_HEX.get(seg["state"], "#475569"),
            edgecolor="none",
        )

    ax_tl.set_yticks(list(range(len(lanes))))
    ax_tl.set_yticklabels(lanes, color=TEXT_MAIN, fontsize=10.5, fontweight="semibold")
    ax_tl.set_xlabel("Timeline (Seconds)", color=TEXT_SUB, fontsize=10)
    ax_tl.tick_params(axis="x", colors=TEXT_SUB, labelsize=9.5)
    ax_tl.set_title(
        "Ergonomic Posture Timeline (Sitting vs. Standing vs. Walking vs. Away)",
        color=TEXT_MAIN,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax_tl.grid(axis="x", color="#262E3D", linestyle="--", alpha=0.5)

    # --- Row 3: Wrist Velocity Activity Heatmap ---
    ax_hm = fig.add_subplot(gs[3, :])
    _style_dark_card(ax_hm)
    hm = build_velocity_heatmap_data(df, bin_seconds=2.0)
    z_mat = np.array(hm["z_matrix"], dtype=np.float64)
    vmax = float(np.percentile(z_mat[z_mat > 0], 96.0)) if np.any(z_mat > 0) else 100.0
    x_min_hm = hm["x_centers"][0] if hm["x_centers"] else 0.0
    x_max_hm = hm["x_centers"][-1] if hm["x_centers"] else 1.0

    im = ax_hm.imshow(
        z_mat,
        aspect="auto",
        cmap="YlGnBu_r",
        vmin=0.0,
        vmax=max(vmax, 50.0),
        extent=[x_min_hm, max(x_max_hm, x_min_hm + 1.0), -0.5, len(hm["y_labels"]) - 0.5],
        origin="lower",
    )
    ax_hm.set_yticks(list(range(len(hm["y_labels"]))))
    ax_hm.set_yticklabels(hm["y_labels"], color=TEXT_MAIN, fontsize=9.5)
    ax_hm.set_xlabel("Timeline (Seconds, 2.0s Bins)", color=TEXT_SUB, fontsize=10)
    ax_hm.tick_params(axis="x", colors=TEXT_SUB, labelsize=9.5)
    ax_hm.set_title(
        "Wrist Velocity Activity Heatmap (30-Frame Rolling Speed, px/s)",
        color=TEXT_MAIN,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.018, pad=0.012)
    cbar.ax.tick_params(colors=TEXT_SUB, labelsize=8.5)
    cbar.set_label("px/s", color=TEXT_SUB, fontsize=9)

    return fig


def _draw_summary_table_figure(summary_df: pl.DataFrame, thresholds: Dict[str, Any]) -> plt.Figure:
    """Render Page 2 of the Executive PDF Report with the Per-Employee KPI Table & Threshold Audit."""
    fig = plt.figure(figsize=(16.5, 9.0), facecolor=DARK_BG)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.84])
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    ax.text(
        0.0,
        0.95,
        "Executive Telemetry Summary & Per-Employee Breakdown",
        fontsize=20,
        fontweight="bold",
        color=TEXT_MAIN,
    )
    fg_t = thresholds.get("posture_zones", {}).get("foreground", {}).get("threshold_px", 457.94)
    vel_t = thresholds.get("wrist_velocity_threshold_px_s", 32.01)
    ax.text(
        0.0,
        0.89,
        f"Calibrated Cutoffs  •  Foreground Sitting/Standing: {fg_t:.2f} px  |  "
        f"Rolling 30f Wrist Velocity: {vel_t:.2f} px/s  |  Idle Confirmation Dwell: 5.0s",
        fontsize=11,
        color=TEXT_SUB,
    )

    col_headers = [
        "Scope",
        "Active Duration",
        "Working %",
        "Idle %",
        "Sitting %",
        "Standing %",
        "Walking %",
        "Away %",
        "Mean Wrist (px/s)",
        "Transitions",
    ]
    cell_rows = []
    for row in summary_df.iter_rows(named=True):
        cell_rows.append(
            [
                row["Scope"],
                row["Duration_Formatted"],
                f"{row['Working_Pct']:.1f}%",
                f"{row['Idle_Pct']:.1f}%",
                f"{row['Sitting_Pct']:.1f}%",
                f"{row['Standing_Pct']:.1f}%",
                f"{row['Walking_Pct']:.1f}%",
                f"{row['Away_Pct']:.1f}%",
                f"{row['Mean_Wrist_Speed_px_s']:.1f}",
                str(row["Posture_Transitions"]),
            ]
        )

    tbl = ax.table(
        cellText=cell_rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.35, 1.0, 0.44],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)

    for (r_idx, c_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor(CARD_BORDER)
        cell.set_linewidth(1.2)
        if r_idx == 0:
            cell.set_facecolor("#1E293B")
            cell.get_text().set_color("#38BDF8")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(CARD_BG if r_idx % 2 == 1 else "#111620")
            cell.get_text().set_color(TEXT_MAIN)

    ax.text(
        0.0,
        0.20,
        "Privacy & Methodology Note: All KPIs are computed strictly from anonymized 2D skeletal coordinates "
        "and torso clothing color histograms without storing biometric facial templates or raw video frames in CSV telemetry.",
        fontsize=10,
        color=TEXT_SUB,
    )
    return fig


def generate_executive_reports(
    csv_path: Optional[Path] = None,
    output_dir: Path = REPORTS_OUTPUT_DIR,
) -> Dict[str, str]:
    """Generate executive CSV, JSON, high-resolution dark-UI PNG snapshot, and 2-page PDF report."""
    ensure_directories()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_telemetry_dataframe(csv_path)
    thresholds = load_calibrated_thresholds()
    kpis = compute_dashboard_kpis(df)
    summary_df = build_employee_summary_table(df)

    csv_out = out_dir / "executive_kpi_summary.csv"
    json_out = out_dir / "executive_kpi_summary.json"
    png_out = out_dir / "executive_dashboard_snapshot.png"
    pdf_out = out_dir / "executive_motion_report.pdf"

    summary_df.write_csv(csv_out)

    json_payload = {
        "kpis": kpis,
        "employee_table": summary_df.to_dicts(),
    }
    with open(json_out, "w") as f:
        json.dump(json_payload, f, indent=2)

    fig_hero = _draw_executive_hero_figure(df, kpis, title_text="Employee Motion Dashboard")
    fig_hero.savefig(png_out, dpi=160, facecolor=DARK_BG)

    fig_tbl = _draw_summary_table_figure(summary_df, thresholds)
    with PdfPages(pdf_out) as pdf:
        pdf.savefig(fig_hero, facecolor=DARK_BG)
        pdf.savefig(fig_tbl, facecolor=DARK_BG)

    plt.close(fig_hero)
    plt.close(fig_tbl)

    return {
        "summary_csv_path": str(csv_out),
        "summary_json_path": str(json_out),
        "dashboard_png_path": str(png_out),
        "executive_pdf_path": str(pdf_out),
    }
