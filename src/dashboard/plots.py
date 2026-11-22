"""Interactive dark-glassmorphic Plotly visualization builders for Milestone 3:
  1. Spatial Posture Donut & Workstation Utilization Donut (with unclipped outside callouts)
  2. Glowing KPI Sparklines (Working % & Idle % trends)
  3. Multi-employee (`1..N`) Ergonomic Posture & Activity Gantt Timeline + Normalized Shoulder Elevation
  4. Scale-Invariant (`SW/s`) 2D Rolling Wrist Velocity Activity Heatmap
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import polars as pl

from src.dashboard.data_loader import (
    ACTIVITY_ORDER,
    POSTURE_ORDER,
    build_velocity_heatmap_data,
    extract_state_segments,
    format_duration_mmss,
)

POSTURE_COLORS: Dict[str, str] = {
    "Sitting": "#23478E",
    "Standing": "#5D3FD3",
    "Walking": "#1F6E9C",
    "Away": "#2CA89A",
}

UTILIZATION_COLORS: Dict[str, str] = {
    "Working": "#3B82F6",
    "Idle": "#8B5CF6",
    "Walking": "#1F6E9C",
    "Away": "#2CA89A",
}

EMPLOYEE_TRACE_PALETTE: List[str] = [
    "#38BDF8",
    "#A78BFA",
    "#34D399",
    "#FBBF24",
    "#F472B6",
    "#FB923C",
    "#2DD4BF",
    "#818CF8",
    "#A3E635",
    "#F87171",
    "#C084FC",
    "#67E8F9",
]

CARD_BG = "rgba(0,0,0,0)"
DARK_SLICE_BORDER = "#131720"
TEXT_PRIMARY = "#F0F6FC"
TEXT_MUTED = "#94A3B8"
GRID_SUBTLE = "rgba(148, 163, 184, 0.12)"


def build_posture_donut_figure(
    kpis: Dict[str, Any],
    height: int = 420,
) -> go.Figure:
    """Build the Spatial Posture & Presence Donut Chart with automatic callout margin protection."""
    pb = kpis.get("posture_breakdown", {})
    labels = list(POSTURE_ORDER)
    counts = [int(pb.get(lbl, {}).get("count", 0)) for lbl in labels]
    secs = [float(pb.get(lbl, {}).get("seconds", 0.0)) for lbl in labels]
    pcts = [float(pb.get(lbl, {}).get("pct", 0.0)) for lbl in labels]
    colors = [POSTURE_COLORS[lbl] for lbl in labels]

    if sum(counts) == 0:
        counts = [0, 0, 0, 1]
        pcts = [0.0, 0.0, 0.0, 100.0]

    callout_text = [
        f"{lbl}<br>{pct:.1f}%" if cnt > 0 else ""
        for lbl, pct, cnt in zip(labels, pcts, counts)
    ]
    hover_text = [
        f"<b>{lbl}</b><br>Share: {pct:.1f}%<br>Duration: {format_duration_mmss(sec)} ({cnt} frames)"
        for lbl, pct, sec, cnt in zip(labels, pcts, secs, counts)
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.56,
                sort=False,
                direction="clockwise",
                rotation=0,
                automargin=True,
                marker=dict(
                    colors=colors,
                    line=dict(color=DARK_SLICE_BORDER, width=3),
                ),
                text=callout_text,
                textinfo="text",
                textposition="outside",
                textfont=dict(size=13.5, color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
                hovertext=hover_text,
                hoverinfo="text",
                pull=[0.0, 0.0, 0.0, 0.015],
            )
        ]
    )

    fig.update_layout(
        height=height,
        margin=dict(l=42, r=42, t=56, b=58),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#CBD5E1"),
            itemsizing="constant",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def build_utilization_donut_figure(
    kpis: Dict[str, Any],
    three_way_only: bool = False,
    height: int = 420,
) -> go.Figure:
    """Build the Workstation Utilization Donut Chart (Working vs. Idle vs. Walking/Away)."""
    if three_way_only:
        u3 = kpis.get("utilization_3way", {})
        labels = ["Working", "Idle", "Away"]
        counts = [int(u3.get(lbl, {}).get("count", 0)) for lbl in labels]
        secs = [float(u3.get(lbl, {}).get("seconds", 0.0)) for lbl in labels]
        pcts = [float(u3.get(lbl, {}).get("share_ex_walk_pct", 0.0)) for lbl in labels]
    else:
        ab = kpis.get("activity_breakdown", {})
        labels = list(ACTIVITY_ORDER)
        counts = [int(ab.get(lbl, {}).get("count", 0)) for lbl in labels]
        secs = [float(ab.get(lbl, {}).get("seconds", 0.0)) for lbl in labels]
        pcts = [float(ab.get(lbl, {}).get("pct", 0.0)) for lbl in labels]

    colors = [UTILIZATION_COLORS[lbl] for lbl in labels]
    if sum(counts) == 0:
        counts[-1] = 1
        pcts[-1] = 100.0

    callout_text = [
        f"{lbl}<br>{pct:.1f}%" if cnt > 0 else ""
        for lbl, pct, cnt in zip(labels, pcts, counts)
    ]
    hover_text = [
        f"<b>{lbl}</b><br>Share: {pct:.1f}%<br>Duration: {format_duration_mmss(sec)} ({cnt} frames)"
        for lbl, pct, sec, cnt in zip(labels, pcts, secs, counts)
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.56,
                sort=False,
                direction="clockwise",
                rotation=0,
                automargin=True,
                marker=dict(colors=colors, line=dict(color=DARK_SLICE_BORDER, width=3)),
                text=callout_text,
                textinfo="text",
                textposition="outside",
                textfont=dict(size=13.5, color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
                hovertext=hover_text,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        height=height,
        margin=dict(l=42, r=42, t=56, b=58),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def build_kpi_sparkline_figure(
    x_sec: List[float],
    y_pct: List[float],
    line_color: str = "#4F7DF3",
    fill_rgba: str = "rgba(79, 125, 243, 0.14)",
    metric_name: str = "Working",
    height: int = 68,
) -> go.Figure:
    """Build a compact frameless neon sparkline for executive KPI cards."""
    xs = list(x_sec) if x_sec else [0.0, 1.0]
    ys = list(y_pct) if y_pct else [0.0, 0.0]
    y_min = min(ys)
    y_max = max(ys)
    span = max(y_max - y_min, 6.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_color, width=3.0, shape="spline", smoothing=0.75),
            fill="tozeroy",
            fillcolor=fill_rgba,
            hovertemplate=f"<b>{metric_name}</b>: %{{y:.1f}}% at %{{x:.1f}}s<extra></extra>",
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=2, r=2, t=4, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(
            visible=False,
            fixedrange=True,
            range=[max(0.0, y_min - 0.35 * span), min(102.0, y_max + 0.30 * span)],
        ),
    )
    return fig


def build_ergonomic_timeline_figure(
    df: pl.DataFrame,
    include_activity_ribbon: bool = True,
    height: int = 300,
) -> go.Figure:
    """Build an interactive Gantt / state-ribbon timeline scaling dynamically to `1..N` employees."""
    posture_segs = extract_state_segments(df, state_col="posture_state")
    activity_segs = extract_state_segments(df, state_col="activity_state") if include_activity_ribbon else []

    fig = go.Figure()
    added_legend_states = set()
    unique_lanes = set()

    def _add_segments(segs: List[Dict[str, Any]], lane_suffix: str, color_map: Dict[str, str]) -> None:
        for seg in segs:
            st = seg["state"]
            lane_name = f"{seg['track_label']} • {lane_suffix}"
            unique_lanes.add(lane_name)
            color = color_map.get(st, "#475569")
            show_leg = st not in added_legend_states
            if show_leg:
                added_legend_states.add(st)

            fig.add_trace(
                go.Bar(
                    x=[seg["duration_sec"]],
                    y=[lane_name],
                    base=[seg["start_sec"]],
                    orientation="h",
                    name=st,
                    legendgroup=st,
                    showlegend=show_leg,
                    marker=dict(
                        color=color,
                        line=dict(width=0),
                    ),
                    hovertemplate=(
                        f"<b>{lane_name}</b><br>"
                        f"State: <b>{st}</b> ({seg['motion_detail']})<br>"
                        f"Window: {seg['start_sec']:.1f}s – {seg['end_sec']:.1f}s "
                        f"({seg['duration_sec']:.1f}s)<br>"
                        f"Frames: {seg['start_frame']} – {seg['end_frame']}<extra></extra>"
                    ),
                )
            )

    combined_colors = {**UTILIZATION_COLORS, **POSTURE_COLORS}
    if include_activity_ribbon and activity_segs:
        _add_segments(activity_segs, "Activity", combined_colors)
    _add_segments(posture_segs, "Posture", combined_colors)

    dyn_height = max(int(height), 48 * max(len(unique_lanes), 1) + 95)

    fig.update_layout(
        barmode="overlay",
        height=dyn_height,
        margin=dict(l=15, r=20, t=30, b=45),
        paper_bgcolor=CARD_BG,
        plot_bgcolor="rgba(15, 20, 30, 0.55)",
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        xaxis=dict(
            title="Video Timeline (Seconds)",
            gridcolor=GRID_SUBTLE,
            zeroline=False,
            color=TEXT_MUTED,
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(0,0,0,0)",
            color=TEXT_PRIMARY,
            tickfont=dict(size=12),
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=12, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def build_posture_elevation_figure(
    df: pl.DataFrame,
    thresholds: Optional[Dict[str, Any]] = None,
    height: int = 295,
) -> go.Figure:
    """Build a scale-invariant normalized shoulder-elevation `(H - Y) / H` chart across `1..N` employees."""
    fig = go.Figure()
    present = df.filter((pl.col("presence") == 1) & pl.col("track_id").is_not_null()).sort(
        ["track_id", "frame"]
    )

    use_norm = "shoulder_dist_bottom_norm" in present.columns
    y_col = "shoulder_dist_bottom_norm" if use_norm else "shoulder_dist_bottom_px"
    y_axis_label = "Normalized Shoulder Height (H - Y) / H" if use_norm else "Vertical Distance (px)"

    if not present.is_empty() and y_col in present.columns:
        for idx, (tid_tuple, grp) in enumerate(present.group_by(["track_id"], maintain_order=True)):
            tid = tid_tuple[0] if isinstance(tid_tuple, tuple) else tid_tuple
            col = EMPLOYEE_TRACE_PALETTE[idx % len(EMPLOYEE_TRACE_PALETTE)]
            unit_fmt = "%{y:.3f} H" if use_norm else "%{y:.1f} px"
            fig.add_trace(
                go.Scatter(
                    x=grp["timestamp_sec"].to_list(),
                    y=grp[y_col].to_list(),
                    mode="lines",
                    name=f"Emp #{tid} Shoulder Elevation",
                    line=dict(color=col, width=2.0),
                    customdata=grp["posture_state"].to_list(),
                    hovertemplate=(
                        f"<b>Emp #{tid}</b> • %{{customdata}}<br>"
                        f"Time: %{{x:.2f}}s<br>Elevation: {unit_fmt}<extra></extra>"
                    ),
                )
            )

    thresh_val = None
    if thresholds:
        fg = thresholds.get("posture_zones", {}).get("foreground", {})
        if use_norm:
            thresh_val = fg.get("threshold_norm") or thresholds.get("global_posture_threshold_norm")
        else:
            thresh_val = fg.get("threshold_px") or thresholds.get("global_posture_threshold_px")

    if thresh_val is not None:
        lbl_str = f"{float(thresh_val):.3f} H" if use_norm else f"{float(thresh_val):.1f} px"
        fig.add_hline(
            y=float(thresh_val),
            line_dash="dash",
            line_color="#F43F5E",
            line_width=1.8,
            annotation_text=f"Sitting ↔ Standing Cutoff ({lbl_str})",
            annotation_position="top right",
            annotation_font=dict(color="#FDA4AF", size=11.5),
        )

    fig.update_layout(
        height=height,
        margin=dict(l=15, r=20, t=30, b=40),
        paper_bgcolor=CARD_BG,
        plot_bgcolor="rgba(15, 20, 30, 0.55)",
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        xaxis=dict(title="Video Timeline (Seconds)", gridcolor=GRID_SUBTLE, color=TEXT_MUTED),
        yaxis=dict(title=y_axis_label, gridcolor=GRID_SUBTLE, color=TEXT_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def build_activity_heatmap_figure(
    df: pl.DataFrame,
    bin_seconds: float = 2.0,
    velocity_threshold_sw_s: Optional[float] = None,
    velocity_threshold_px_s: Optional[float] = None,
    height: int = 305,
) -> go.Figure:
    """Build an interactive 2D Wrist Velocity Activity Heatmap (`SW/s`) scaling cleanly to `1..N` employees."""
    hm = build_velocity_heatmap_data(df, bin_seconds=bin_seconds)
    unit = hm.get("unit", "SW/s")

    z_arr = np.array(hm["z_matrix"], dtype=np.float64)
    if unit == "SW/s":
        z_cap = float(np.percentile(z_arr[z_arr > 0], 96.0)) if np.any(z_arr > 0) else 0.8
        ref_cut = velocity_threshold_sw_s if velocity_threshold_sw_s is not None else 0.25
        z_max = max(z_cap, float(ref_cut) * 2.2, 0.45)
        z_max_rounded = round(z_max, 2)
    else:
        z_cap = float(np.percentile(z_arr[z_arr > 0], 96.0)) if np.any(z_arr > 0) else 120.0
        z_max_rounded = round(max(z_cap, (velocity_threshold_px_s or 32.0) * 2.5, 60.0), 1)

    colorscale = [
        [0.00, "#0D131F"],
        [0.16, "#162447"],
        [0.35, "#1E4E8C"],
        [0.60, "#0284C7"],
        [0.82, "#22D3EE"],
        [1.00, "#A5F3FC"],
    ]

    dyn_height = max(int(height), 32 * max(len(hm["y_labels"]), 1) + 95)

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=hm["x_centers"],
                y=hm["y_labels"],
                z=hm["z_matrix"],
                text=hm["hover_text"],
                hoverinfo="text",
                colorscale=colorscale,
                zmin=0.0,
                zmax=z_max_rounded,
                xgap=1.5,
                ygap=3.0,
                colorbar=dict(
                    title=dict(text=unit, font=dict(color=TEXT_MUTED, size=11)),
                    tickfont=dict(color=TEXT_MUTED, size=11),
                    thickness=12,
                    len=0.88,
                ),
            )
        ]
    )

    fig.update_layout(
        height=dyn_height,
        margin=dict(l=15, r=20, t=20, b=45),
        paper_bgcolor=CARD_BG,
        plot_bgcolor="rgba(13, 19, 31, 0.85)",
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        xaxis=dict(
            title=f"Video Timeline (Seconds, {hm.get('bin_seconds', bin_seconds):.1f}s Bins)",
            gridcolor="rgba(0,0,0,0)",
            color=TEXT_MUTED,
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(0,0,0,0)",
            color=TEXT_PRIMARY,
            tickfont=dict(size=12),
            automargin=True,
        ),
    )
    return fig
