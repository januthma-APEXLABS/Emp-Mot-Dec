"""Milestone 3: Interactive Executive Dashboard & Reporting Application (Streamlit + Plotly).

Launch locally with:
    ./venv/bin/streamlit run app.py
or:
    ./venv/bin/python scripts/run_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config  # noqa: F401
import polars as pl
import streamlit as st

from src.calibration.rules_engine import run_milestone2_pipeline
from src.config import (
    DEFAULT_YOLO_MODEL,
    PLOTS_OUTPUT_DIR,
    RAW_VIDEOS_DIR,
    REPORTS_OUTPUT_DIR,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
    discover_raw_videos,
    ensure_directories,
)
from src.dashboard.data_loader import (
    build_employee_detail_profiles,
    build_employee_summary_table,
    compute_dashboard_kpis,
    filter_telemetry,
    format_duration_mmss,
    get_available_employees,
    get_available_videos,
    get_latest_played_video,
    get_telemetry_cache_fingerprint,
    load_all_played_telemetry,
    load_calibrated_thresholds,
)
from src.dashboard.plots import (
    build_activity_heatmap_figure,
    build_ergonomic_timeline_figure,
    build_kpi_sparkline_figure,
    build_posture_donut_figure,
    build_posture_elevation_figure,
    build_utilization_donut_figure,
)
from src.dashboard.reporter import generate_executive_reports
from src.tracking.pipeline import BatchTrackingPipeline, render_video_from_telemetry_csv

st.set_page_config(
    page_title="Employee Motion Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_DARK_CSS = """
<style>
    /* Overall carbon-black canvas */
    .stApp {
        background: radial-gradient(circle at 78% 18%, rgba(45, 85, 165, 0.12) 0%, rgba(11, 14, 20, 1) 48%), #0B0E14;
        color: #F0F6FC;
    }
    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 2.5rem;
        max-width: 1420px;
    }
    section[data-testid="stSidebar"] {
        background-color: #10141D;
        border-right: 1px solid #1E2636;
    }

    /* Glassmorphic bordered containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(22, 27, 38, 0.92) 0%, rgba(17, 21, 30, 0.96) 100%);
        border: 1px solid rgba(56, 70, 97, 0.48) !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        padding: 0.35rem 0.55rem;
    }

    /* Dashboard Header */
    .dash-title {
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .dash-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-bottom: 1.25rem;
    }
    .meta-pill {
        display: inline-block;
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(71, 85, 105, 0.45);
        border-radius: 999px;
        padding: 0.2rem 0.72rem;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
        font-size: 0.80rem;
        color: #CBD5E1;
    }

    /* Mockup-matched KPI header & big stat layout */
    .kpi-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0.5rem 0.1rem 0.5rem;
    }
    .kpi-label {
        font-size: 1.22rem;
        font-weight: 500;
        color: #9CA8BA;
    }
    .kpi-pills {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .pill-muted {
        width: 24px;
        height: 7px;
        border-radius: 999px;
        background: #333C4D;
        display: inline-block;
    }
    .pill-teal {
        width: 24px;
        height: 7px;
        border-radius: 999px;
        background: #2DD4BF;
        box-shadow: 0 0 10px rgba(45, 212, 191, 0.45);
        display: inline-block;
    }
    .kpi-mid-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.15rem 0.5rem 0.0rem 0.5rem;
    }
    .kpi-big-value {
        font-size: 3.15rem;
        font-weight: 700;
        line-height: 1.08;
        letter-spacing: -0.025em;
        color: #F8FAFC;
    }
    .kpi-sub-dur {
        font-size: 0.85rem;
        color: #64748B;
        padding-left: 0.55rem;
        margin-top: 0.1rem;
    }
    .kpi-bar-track {
        width: 92px;
        height: 7px;
        border-radius: 999px;
        background: #252C3B;
        overflow: hidden;
    }
    .kpi-bar-fill-blue {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #3B82F6, #60A5FA);
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.6);
    }
    .kpi-bar-fill-purple {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #7C3AED, #C084FC);
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
    }
    .section-heading {
        font-size: 1.15rem;
        font-weight: 600;
        color: #E2E8F0;
        margin: 0.25rem 0 0.15rem 0.25rem;
    }
    .section-caption {
        font-size: 0.84rem;
        color: #94A3B8;
        margin: 0 0 0.5rem 0.25rem;
    }

    /* Individual Employee Detail Card Styling */
    .emp-card-hdr {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }
    .emp-card-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .emp-state-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.58rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(96, 165, 250, 0.45);
        color: #93C5FD;
    }
    .emp-meta-line {
        font-size: 0.81rem;
        color: #94A3B8;
        margin-bottom: 0.45rem;
    }
    .emp-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        margin-bottom: 0.45rem;
    }
    .emp-stat-box {
        background: rgba(15, 20, 30, 0.72);
        border: 1px solid rgba(51, 65, 85, 0.45);
        border-radius: 8px;
        padding: 0.3rem 0.45rem;
        text-align: center;
    }
    .emp-stat-val {
        font-size: 0.96rem;
        font-weight: 700;
        color: #F1F5F9;
    }
    .emp-stat-lbl {
        font-size: 0.70rem;
        color: #94A3B8;
    }
</style>
"""
st.markdown(CUSTOM_DARK_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_load_all_telemetry(_fingerprint: tuple) -> pl.DataFrame:
    return load_all_played_telemetry(TELEMETRY_OUTPUT_DIR)


@st.cache_data(show_spinner=False)
def _cached_load_thresholds(_fingerprint: tuple, video_name: str) -> dict:
    return load_calibrated_thresholds(video_name=video_name, telemetry_dir=TELEMETRY_OUTPUT_DIR)


def _play_and_register_video_stream(video_path: Path) -> None:
    """Run tracking, 6-class calibration, and H.264 rendering on `video_path` and register it immediately."""
    ensure_directories()
    with st.sidebar.status(f"Playing & tracking `{video_path.name}`...", expanded=True) as status:
        st.write("1/3 Running multi-employee skeletal & pixel-motion tracking...")
        pipeline = BatchTrackingPipeline(model_path=DEFAULT_YOLO_MODEL, smoother_type="kalman")
        stats = pipeline.process_video(video_path=video_path, save_video=False)

        st.write("2/3 Calibrating scale-invariant 6-class motion states...")
        landmarks_csv = Path(stats["output_csv_path"])
        run_milestone2_pipeline(
            input_csv_path=landmarks_csv,
            output_telemetry_dir=TELEMETRY_OUTPUT_DIR,
            output_plots_dir=PLOTS_OUTPUT_DIR,
        )

        st.write("3/3 Encoding annotated H.264 stream...")
        classified_csv = TELEMETRY_OUTPUT_DIR / f"{video_path.stem}_classified.csv"
        out_vid = VIDEOS_OUTPUT_DIR / f"{video_path.stem}_tracked.mp4"
        render_video_from_telemetry_csv(
            video_path=video_path,
            landmarks_csv_path=classified_csv if classified_csv.exists() else landmarks_csv,
            output_video_path=out_vid,
            smoother_name="Kalman",
        )
        status.update(label=f"`{video_path.name}` added to Video Stream!", state="complete", expanded=False)

    st.session_state["selected_video_stream"] = video_path.name
    st.session_state["auto_expand_video_player"] = True


def _parse_emp_option_id(option_label: str) -> Optional[int]:
    if option_label.startswith("All"):
        return None
    token = option_label.replace("Emp #", "").split("•")[0].strip()
    return int(token)


def main() -> None:
    fingerprint = get_telemetry_cache_fingerprint(TELEMETRY_OUTPUT_DIR)

    # Discover raw videos in `data/raw_videos/` so users can play/add any new stream right from the UI
    raw_video_paths = discover_raw_videos(RAW_VIDEOS_DIR)

    if not fingerprint:
        with st.sidebar:
            st.markdown("### 🎛️ Dashboard Controls")
            if raw_video_paths:
                pick_init = st.selectbox("Select Raw Video to Play", options=[p.name for p in raw_video_paths])
                if st.button("▶️ Play & Track Video Stream", type="primary", width="stretch"):
                    _play_and_register_video_stream(RAW_VIDEOS_DIR / pick_init)
                    st.rerun()
        st.info("No played video telemetry found yet. Play a video stream from the sidebar or run Milestone 1.")
        return

    raw_df = _cached_load_all_telemetry(fingerprint)
    videos = get_available_videos(raw_df, telemetry_dir=TELEMETRY_OUTPUT_DIR)
    if not videos:
        st.error("No video streams available in telemetry.")
        return

    # Auto-select newly played video stream the moment it appears in `videos` while keeping all prior streams
    known_prev = st.session_state.get("known_played_videos", set())
    newly_added = [v for v in videos if v not in known_prev]
    if known_prev and newly_added:
        latest_manifest = get_latest_played_video(TELEMETRY_OUTPUT_DIR)
        st.session_state["selected_video_stream"] = (
            latest_manifest if latest_manifest in newly_added else newly_added[-1]
        )
    st.session_state["known_played_videos"] = set(videos)

    if st.session_state.get("selected_video_stream") not in videos:
        latest_manifest = get_latest_played_video(TELEMETRY_OUTPUT_DIR)
        st.session_state["selected_video_stream"] = (
            latest_manifest if latest_manifest in videos else videos[0]
        )

    # --- Sidebar Filters & Stream Player Controls ---
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        selected_video = st.selectbox(
            "Video Stream",
            options=videos,
            index=videos.index(st.session_state["selected_video_stream"]),
            key="video_stream_selectbox",
        )
        st.session_state["selected_video_stream"] = selected_video

        emp_ids = get_available_employees(raw_df, video_name=selected_video)
        emp_options = ["All Employees (Overview)"] + [f"Emp #{eid}" for eid in emp_ids]

        emp_state_key = f"emp_filter_choice_{selected_video}"
        pending_focus = st.session_state.pop("pending_emp_focus_label", None)
        if pending_focus in emp_options:
            st.session_state[emp_state_key] = pending_focus
        if st.session_state.get(emp_state_key) not in emp_options:
            st.session_state[emp_state_key] = emp_options[0]

        selected_emp_label = st.selectbox(
            "Employee Filter",
            options=emp_options,
            key=emp_state_key,
        )
        selected_emp_id = _parse_emp_option_id(selected_emp_label)

        vid_df = raw_df.filter(pl.col("video_name") == selected_video)
        t_min_all = float(vid_df["timestamp_sec"].min() or 0.0)
        t_max_all = float(vid_df["timestamp_sec"].max() or 1.0)
        t_hi_safe = max(round(t_max_all, 1), round(t_min_all, 1) + 1.0)

        selected_time = st.slider(
            "Timeline Window (Seconds)",
            min_value=round(t_min_all, 1),
            max_value=t_hi_safe,
            value=(round(t_min_all, 1), t_hi_safe),
            step=1.0,
            key=f"time_slider_{selected_video}",
        )

        donut_view = st.radio(
            "Primary Donut Chart Mode",
            options=[
                "Spatial Posture (Sitting / Standing / Walking / Away)",
                "Utilization (Working / Idle / Walking / Away)",
                "Workstation Utilization (Working / Idle / Away)",
            ],
            index=0,
        )

        bin_sec = st.select_slider(
            "Velocity Heatmap Bin Size (s)",
            options=[1.0, 2.0, 5.0, 10.0],
            value=2.0,
        )

        with st.expander("🎬 Play / Add Video Stream", expanded=False):
            st.caption("Play or upload any video to immediately track and add it to `Video Stream`.")
            uploaded_vid = st.file_uploader(
                "Upload Video (.mp4, .mov, .avi, .mkv)",
                type=["mp4", "mov", "avi", "mkv"],
                key="sidebar_vid_uploader",
            )
            if uploaded_vid is not None:
                ensure_directories()
                dest_path = RAW_VIDEOS_DIR / Path(uploaded_vid.name).name
                if not dest_path.exists() or dest_path.stat().st_size != uploaded_vid.size:
                    dest_path.write_bytes(uploaded_vid.getbuffer())
                raw_video_paths = discover_raw_videos(RAW_VIDEOS_DIR)

            if raw_video_paths:
                played_set = set(videos)
                raw_labels = [
                    f"{p.name} ({'Played ✓' if p.name in played_set else 'Unplayed • Ready'})"
                    for p in raw_video_paths
                ]
                unplayed_indices = [i for i, p in enumerate(raw_video_paths) if p.name not in played_set]
                default_raw_idx = unplayed_indices[0] if unplayed_indices else 0
                chosen_raw_lbl = st.selectbox(
                    "Raw Videos (`data/raw_videos/`)",
                    options=raw_labels,
                    index=default_raw_idx,
                )
                chosen_raw_path = raw_video_paths[raw_labels.index(chosen_raw_lbl)]
                if st.button("▶️ Play & Track Selected Video", type="primary", width="stretch"):
                    _play_and_register_video_stream(chosen_raw_path)
                    st.rerun()

        st.divider()
        st.caption(
            f"Streams Loaded: **{len(videos)}** • Employees in `{selected_video}`: **{len(emp_ids)}**\n\n"
            "Privacy-Preserving Architecture: Metrics are derived strictly from scale-invariant "
            "keypoint telemetry without storing facial biometrics."
        )

    thresholds = _cached_load_thresholds(fingerprint, selected_video)

    # Apply reactive filters
    filtered_df = filter_telemetry(
        raw_df,
        video_name=selected_video,
        employee_id=selected_emp_id,
        time_range=selected_time,
    )
    kpis = compute_dashboard_kpis(filtered_df)
    emp_profiles = build_employee_detail_profiles(
        raw_df,
        video_name=selected_video,
        time_range=selected_time,
    )

    # --- Top Header ---
    scope_tag = f"All Employees ({len(emp_ids)})" if selected_emp_id is None else f"Emp #{selected_emp_id}"
    fg_cutoff_norm = float(
        thresholds.get("posture_zones", {})
        .get("foreground", {})
        .get("threshold_norm", thresholds.get("global_posture_threshold_norm", 0.45))
    )
    vel_cutoff_sw = float(thresholds.get("wrist_velocity_threshold_sw_s", 0.25))

    st.markdown('<div class="dash-title">Employee Motion Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dash-subtitle">
            <span class="meta-pill">📹 {selected_video} ({len(videos)} streams in list)</span>
            <span class="meta-pill">👤 {scope_tag}</span>
            <span class="meta-pill">⏱️ {format_duration_mmss(kpis['video_duration_sec'])} ({kpis['unique_frames']:,} frames)</span>
            <span class="meta-pill">📐 Posture Cutoff: {fg_cutoff_norm:.2f} H</span>
            <span class="meta-pill">⚡ Wrist Cutoff: {vel_cutoff_sw:.2f} SW/s</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tracked H.264 Stream Player ---
    tracked_mp4 = VIDEOS_OUTPUT_DIR / f"{Path(selected_video).stem}_tracked.mp4"
    raw_mp4 = RAW_VIDEOS_DIR / selected_video
    video_to_play = tracked_mp4 if tracked_mp4.exists() else (raw_mp4 if raw_mp4.exists() else None)
    if video_to_play is not None:
        expand_player = bool(st.session_state.pop("auto_expand_video_player", False))
        with st.expander(
            f"🎬 Preview Video Stream — {video_to_play.name} ({len(emp_ids)} Tracked Employees)",
            expanded=expand_player,
        ):
            st.video(str(video_to_play))

    # --- Row 1: Mockup-Matched Hero Grid (Left Donut Card + Right Stacked Working/Idle Sparkline Cards) ---
    col_donut, col_kpis = st.columns([1.48, 1.0], gap="medium")

    with col_donut:
        with st.container(border=True):
            if donut_view.startswith("Spatial Posture"):
                fig_donut = build_posture_donut_figure(kpis, height=420)
            elif donut_view.startswith("Workstation Utilization"):
                fig_donut = build_utilization_donut_figure(kpis, three_way_only=True, height=420)
            else:
                fig_donut = build_utilization_donut_figure(kpis, three_way_only=False, height=420)

            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with col_kpis:
        # Card 1: Working % + Sparkline
        work_bar_w = min(max(kpis["working_pct"], 12.0), 100.0)
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="kpi-top-row">
                    <span class="kpi-label">Working</span>
                    <span class="kpi-pills">
                        <span class="pill-muted"></span>
                        <span class="pill-teal"></span>
                    </span>
                </div>
                <div class="kpi-mid-row">
                    <span class="kpi-big-value">{kpis['working_pct']:.1f}%</span>
                    <div class="kpi-bar-track">
                        <div class="kpi-bar-fill-blue" style="width: {work_bar_w:.0f}%;"></div>
                    </div>
                </div>
                <div class="kpi-sub-dur">Active Work Time: {format_duration_mmss(kpis['working_sec'])} • Mean Wrist: {kpis['mean_wrist_vel_sw_s']:.2f} SW/s</div>
                """,
                unsafe_allow_html=True,
            )
            fig_sp_work = build_kpi_sparkline_figure(
                kpis["sparkline_x"],
                kpis["working_sparkline"],
                line_color="#4F7DF3",
                fill_rgba="rgba(79, 125, 243, 0.15)",
                metric_name="Working",
                height=72,
            )
            st.plotly_chart(fig_sp_work, use_container_width=True, config={"displayModeBar": False})

        # Card 2: Idle % + Sparkline
        idle_bar_w = min(max(kpis["idle_pct"] * 2.2, 16.0), 100.0)
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="kpi-top-row">
                    <span class="kpi-label">Idle</span>
                    <span class="kpi-pills">
                        <span class="pill-teal"></span>
                        <span class="pill-muted"></span>
                    </span>
                </div>
                <div class="kpi-mid-row">
                    <span class="kpi-big-value">{kpis['idle_pct']:.1f}%</span>
                    <div class="kpi-bar-track">
                        <div class="kpi-bar-fill-purple" style="width: {idle_bar_w:.0f}%;"></div>
                    </div>
                </div>
                <div class="kpi-sub-dur">Idle / Break Time: {format_duration_mmss(kpis['idle_sec'])} • Away: {kpis['away_pct']:.1f}% ({format_duration_mmss(kpis['posture_breakdown']['Away']['seconds'])})</div>
                """,
                unsafe_allow_html=True,
            )
            fig_sp_idle = build_kpi_sparkline_figure(
                kpis["sparkline_x"],
                kpis["idle_sparkline"],
                line_color="#2CA89A",
                fill_rgba="rgba(44, 168, 154, 0.15)",
                metric_name="Idle",
                height=72,
            )
            st.plotly_chart(fig_sp_idle, use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # --- Row 1.5: Individual Employee Details & Multi-Employee (`1..N`) Inspector ---
    if emp_profiles:
        with st.container(border=True):
            hdr_left, hdr_right = st.columns([3.0, 1.0])
            with hdr_left:
                st.markdown(
                    f'<div class="section-heading">👥 Individual Employee Details ({len(emp_profiles)} Tracked in {selected_video})</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-caption">Inspect per-employee presence windows, posture & activity shares, wrist speeds (SW/s), and focus any individual employee (`Emp #1` … `Emp #N`).</div>',
                    unsafe_allow_html=True,
                )
            with hdr_right:
                if selected_emp_id is not None:
                    if st.button("↩️ Show All Employees", key="reset_all_emp_btn", width="stretch"):
                        st.session_state["pending_emp_focus_label"] = "All Employees (Overview)"
                        st.rerun()

            cols_per_row = min(max(len(emp_profiles), 1), 3)
            for start_i in range(0, len(emp_profiles), cols_per_row):
                row_slice = emp_profiles[start_i : start_i + cols_per_row]
                card_cols = st.columns(cols_per_row, gap="small")
                for col_obj, prof in zip(card_cols, row_slice):
                    eid = prof["track_id"]
                    is_focused = selected_emp_id == eid
                    with col_obj:
                        with st.container(border=True):
                            st.markdown(
                                f"""
                                <div class="emp-card-hdr">
                                    <span class="emp-card-title">{'🎯 ' if is_focused else '👤 '}{prof['label']}</span>
                                    <span class="emp-state-badge">{prof['dominant_posture']} • {prof['dominant_activity']}</span>
                                </div>
                                <div class="emp-meta-line">
                                    Active: <b>{prof['duration_formatted']}</b> ({prof['active_frames']}f) • Seen: {prof['window_formatted']} • Zone: {prof['camera_zone']}
                                </div>
                                <div class="emp-stat-grid">
                                    <div class="emp-stat-box">
                                        <div class="emp-stat-val">{prof['working_pct']:.1f}%</div>
                                        <div class="emp-stat-lbl">Working</div>
                                    </div>
                                    <div class="emp-stat-box">
                                        <div class="emp-stat-val">{prof['idle_pct']:.1f}%</div>
                                        <div class="emp-stat-lbl">Idle</div>
                                    </div>
                                    <div class="emp-stat-box">
                                        <div class="emp-stat-val">{prof['away_pct']:.1f}%</div>
                                        <div class="emp-stat-lbl">Away</div>
                                    </div>
                                    <div class="emp-stat-box">
                                        <div class="emp-stat-val">{prof['sitting_pct']:.1f}%</div>
                                        <div class="emp-stat-lbl">Sitting</div>
                                    </div>
                                    <div class="emp-stat-box">
                                        <div class="emp-stat-val">{prof['standing_pct']:.1f}%</div>
                                        <div class="emp-stat-lbl">Standing</div>
                                    </div>
                                    <div class="emp-stat-box">
                                        <div class="emp-stat-val">{prof['walking_pct']:.1f}%</div>
                                        <div class="emp-stat-lbl">Walking</div>
                                    </div>
                                </div>
                                <div class="emp-meta-line">
                                    ⚡ Wrist Mean: <b>{prof['mean_wrist_sw_s']:.2f} SW/s</b> (L: {prof['left_wrist_sw_s']:.2f} / R: {prof['right_wrist_sw_s']:.2f}) • Peak: {prof['peak_wrist_sw_s']:.2f} SW/s • Transitions: {prof['posture_transitions']}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if not is_focused:
                                if st.button(
                                    f"🔍 View {prof['label']} Details",
                                    key=f"focus_btn_{selected_video}_{eid}",
                                    width="stretch",
                                ):
                                    st.session_state["pending_emp_focus_label"] = f"Emp #{eid}"
                                    st.rerun()

    st.write("")

    # --- Row 2: Ergonomic Timeline & Dual Utilization Breakdown ---
    with st.container(border=True):
        tab_gantt, tab_elev, tab_dual = st.tabs(
            [
                "🕒 Ergonomic Posture & Activity Timeline",
                "📏 Normalized Shoulder Elevation (Sitting vs. Standing)",
                "🍩 Posture vs. Utilization Side-by-Side",
            ]
        )
        with tab_gantt:
            st.markdown(
                '<div class="section-caption">Chronological state ribbons tracking Sitting, Standing, Walking, Away, Working, and Idle intervals across all selected employees.</div>',
                unsafe_allow_html=True,
            )
            fig_tl = build_ergonomic_timeline_figure(filtered_df, include_activity_ribbon=True, height=305)
            st.plotly_chart(fig_tl, use_container_width=True)

        with tab_elev:
            st.markdown(
                '<div class="section-caption">Scale-invariant vertical shoulder height ((H - Y) / H) vs. calibrated Standing threshold across all tracked employees.</div>',
                unsafe_allow_html=True,
            )
            fig_el = build_posture_elevation_figure(filtered_df, thresholds=thresholds, height=295)
            st.plotly_chart(fig_el, use_container_width=True)

        with tab_dual:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-heading">Spatial Posture Share</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    build_posture_donut_figure(kpis, height=360),
                    use_container_width=True,
                    key="dual_posture_donut",
                )
            with c2:
                st.markdown(
                    '<div class="section-heading">Workstation Utilization (Working vs. Idle vs. Away)</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    build_utilization_donut_figure(kpis, three_way_only=False, height=360),
                    use_container_width=True,
                    key="dual_util_donut",
                )

    st.write("")

    # --- Row 3: Wrist Velocity Activity Heatmap ---
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">🔥 Wrist Velocity Activity Heatmap</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Scale-invariant 30-frame rolling wrist velocity (SW/s) across Left, Right, and Dominant wrists for every tracked employee.</div>',
            unsafe_allow_html=True,
        )
        fig_hm = build_activity_heatmap_figure(
            filtered_df,
            bin_seconds=float(bin_sec),
            velocity_threshold_sw_s=vel_cutoff_sw,
            height=305,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    st.write("")

    # --- Row 4: Executive Employee Summary Table & Report Downloads ---
    col_tbl, col_export = st.columns([1.65, 1.0], gap="medium")

    summary_table = build_employee_summary_table(vid_df)

    with col_tbl:
        with st.container(border=True):
            st.markdown(
                f'<div class="section-heading">📋 Per-Employee Executive Telemetry Summary ({selected_video})</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                summary_table.to_pandas(),
                width="stretch",
                hide_index=True,
            )

    with col_export:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">📥 Executive Reports & Exports</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-caption">Download presentation-ready KPI summaries and executive PDF/PNG reports.</div>',
                unsafe_allow_html=True,
            )

            csv_bytes = summary_table.write_csv().encode("utf-8")
            st.download_button(
                label=f"⬇️ Download {Path(selected_video).stem} KPI Summary (.CSV)",
                data=csv_bytes,
                file_name=f"{Path(selected_video).stem}_executive_kpi_summary.csv",
                mime="text/csv",
                width="stretch",
            )

            pdf_path = REPORTS_OUTPUT_DIR / "executive_motion_report.pdf"
            png_path = REPORTS_OUTPUT_DIR / "executive_dashboard_snapshot.png"
            if not pdf_path.exists() or not png_path.exists():
                generate_executive_reports()

            if pdf_path.exists():
                st.download_button(
                    label="📄 Download 2-Page Executive Report (.PDF)",
                    data=pdf_path.read_bytes(),
                    file_name="executive_motion_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            if png_path.exists():
                st.download_button(
                    label="🖼️ Download Dashboard Snapshot (.PNG)",
                    data=png_path.read_bytes(),
                    file_name="executive_dashboard_snapshot.png",
                    mime="image/png",
                    width="stretch",
                )


if __name__ == "__main__":
    main()
