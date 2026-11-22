"""Unit and integration tests for Milestone 3: Executive Dashboard & Reporting."""

from pathlib import Path
import tempfile
import unittest
import polars as pl

import src.config  # noqa: F401
from src.config import PROJECT_ROOT, TELEMETRY_OUTPUT_DIR
from src.dashboard.data_loader import (
    build_employee_detail_profiles,
    build_employee_summary_table,
    build_velocity_heatmap_data,
    compute_dashboard_kpis,
    extract_state_segments,
    filter_telemetry,
    get_available_employees,
    get_available_videos,
    load_all_played_telemetry,
    load_calibrated_thresholds,
    load_telemetry_dataframe,
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


class TestMilestone3Dashboard(unittest.TestCase):
    """Verify Milestone 3 telemetry ingestion, KPI percentages, Plotly figures, exports, and Streamlit app."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_telemetry_dataframe(TELEMETRY_OUTPUT_DIR / "temporal_features.csv")
        cls.thresholds = load_calibrated_thresholds()

    def test_data_ingest_and_exact_kpi_percentages(self):
        """Verify KPI aggregation is video-agnostic, contains Track_ID, and sums cleanly to 100%."""
        self.assertIn("Track_ID", self.df.columns)
        kpis = compute_dashboard_kpis(self.df)
        self.assertGreater(kpis["total_rows"], 0)
        self.assertGreater(kpis["unique_frames"], 0)
        self.assertGreaterEqual(kpis["unique_employees"], 1)

        posture_sum = (
            kpis["sitting_pct"] + kpis["standing_pct"] + kpis["walking_pct"] + kpis["away_pct"]
        )
        self.assertAlmostEqual(posture_sum, 100.0, delta=0.3)

        activity_sum = (
            kpis["working_pct"] + kpis["idle_pct"] + kpis["walking_pct"] + kpis["away_pct"]
        )
        self.assertAlmostEqual(activity_sum, 100.0, delta=0.3)
        self.assertGreater(len(kpis["working_sparkline"]), 10)
        self.assertEqual(len(kpis["sparkline_x"]), len(kpis["working_sparkline"]))

    def test_employee_and_time_window_filtering(self):
        """Verify per-employee filtering builds complete 1-row-per-frame timelines for any video."""
        kpis_all = compute_dashboard_kpis(self.df)
        total_unique_frames = kpis_all["unique_frames"]

        for emp_id in kpis_all["employee_ids"]:
            emp_df = filter_telemetry(self.df, employee_id=emp_id)
            self.assertEqual(len(emp_df), total_unique_frames)
            k_emp = compute_dashboard_kpis(emp_df)
            self.assertAlmostEqual(
                k_emp["sitting_pct"] + k_emp["standing_pct"] + k_emp["walking_pct"] + k_emp["away_pct"],
                100.0,
                delta=0.3,
            )

        # Dynamic time-window slice around first active observation of primary employee
        primary_id = kpis_all["employee_ids"][0]
        pres_times = self.df.filter(pl.col("track_id") == primary_id)["timestamp_sec"]
        t_start = float(pres_times.min())
        t_end = min(float(pres_times.max()), t_start + 10.0)
        emp_win = filter_telemetry(self.df, employee_id=primary_id, time_range=(t_start, t_end))
        self.assertGreater(len(emp_win), 0)

    def test_plotly_figures_construct_cleanly(self):
        """Ensure all Plotly figures build valid traces without exceptions."""
        kpis = compute_dashboard_kpis(self.df)

        f_posture = build_posture_donut_figure(kpis)
        self.assertEqual(len(f_posture.data), 1)

        f_util4 = build_utilization_donut_figure(kpis, three_way_only=False)
        self.assertEqual(len(f_util4.data), 1)

        f_util3 = build_utilization_donut_figure(kpis, three_way_only=True)
        self.assertEqual(len(f_util3.data), 1)

        f_spark = build_kpi_sparkline_figure(kpis["sparkline_x"], kpis["working_sparkline"])
        self.assertEqual(len(f_spark.data), 1)

        segs = extract_state_segments(self.df, state_col="posture_state")
        self.assertGreater(len(segs), 5)

        f_tl = build_ergonomic_timeline_figure(self.df)
        self.assertGreater(len(f_tl.data), 5)

        f_el = build_posture_elevation_figure(self.df, thresholds=self.thresholds)
        self.assertGreaterEqual(len(f_el.data), 2)

        hm = build_velocity_heatmap_data(self.df, bin_seconds=2.0)
        self.assertEqual(len(hm["y_labels"]), len(hm["z_matrix"]))
        f_hm = build_activity_heatmap_figure(self.df, bin_seconds=2.0)
        self.assertEqual(len(f_hm.data), 1)

    def test_report_generation_in_isolated_tempdir(self):
        """Verify CSV, JSON, PNG snapshot, and PDF executive report generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = generate_executive_reports(
                csv_path=TELEMETRY_OUTPUT_DIR / "temporal_features.csv",
                output_dir=Path(tmp_dir),
            )
            for p_str in out.values():
                p = Path(p_str)
                self.assertTrue(p.exists(), f"Missing generated report file: {p}")
                self.assertGreater(p.stat().st_size, 200)

            # Check PDF magic bytes (%PDF) and PNG magic bytes
            self.assertTrue(Path(out["executive_pdf_path"]).read_bytes().startswith(b"%PDF"))
            self.assertEqual(Path(out["dashboard_png_path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            tbl = pl.read_csv(out["summary_csv_path"])
            expected_rows = 1 + compute_dashboard_kpis(self.df)["unique_employees"]
            self.assertEqual(len(tbl), expected_rows)

    def test_streamlit_app_headless_render(self):
        """Execute `app.py` via Streamlit's native `AppTest` harness across multi-video & 3-employee streams."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=20)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Streamlit app raised exception: {at.exception}")

        # Verify all played streams appear in 'Video Stream' selectbox (not just the last-played video!)
        vid_select = at.sidebar.selectbox(key="video_stream_selectbox")
        self.assertGreaterEqual(len(vid_select.options), 6, f"Expected all 6 played videos, got: {vid_select.options}")
        self.assertIn("Input_3.mp4", vid_select.options)
        self.assertIn("Input_6.mp4", vid_select.options)

        # Select 3-employee open-office stream `Input_3.mp4` and verify >2 employees in Employee Filter + Details!
        vid_select.select("Input_3.mp4").run()
        self.assertEqual(len(at.exception), 0, f"Switching to Input_3.mp4 failed: {at.exception}")
        emp_select = at.sidebar.selectbox(key="emp_filter_choice_Input_3.mp4")
        self.assertEqual(
            emp_select.options,
            ["All Employees (Overview)", "Emp #1", "Emp #2", "Emp #3"],
        )

        # Select Emp #3 specifically and verify detailed view renders cleanly with 0 exceptions
        emp_select.select("Emp #3").run()
        self.assertEqual(len(at.exception), 0, f"Selecting Emp #3 raised exception: {at.exception}")

    def test_cumulative_video_stream_discovery_and_4plus_employee_details(self):
        """Verify sequential single-video playback adds every stream immediately (including raw landmarks)
        and supports >2 employees (`Emp #1` .. `Emp #4`) in Employee Filter & detail profiles.
        """
        from src.calibration.rules_engine import run_milestone2_pipeline
        from src.tracking.pipeline import CSV_HEADER, record_played_stream

        with tempfile.TemporaryDirectory() as tmp_dir:
            t_dir = Path(tmp_dir) / "telemetry"
            p_dir = Path(tmp_dir) / "plots"
            t_dir.mkdir(parents=True, exist_ok=True)
            p_dir.mkdir(parents=True, exist_ok=True)

            def _write_mock_landmarks(vid_name: str, num_employees: int) -> Path:
                stem = Path(vid_name).stem
                out_csv = t_dir / f"{stem}_landmarks.csv"
                rows = []
                for f_idx in range(1, 36):
                    t_sec = round((f_idx - 1) / 30.0, 4)
                    for eid in range(1, num_employees + 1):
                        sx = 200.0 * eid
                        sy = 420.0
                        kpts = []
                        for _ in range(13):
                            kpts.extend([sx, sy, 0.92])
                        row = (
                            [vid_name, f_idx, t_sec, 30.0, 1920, 1080, 1, eid, eid, "tracked", sx - 60, 300.0, sx + 60, 720.0, 0.92]
                            + kpts
                            + ["Sitting", "Working", "Sitting | Working", 0.45, 0.04, 0.02, 0.05, 0.42]
                        )
                        rows.append(row)
                import csv
                with open(out_csv, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(CSV_HEADER)
                    writer.writerows(rows)
                record_played_stream(t_dir, vid_name)
                return out_csv

            # 1. Play Stream_A (2 employees) + run Milestone 2
            csv_a = _write_mock_landmarks("Stream_A.mp4", num_employees=2)
            run_milestone2_pipeline(input_csv_path=csv_a, output_telemetry_dir=t_dir, output_plots_dir=p_dir)

            # 2. Play Stream_B (4 employees) + run Milestone 2 (single-video call MUST NOT erase Stream_A!)
            csv_b = _write_mock_landmarks("Stream_B.mp4", num_employees=4)
            run_milestone2_pipeline(input_csv_path=csv_b, output_telemetry_dir=t_dir, output_plots_dir=p_dir)

            # 3. Play Stream_C (3 employees) in Milestone 1 ONLY (before Milestone 2 runs!)
            _write_mock_landmarks("Stream_C.mp4", num_employees=3)

            merged = load_all_played_telemetry(t_dir)
            streams = get_available_videos(merged, telemetry_dir=t_dir)
            self.assertEqual(streams, ["Stream_A.mp4", "Stream_B.mp4", "Stream_C.mp4"])

            # Verify temporal_features.csv itself retained both Stream_A and Stream_B via upsert
            tf_only = pl.read_csv(t_dir / "temporal_features.csv")
            self.assertEqual(sorted(tf_only["video_name"].unique().to_list()), ["Stream_A.mp4", "Stream_B.mp4"])

            # Verify 4-employee details on Stream_B
            emps_b = get_available_employees(merged, video_name="Stream_B.mp4")
            self.assertEqual(emps_b, [1, 2, 3, 4])
            profiles_b = build_employee_detail_profiles(merged, video_name="Stream_B.mp4")
            self.assertEqual([p["label"] for p in profiles_b], ["Emp #1", "Emp #2", "Emp #3", "Emp #4"])
            summary_b = build_employee_summary_table(merged.filter(pl.col("video_name") == "Stream_B.mp4"))
            self.assertEqual(len(summary_b), 5)


if __name__ == "__main__":
    unittest.main()
