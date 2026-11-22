"""Unit and integration tests for Milestone 3: Executive Dashboard & Reporting."""

from pathlib import Path
import tempfile
import unittest
import polars as pl

import src.config  # noqa: F401
from src.config import PROJECT_ROOT, TELEMETRY_OUTPUT_DIR
from src.dashboard.data_loader import (
    build_employee_summary_table,
    build_velocity_heatmap_data,
    compute_dashboard_kpis,
    extract_state_segments,
    filter_telemetry,
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
        """Execute `app.py` via Streamlit's native `AppTest` harness and verify zero runtime exceptions."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=20)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Streamlit app raised exception: {at.exception}")


if __name__ == "__main__":
    unittest.main()
