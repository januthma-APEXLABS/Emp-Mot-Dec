"""Unit and integration tests for Milestone 2 & 6-Class Motion Categorization Rules Engine
(Sitting, Standing, Walking, Away, Idle, Working).
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import polars as pl

from src.calibration.calibrator import calibrate_thresholds_from_features, otsu_threshold_1d
from src.calibration.features import compute_temporal_features
from src.calibration.rules_engine import (
    BehaviorRulesEngine,
    SIX_MOTION_CLASSES,
    apply_hysteresis_and_dwell,
    run_milestone2_pipeline,
)
from src.config import TELEMETRY_OUTPUT_DIR


def _make_base_13_row(
    f: int,
    fps: float = 30.0,
    sh_x: float = 630.0,
    sh_y: float = 820.0,
    hip_x: float = 630.0,
    hip_y: float = 1030.0,
    lw_x: float = 570.0,
    lw_y: float = 910.0,
    rw_x: float = 690.0,
    rw_y: float = 910.0,
    nose_x: float = 630.0,
    nose_y: float = 735.0,
    eye_y: float = 720.0,
    ear_y: float = 725.0,
) -> dict:
    """Helper constructing a complete 13-keypoint observation row."""
    return {
        "video_name": "synthetic.mp4",
        "frame": f,
        "timestamp_sec": round((f - 1) / fps, 4),
        "fps": fps,
        "frame_width": 1920,
        "frame_height": 1080,
        "presence": 1,
        "track_id": 1,
        "tracking_status": "tracked",
        "bbox_x1": sh_x - 130.0,
        "bbox_y1": sh_y - 140.0,
        "bbox_x2": sh_x + 130.0,
        "bbox_y2": 1080.0,
        "bbox_conf": 0.92,
        "nose_x": nose_x,
        "nose_y": nose_y,
        "nose_conf": 0.92,
        "left_eye_x": nose_x - 16.0,
        "left_eye_y": eye_y,
        "left_eye_conf": 0.90,
        "right_eye_x": nose_x + 16.0,
        "right_eye_y": eye_y,
        "right_eye_conf": 0.90,
        "left_ear_x": sh_x - 36.0,
        "left_ear_y": ear_y,
        "left_ear_conf": 0.88,
        "right_ear_x": sh_x + 36.0,
        "right_ear_y": ear_y,
        "right_ear_conf": 0.88,
        "left_shoulder_x": sh_x - 60.0,
        "left_shoulder_y": sh_y,
        "left_shoulder_conf": 0.96,
        "right_shoulder_x": sh_x + 60.0,
        "right_shoulder_y": sh_y,
        "right_shoulder_conf": 0.96,
        "left_elbow_x": sh_x - 75.0,
        "left_elbow_y": sh_y + 70.0,
        "left_elbow_conf": 0.85,
        "right_elbow_x": sh_x + 75.0,
        "right_elbow_y": sh_y + 70.0,
        "right_elbow_conf": 0.85,
        "left_wrist_x": lw_x,
        "left_wrist_y": lw_y,
        "left_wrist_conf": 0.95,
        "right_wrist_x": rw_x,
        "right_wrist_y": rw_y,
        "right_wrist_conf": 0.95,
        "left_hip_x": hip_x - 45.0,
        "left_hip_y": hip_y,
        "left_hip_conf": 0.80,
        "right_hip_x": hip_x + 45.0,
        "right_hip_y": hip_y,
        "right_hip_conf": 0.80,
    }


def _make_away_row(f: int, fps: float = 30.0) -> dict:
    row = {
        "video_name": "synthetic.mp4",
        "frame": f,
        "timestamp_sec": round((f - 1) / fps, 4),
        "fps": fps,
        "frame_width": 1920,
        "frame_height": 1080,
        "presence": 0,
        "track_id": None,
        "tracking_status": "away",
        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None,
        "bbox_conf": None,
    }
    from src.config import TARGET_LANDMARKS

    for j in TARGET_LANDMARKS:
        row[f"{j}_x"] = None
        row[f"{j}_y"] = None
        row[f"{j}_conf"] = None
    return row


def _build_synthetic_landmark_df() -> pl.DataFrame:
    """Create deterministic synthetic telemetry with Sitting/Standing + Idle/Working phases."""
    rows = []
    fps = 30.0
    lw_x, rw_x = 570.0, 690.0
    for f in range(1, 181):
        sh_y = 920.0 if f <= 120 else 620.0
        hip_y = 1060.0 if f <= 120 else 850.0
        step = (0.1 if f % 2 == 0 else -0.1) if f <= 60 else (2.0 if f % 2 == 0 else -2.0)
        lw_x += step
        rw_x -= step
        rows.append(
            _make_base_13_row(
                f=f,
                fps=fps,
                sh_y=sh_y,
                hip_y=hip_y,
                lw_x=lw_x,
                lw_y=sh_y + 80.0,
                rw_x=rw_x,
                rw_y=sh_y + 80.0,
                nose_y=sh_y - 80.0,
                eye_y=sh_y - 95.0,
                ear_y=sh_y - 90.0,
            )
        )

    for f in range(181, 186):
        rows.append(_make_away_row(f=f, fps=fps))

    return pl.DataFrame(rows)


class TestMilestone2Calibration(unittest.TestCase):
    """Test suite verifying mathematical calibration, anti-flicker filtering, and all 6 motion classes."""

    def test_posture_distance_and_rolling_30f_velocity_math(self):
        raw_df = _build_synthetic_landmark_df()
        feat_df = compute_temporal_features(raw_df, wrist_window_frames=30)

        present = feat_df.filter(pl.col("presence") == 1)
        away = feat_df.filter(pl.col("presence") == 0)
        self.assertEqual(len(present), 180)
        self.assertEqual(len(away), 5)

        f10 = present.filter(pl.col("frame") == 10).row(0, named=True)
        self.assertAlmostEqual(f10["shoulder_dist_bottom_px"], 160.0, places=3)
        self.assertAlmostEqual(f10["shoulder_dist_bottom_norm"], 160.0 / 1080.0, places=4)

        f150 = present.filter(pl.col("frame") == 150).row(0, named=True)
        self.assertAlmostEqual(f150["shoulder_dist_bottom_px"], 460.0, places=3)

        f50 = present.filter(pl.col("frame") == 50).row(0, named=True)
        self.assertAlmostEqual(f50["rolling_30f_wrist_vel_px_s"], 3.0, places=2)

        f110 = present.filter(pl.col("frame") == 110).row(0, named=True)
        self.assertAlmostEqual(f110["rolling_30f_wrist_vel_px_s"], 60.0, places=2)

    def test_otsu_bimodal_threshold(self):
        rng = np.random.default_rng(123)
        bimodal = np.concatenate([rng.normal(155.0, 8.0, 500), rng.normal(455.0, 12.0, 300)])
        thresh, eta = otsu_threshold_1d(bimodal)
        self.assertGreater(thresh, 200.0)
        self.assertLess(thresh, 400.0)
        self.assertGreater(eta, 0.85)

    def test_hysteresis_and_dwell_eliminates_false_positives(self):
        signal = np.concatenate(
            [
                np.full(50, 4.0),
                np.full(4, 35.0),
                np.full(50, 4.0),
                np.full(50, 40.0),
                np.full(4, 2.0),
                np.full(50, 40.0),
            ]
        )
        labels, suppressed = apply_hysteresis_and_dwell(
            values=signal,
            low_thresh=12.0,
            high_thresh=15.0,
            low_label="Idle",
            high_label="Working",
            min_dwell_frames=15,
        )
        self.assertTrue(all(lbl == "Idle" for lbl in labels[:104]))
        self.assertTrue(all(lbl == "Working" for lbl in labels[104:]))
        transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
        self.assertEqual(transitions, 1)
        self.assertGreaterEqual(suppressed, 4)

    def test_synthetic_calibration_and_rules_engine(self):
        feat_df = compute_temporal_features(_build_synthetic_landmark_df(), wrist_window_frames=30)
        thresholds = calibrate_thresholds_from_features(
            feat_df, wrist_window_frames=30, min_dwell_frames=10, enable_perspective_zones=False
        )
        engine = BehaviorRulesEngine(thresholds)
        classified, metrics = engine.classify_dataframe(feat_df)

        self.assertIn("six_class_summary", metrics)
        for cls_name in SIX_MOTION_CLASSES:
            self.assertIn(cls_name, metrics["six_class_summary"])

        f30 = classified.filter(pl.col("frame") == 30).row(0, named=True)
        self.assertEqual(f30["posture_state"], "Sitting")
        self.assertEqual(f30["activity_state"], "Idle")
        self.assertEqual(f30["motion_state"], "Sitting | Idle")

        f105 = classified.filter(pl.col("frame") == 105).row(0, named=True)
        self.assertEqual(f105["posture_state"], "Sitting")
        self.assertEqual(f105["activity_state"], "Working")
        self.assertEqual(f105["motion_state"], "Sitting | Working")

        f160 = classified.filter(pl.col("frame") == 160).row(0, named=True)
        self.assertEqual(f160["posture_state"], "Standing")
        self.assertEqual(f160["activity_state"], "Working")
        self.assertEqual(f160["motion_state"], "Standing | Working")

        f183 = classified.filter(pl.col("frame") == 183).row(0, named=True)
        self.assertEqual(f183["posture_state"], "Away")
        self.assertEqual(f183["activity_state"], "Away")
        self.assertEqual(f183["motion_state"], "Away")

    def test_all_6_classes_and_all_subclauses(self):
        """Comprehensive multi-phase scenario testing all 6 classes and every user sub-clause:
        - Walking (rapid upright horizontal shift across frame)
        - Sitting chair scoot (horizontal shift while seated MUST remain Sitting, NOT Walking!)
        - Working (bilateral typing, unilateral dominant-wrist writing, static ear phone call, downward/locked-forward reading focus)
        - Idle (>5s prolonged zero wrist velocity AND repeated desk-to-face drinking transition)
        - Standing (maximized shoulder/hip vertical distance)
        - Away (zero keypoints detected)
        """
        rows = []
        fps = 30.0

        # Phase 1 (Frames 1..45): Upright Walking across frame (+6.2 px/frame = 186 px/s, span > 140 px)
        for f in range(1, 46):
            cx = 320.0 + 6.2 * f
            rows.append(
                _make_base_13_row(
                    f=f,
                    sh_x=cx,
                    sh_y=560.0,
                    hip_x=cx,
                    hip_y=800.0,
                    lw_x=cx - 60.0,
                    lw_y=780.0,
                    rw_x=cx + 60.0,
                    rw_y=780.0,
                    nose_x=cx,
                    nose_y=475.0,
                    eye_y=460.0,
                    ear_y=465.0,
                )
            )

        # Phase 2 (Frames 46..80): Seated rolling-chair adjustment (sh_y=830 -> Sitting, scooting horizontally!)
        # Must stay Sitting (never false-trigger Walking!)
        for f in range(46, 81):
            scoot_x = 600.0 + 3.5 * min(f - 46, 20)
            step = 2.2 if f % 2 == 0 else -2.2
            rows.append(
                _make_base_13_row(
                    f=f,
                    sh_x=scoot_x,
                    sh_y=830.0,
                    hip_x=scoot_x,
                    hip_y=1035.0,
                    lw_x=scoot_x - 60.0 + step,
                    lw_y=915.0,
                    rw_x=scoot_x + 60.0 - step,
                    rw_y=915.0,
                    nose_x=scoot_x,
                    nose_y=745.0,
                    eye_y=730.0,
                    ear_y=735.0,
                )
            )

        # Phase 3 (Frames 81..135): Seated Working Clause 1 — Bilateral Typing / Mouse Usage
        lw_x, rw_x = 610.0, 730.0
        for f in range(81, 136):
            step = 2.4 if f % 2 == 0 else -2.4
            lw_x += step
            rw_x -= step
            rows.append(_make_base_13_row(f=f, sh_x=670.0, hip_x=670.0, nose_x=670.0, lw_x=lw_x, rw_x=rw_x))

        # Phase 4 (Frames 136..190): Seated Working Clause 2 — Unilateral Writing (dominant right wrist rhythmic, left static)
        lw_x = 610.0
        rw_x = 730.0
        for f in range(136, 191):
            rw_x += 2.6 if f % 2 == 0 else -2.6
            rows.append(_make_base_13_row(f=f, sh_x=670.0, hip_x=670.0, nose_x=670.0, lw_x=lw_x, rw_x=rw_x))

        # Phase 5 (Frames 191..250): Seated Working Clause 3 — Phone Call (one wrist static near ear)
        for f in range(191, 251):
            rows.append(
                _make_base_13_row(
                    f=f,
                    sh_x=670.0,
                    hip_x=670.0,
                    nose_x=670.0,
                    lw_x=610.0,
                    lw_y=920.0,
                    rw_x=710.0,  # right_ear_x = 670 + 36 = 706, ear_y = 725 -> close & static
                    rw_y=727.0,
                )
            )

        # Phase 6 (Frames 251..315): Seated Working Clause 4 — Reading / Screen Focus (idle wrists + nose/eyes tilted downward)
        for f in range(251, 316):
            rows.append(
                _make_base_13_row(
                    f=f,
                    sh_x=670.0,
                    hip_x=670.0,
                    lw_x=610.0,
                    lw_y=920.0,
                    rw_x=730.0,
                    rw_y=920.0,
                    nose_x=670.0,
                    nose_y=762.0,
                    eye_y=724.0,  # downward pitch_r = 38 / 120 = 0.317 >= 0.22
                    ear_y=728.0,
                )
            )

        # Phase 7 (Frames 316..515): Seated Idle Clause 1 — Prolonged (>5s = 200 frames = 6.67s) zero wrist velocity while resting upright
        for f in range(316, 516):
            rows.append(
                _make_base_13_row(
                    f=f,
                    sh_x=670.0,
                    hip_x=670.0,
                    lw_x=610.0,
                    lw_y=920.0,
                    rw_x=730.0,
                    rw_y=920.0,
                    nose_x=692.0,  # casually looking aside, not downward or locked-forward
                    nose_y=734.0,
                    eye_y=722.0,
                    ear_y=726.0,
                )
            )

        # Phase 8 (Frames 516..570): Seated Active Working brief resume so we test immediate Drinking -> Idle transition!
        lw_x, rw_x = 610.0, 730.0
        for f in range(516, 571):
            step = 2.4 if f % 2 == 0 else -2.4
            lw_x += step
            rw_x -= step
            rows.append(_make_base_13_row(f=f, sh_x=670.0, hip_x=670.0, nose_x=670.0, lw_x=lw_x, rw_x=rw_x))

        # Phase 9 (Frames 571..635): Seated Idle Clause 2 — Desk-to-face wrist transition (Drinking, <5s duration)
        for f in range(571, 636):
            sip = np.sin(np.pi * (f - 571) / 64.0)
            rw_y_drink = 930.0 - 185.0 * sip
            rw_x_drink = 730.0 - 58.0 * sip
            rows.append(
                _make_base_13_row(
                    f=f,
                    sh_x=670.0,
                    hip_x=670.0,
                    nose_x=670.0,
                    lw_x=610.0,
                    lw_y=920.0,
                    rw_x=rw_x_drink,
                    rw_y=rw_y_drink,
                )
            )

        # Phase 10 (Frames 636..685): Standing + Working at workstation
        lw_x, rw_x = 610.0, 730.0
        for f in range(636, 686):
            step = 2.4 if f % 2 == 0 else -2.4
            lw_x += step
            rw_x -= step
            rows.append(
                _make_base_13_row(
                    f=f,
                    sh_x=670.0,
                    sh_y=560.0,
                    hip_x=670.0,
                    hip_y=800.0,
                    lw_x=lw_x,
                    lw_y=680.0,
                    rw_x=rw_x,
                    rw_y=680.0,
                    nose_x=670.0,
                    nose_y=475.0,
                    eye_y=460.0,
                    ear_y=465.0,
                )
            )

        # Phase 11 (Frames 686..700): Away (zero keypoints detected)
        for f in range(686, 701):
            rows.append(_make_away_row(f=f, fps=fps))

        df = pl.DataFrame(rows)
        feat_df = compute_temporal_features(df, wrist_window_frames=30)
        thresholds = calibrate_thresholds_from_features(
            feat_df, wrist_window_frames=30, min_dwell_frames=10, enable_perspective_zones=False
        )
        engine = BehaviorRulesEngine(thresholds)
        classified, _ = engine.classify_dataframe(feat_df)

        # 1. Walking
        r_walk = classified.filter(pl.col("frame") == 25).row(0, named=True)
        self.assertEqual(r_walk["posture_state"], "Walking")
        self.assertEqual(r_walk["motion_state"], "Walking")

        # 2. Seated rolling-chair shift MUST stay Sitting (not Walking)
        r_scoot = classified.filter(pl.col("frame") == 60).row(0, named=True)
        self.assertEqual(r_scoot["posture_state"], "Sitting")

        # 3. Working: Bilateral typing
        r_type = classified.filter(pl.col("frame") == 115).row(0, named=True)
        self.assertEqual(r_type["posture_state"], "Sitting")
        self.assertEqual(r_type["activity_state"], "Working")

        # 4. Working: Unilateral dominant-wrist writing
        r_write = classified.filter(pl.col("frame") == 170).row(0, named=True)
        self.assertEqual(r_write["activity_state"], "Working")

        # 5. Working: Static wrist near ear (Phone call)
        r_phone = classified.filter(pl.col("frame") == 230).row(0, named=True)
        self.assertEqual(r_phone["activity_state"], "Working")

        # 6. Working: Idle wrists + downward head tilt (Reading / screen focus)
        r_focus = classified.filter(pl.col("frame") == 290).row(0, named=True)
        self.assertEqual(r_focus["activity_state"], "Working")

        # 7. Idle: Prolonged >5s zero wrist velocity
        r_idle = classified.filter(pl.col("frame") == 490).row(0, named=True)
        self.assertEqual(r_idle["posture_state"], "Sitting")
        self.assertEqual(r_idle["activity_state"], "Idle")

        # 8. Idle: Desk-to-face wrist transition (Drinking)
        r_drink = classified.filter(pl.col("frame") == 603).row(0, named=True)
        self.assertEqual(r_drink["activity_state"], "Idle")

        # 9. Standing + Working
        r_stand = classified.filter(pl.col("frame") == 665).row(0, named=True)
        self.assertEqual(r_stand["posture_state"], "Standing")
        self.assertEqual(r_stand["activity_state"], "Working")

        # 10. Away
        r_away = classified.filter(pl.col("frame") == 695).row(0, named=True)
        self.assertEqual(r_away["posture_state"], "Away")
        self.assertEqual(r_away["activity_state"], "Away")
        self.assertEqual(r_away["motion_state"], "Away")

    def test_end_to_end_real_telemetry_pipeline(self):
        batch_csv = TELEMETRY_OUTPUT_DIR / "batch_landmarks.csv"
        if not batch_csv.exists():
            self.skipTest("Real batch_landmarks.csv not found.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary = run_milestone2_pipeline(
                input_csv_path=batch_csv,
                output_telemetry_dir=tmp_path / "telemetry",
                output_plots_dir=tmp_path / "plots",
                wrist_window_frames=30,
                min_dwell_frames=15,
            )

            self.assertTrue(Path(summary["temporal_features_csv_path"]).exists())
            self.assertTrue(Path(summary["calibrated_thresholds_json_path"]).exists())
            for plot_file in summary["plots"].values():
                self.assertTrue(Path(plot_file).exists())

            out_df = pl.read_csv(summary["temporal_features_csv_path"], infer_schema_length=15000)
            self.assertEqual(len(out_df), summary["classification_metrics"]["total_rows"])
            for col in (
                "Track_ID",
                "shoulder_dist_bottom_norm",
                "rolling_30f_wrist_vel_sw_s",
                "presence_state",
                "posture_state",
                "activity_state",
                "motion_state",
            ):
                self.assertIn(col, out_df.columns)

    def test_camera_scale_invariance_and_multi_employee_deques(self):
        """Verify scale-invariant math produces identical SW/s velocities and identical labels
        across near (1920x1080) vs far (640x360, 3x smaller pixels!) cameras, while isolating
        multiple simultaneous employees in `employee_states` with `deque(maxlen=30)` and 15f debouncing.
        """
        from src.calibration.online_engine import HysteresisStateMachine, ScaleInvariantMotionEngine
        from src.config import TARGET_LANDMARKS
        from src.tracking.landmarks import LandmarkPoint, SmoothedObservation

        # 1. Verify HysteresisStateMachine strictly requires >=15 continuous frames before flipping
        fsm = HysteresisStateMachine(initial_state="Sitting", min_dwell_frames=15)
        fsm.step("Sitting")
        # 14-frame transient pulse MUST NOT flip Sitting -> Standing
        for _ in range(14):
            self.assertEqual(fsm.step("Standing"), "Sitting")
        self.assertEqual(fsm.step("Sitting"), "Sitting")
        self.assertGreaterEqual(fsm.flickers_suppressed, 1)
        # 15 consecutive frames MUST flip Sitting -> Standing on frame 15
        for k in range(14):
            self.assertEqual(fsm.step("Standing"), "Sitting")
        self.assertEqual(fsm.step("Standing"), "Standing")

        # 2. Verify Near (1.0x) vs Far (0.333x) camera scale invariance + simultaneous Emp #1 & Emp #2 isolation
        def run_online_at_scale(scale: float):
            w, h = int(round(1920 * scale)), int(round(1080 * scale))
            engine = ScaleInvariantMotionEngine(window_size=30, min_dwell_frames=10, posture_threshold_norm=0.35)
            out_e1, out_e2 = [], []
            for idx in range(45):
                t_sec = idx / 30.0
                # Emp #1: Seated active typing (oscillating wrists)
                osc = (2.4 if idx % 2 == 0 else -2.4) * scale
                lms1 = {j: LandmarkPoint(j, 500 * scale, 750 * scale, 0.92) for j in TARGET_LANDMARKS}
                lms1["left_shoulder"] = LandmarkPoint("left_shoulder", 440 * scale, 820 * scale, 0.95)
                lms1["right_shoulder"] = LandmarkPoint("right_shoulder", 560 * scale, 820 * scale, 0.95)
                lms1["left_hip"] = LandmarkPoint("left_hip", 455 * scale, 1020 * scale, 0.90)
                lms1["right_hip"] = LandmarkPoint("right_hip", 545 * scale, 1020 * scale, 0.90)
                lms1["left_wrist"] = LandmarkPoint("left_wrist", 445 * scale + osc, 900 * scale, 0.95)
                lms1["right_wrist"] = LandmarkPoint("right_wrist", 555 * scale - osc, 900 * scale, 0.95)
                obs1 = SmoothedObservation(
                    track_id=1,
                    tracking_status="tracked",
                    bbox=(390 * scale, 700 * scale, 610 * scale, 1060 * scale),
                    bbox_conf=0.93,
                    landmarks=lms1,
                )

                # Emp #2 (simultaneous in same frame!): Standing completely idle & casually looking aside
                lms2 = {j: LandmarkPoint(j, 1200 * scale, 480 * scale, 0.92) for j in TARGET_LANDMARKS}
                lms2["nose"] = LandmarkPoint("nose", 1228 * scale, 482 * scale, 0.92)
                lms2["left_eye"] = LandmarkPoint("left_eye", 1185 * scale, 470 * scale, 0.90)
                lms2["right_eye"] = LandmarkPoint("right_eye", 1215 * scale, 470 * scale, 0.90)
                lms2["left_shoulder"] = LandmarkPoint("left_shoulder", 1140 * scale, 560 * scale, 0.95)
                lms2["right_shoulder"] = LandmarkPoint("right_shoulder", 1260 * scale, 560 * scale, 0.95)
                lms2["left_hip"] = LandmarkPoint("left_hip", 1155 * scale, 800 * scale, 0.90)
                lms2["right_hip"] = LandmarkPoint("right_hip", 1245 * scale, 800 * scale, 0.90)
                lms2["left_wrist"] = LandmarkPoint("left_wrist", 1135 * scale, 770 * scale, 0.95)
                lms2["right_wrist"] = LandmarkPoint("right_wrist", 1265 * scale, 770 * scale, 0.95)
                obs2 = SmoothedObservation(
                    track_id=2,
                    tracking_status="tracked",
                    bbox=(1090 * scale, 430 * scale, 1310 * scale, 1020 * scale),
                    bbox_conf=0.93,
                    landmarks=lms2,
                )

                out_e1.append(engine.update_observation(obs1, t_sec, 30.0, w, h))
                out_e2.append(engine.update_observation(obs2, t_sec, 30.0, w, h))

            self.assertEqual(set(engine.employee_states.keys()), {1, 2})
            self.assertEqual(engine.employee_states[1].history.maxlen, 30)
            self.assertEqual(len(engine.employee_states[1].history), 30)
            return out_e1[-1], out_e2[-1]

        near_e1, near_e2 = run_online_at_scale(1.0)
        far_e1, far_e2 = run_online_at_scale(1.0 / 3.0)

        # Identical normalized wrist speed (SW/s) and identical classifications across 3x scale change!
        self.assertAlmostEqual(
            near_e1["rolling_30f_wrist_vel_sw_s"],
            far_e1["rolling_30f_wrist_vel_sw_s"],
            places=3,
        )
        self.assertEqual(near_e1["motion_state"], "Sitting | Working")
        self.assertEqual(far_e1["motion_state"], "Sitting | Working")
        self.assertEqual(near_e2["motion_state"], "Standing | Idle")
        self.assertEqual(far_e2["motion_state"], "Standing | Idle")


if __name__ == "__main__":
    unittest.main()
