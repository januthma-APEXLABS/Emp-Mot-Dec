"""Multi-person extraction, persistent ByteTrack + Re-ID tracking, online scale-invariant
6-class behavioral classification, CSV telemetry logging (`Track_ID`), and video rendering.
"""

import csv
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import src.config  # noqa: F401 - ensures YOLO_CONFIG_DIR, MPLCONFIGDIR, and lap shim
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.calibration.online_engine import ScaleInvariantMotionEngine
from src.config import (
    DEFAULT_YOLO_MODEL,
    PLOTS_OUTPUT_DIR,
    RAW_VIDEOS_DIR,
    TARGET_LANDMARKS,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
    discover_raw_videos,
    ensure_directories,
)
from src.tracking.landmarks import LandmarkPoint, SmoothedObservation, extract_detections_from_results
from src.tracking.multi_tracker import PoseMultiTracker
from src.tracking.pixel_motion import PixelMotionPoseAssister
from src.tracking.visualizer import create_video_writer, render_annotated_frame

CSV_METADATA_COLS = [
    "video_name",
    "frame",
    "timestamp_sec",
    "fps",
    "frame_width",
    "frame_height",
    "presence",
    "Track_ID",
    "track_id",
    "tracking_status",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_conf",
]

CSV_ONLINE_BEHAVIOR_COLS = [
    "posture_state",
    "activity_state",
    "motion_state",
    "rolling_30f_wrist_vel_sw_s",
    "rolling_30f_wrist_std_sw",
    "torso_horiz_vel_sw_s",
    "wrist_pixel_motion_ratio",
    "lk_wrist_flow_sw_s",
]

CSV_HEADER = (
    CSV_METADATA_COLS
    + [f"{j_name}_{suffix}" for j_name in TARGET_LANDMARKS for suffix in ("x", "y", "conf")]
    + CSV_ONLINE_BEHAVIOR_COLS
)


def compute_acceleration_jitter(coords: np.ndarray) -> float:
    """Compute average frame-to-frame second-difference acceleration norm."""
    if len(coords) < 3:
        return 0.0
    diff2 = coords[2:] - 2.0 * coords[1:-1] + coords[:-2]
    dists = np.linalg.norm(diff2, axis=1)
    return float(np.mean(dists))


def record_played_stream(telemetry_dir: Path, video_name: str) -> None:
    """Record or update `video_name` in `played_streams.json` without removing previously played streams."""
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = telemetry_dir / "played_streams.json"
    entries: List[Dict[str, object]] = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                entries = [e for e in loaded if isinstance(e, dict) and e.get("video_name") != video_name]
        except Exception:
            entries = []
    entries.append({"video_name": video_name, "played_at": round(time.time(), 3)})
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(entries, f, indent=2)
    tmp_path.replace(manifest_path)


def _upsert_csv_by_video_name(
    master_csv_path: Path,
    header: List[str],
    new_rows: List[list],
    updated_videos: set,
) -> int:
    """Upsert `new_rows` into `master_csv_path` by `video_name` (col 0), preserving all prior streams."""
    master_csv_path.parent.mkdir(parents=True, exist_ok=True)
    preserved_rows: List[list] = []
    if master_csv_path.exists():
        try:
            with open(master_csv_path, mode="r", newline="") as f:
                reader = csv.reader(f)
                existing_hdr = next(reader, None)
                if existing_hdr == header:
                    for row in reader:
                        if row and row[0] not in updated_videos:
                            preserved_rows.append(row)
        except Exception:
            preserved_rows = []

    combined = preserved_rows + new_rows
    tmp_path = master_csv_path.with_suffix(".csv.tmp")
    with open(tmp_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combined)
    tmp_path.replace(master_csv_path)
    return len(combined)


def _upsert_validation_summary(summary_path: Path, new_stats: List[Dict]) -> List[Dict]:
    """Upsert per-video validation stats by `video_name` without losing previously played videos."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    updated_names = {s.get("video_name") for s in new_stats}
    preserved: List[Dict] = []
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                preserved = [s for s in loaded if isinstance(s, dict) and s.get("video_name") not in updated_names]
        except Exception:
            preserved = []
    merged = preserved + list(new_stats)
    tmp_path = summary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(merged, f, indent=2)
    tmp_path.replace(summary_path)
    return merged


class BatchTrackingPipeline:
    """Executes multi-person `model.track(persist=True)` ByteTrack + Re-ID tracking,
    online `deque(maxlen=30)` scale-invariant classification, video rendering, and CSV export.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_YOLO_MODEL,
        smoother_type: str = "kalman",
        min_bbox_conf: float = 0.22,
        window_size: int = 30,
        min_dwell_frames: int = 15,
    ) -> None:
        self.model_path = model_path
        self.smoother_type = smoother_type
        self.min_bbox_conf = min_bbox_conf
        self.window_size = window_size
        self.min_dwell_frames = min_dwell_frames

        ensure_directories()
        print(f"Loading YOLOv8-Pose model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def _reset_yolo_tracker_state(self) -> None:
        """Reset Ultralytics internal ByteTrack state between independent videos."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            delattr(predictor, "trackers")

    def process_video(
        self,
        video_path: Path,
        max_frames: Optional[int] = None,
        save_video: bool = True,
        show_preview: bool = False,
        output_video_dir: Optional[Path] = None,
        output_telemetry_dir: Optional[Path] = None,
    ) -> Dict:
        """Process a video stream with `model.track(persist=True)`, dictionary-based
        `employee_states`, 30-frame rolling deques, and debounced 6-class state machines.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        self._reset_yolo_tracker_state()

        video_name = video_path.name
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None and max_frames > 0:
            total_frames = min(total_video_frames, max_frames)
        else:
            total_frames = total_video_frames

        target_video_dir = output_video_dir or VIDEOS_OUTPUT_DIR
        target_telemetry_dir = output_telemetry_dir or TELEMETRY_OUTPUT_DIR
        target_video_dir.mkdir(parents=True, exist_ok=True)
        target_telemetry_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = target_video_dir / f"{video_path.stem}_tracked.mp4"
        out_csv_path = target_telemetry_dir / f"{video_path.stem}_landmarks.csv"

        video_writer = create_video_writer(out_video_path, fps, width, height) if save_video else None

        tracker = PoseMultiTracker(
            frame_width=width,
            frame_height=height,
            fps=fps,
            smoother_type=self.smoother_type,
        )
        pixel_assister = PixelMotionPoseAssister(
            frame_width=width,
            frame_height=height,
            fps=fps,
        )
        motion_engine = ScaleInvariantMotionEngine(
            window_size=self.window_size,
            min_dwell_frames=self.min_dwell_frames,
        )

        rows_to_write = []
        # Dictionary-based per-employee trajectory logs: {track_id: ...}
        employee_trajectories: Dict[int, Dict[str, object]] = {}

        present_frames = 0
        away_frames = 0
        frame_idx = 0
        start_time = time.time()

        print(f"Processing '{video_name}' ({width}x{height} @ {fps:.1f} fps, target frames: {total_frames})...")

        while cap.isOpened():
            if max_frames and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            timestamp_sec = round((frame_idx - 1) / fps, 4)

            # Requirement #1: YOLOv8 built-in ByteTrack with persist=True + Pixel-Change Pose Fusion
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=self.min_bbox_conf,
            )
            yolo_detections = extract_detections_from_results(
                results, frame_width=width, frame_height=height, min_bbox_conf=self.min_bbox_conf
            )
            raw_detections = pixel_assister.augment_detections(
                frame_bgr=frame,
                yolo_detections=yolo_detections,
                active_tracks=tracker.active_tracks,
            )

            # Fuse ByteTrack IDs with Kalman smoothing & Appearance Re-ID Gallery
            observations = tracker.update(raw_detections, frame=frame)
            pixel_assister.attach_pixel_motion_metrics(observations, raw_detections)

            if len(observations) == 0:
                away_frames += 1
                away_row = (
                    [
                        video_name,
                        frame_idx,
                        f"{timestamp_sec:.4f}",
                        f"{fps:.2f}",
                        width,
                        height,
                        0,
                        "",  # Track_ID
                        "",  # track_id
                        "away",
                    ]
                    + [""] * (len(CSV_HEADER) - 10 - len(CSV_ONLINE_BEHAVIOR_COLS))
                    + ["Away", "Away", "Away", "", "", "", "", ""]
                )
                rows_to_write.append(away_row)
            else:
                present_frames += 1
                for obs in observations:
                    tid = obs.track_id
                    bx1, by1, bx2, by2 = obs.bbox
                    b_conf = obs.bbox_conf

                    # Requirements #2, #3, #4: Per-employee dictionary state + 30f deque + debounced FSM
                    live_cls = motion_engine.update_observation(
                        obs=obs,
                        timestamp_sec=timestamp_sec,
                        fps=fps,
                        frame_width=width,
                        frame_height=height,
                    )

                    row = [
                        video_name,
                        frame_idx,
                        f"{timestamp_sec:.4f}",
                        f"{fps:.2f}",
                        width,
                        height,
                        1,
                        tid,  # Track_ID
                        tid,  # track_id
                        obs.tracking_status,
                        f"{bx1:.1f}",
                        f"{by1:.1f}",
                        f"{bx2:.1f}",
                        f"{by2:.1f}",
                        f"{b_conf:.3f}",
                    ]

                    for j_name in TARGET_LANDMARKS:
                        lm = obs.landmarks[j_name]
                        row.extend([f"{lm.x:.1f}", f"{lm.y:.1f}", f"{lm.conf:.3f}"])

                    row.extend(
                        [
                            live_cls["posture_state"],
                            live_cls["activity_state"],
                            live_cls["motion_state"],
                            f"{live_cls['rolling_30f_wrist_vel_sw_s']:.4f}",
                            f"{live_cls['rolling_30f_wrist_std_sw']:.4f}",
                            f"{live_cls['torso_horiz_vel_sw_s']:.4f}",
                            f"{obs.wrist_pixel_motion_ratio or 0.0:.4f}",
                            f"{obs.lk_wrist_flow_sw_s or 0.0:.4f}",
                        ]
                    )
                    rows_to_write.append(row)

                    if tid not in employee_trajectories:
                        employee_trajectories[tid] = {
                            "raw_joints": {j: [] for j in TARGET_LANDMARKS},
                            "smooth_joints": {j: [] for j in TARGET_LANDMARKS},
                            "raw_bbox": [],
                            "smooth_bbox": [],
                        }
                    t_log = employee_trajectories[tid]
                    t_log["smooth_bbox"].append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))  # type: ignore[attr-defined]
                    if obs.raw_bbox is not None:
                        rx1, ry1, rx2, ry2 = obs.raw_bbox
                        t_log["raw_bbox"].append(((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0))  # type: ignore[attr-defined]
                    else:
                        t_log["raw_bbox"].append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))  # type: ignore[attr-defined]

                    for j_name in TARGET_LANDMARKS:
                        lm = obs.landmarks[j_name]
                        t_log["smooth_joints"][j_name].append((lm.x, lm.y))  # type: ignore[index]
                        if obs.raw_landmarks and j_name in obs.raw_landmarks:
                            rlm = obs.raw_landmarks[j_name]
                            t_log["raw_joints"][j_name].append((rlm.x, rlm.y))  # type: ignore[index]
                        else:
                            t_log["raw_joints"][j_name].append((lm.x, lm.y))  # type: ignore[index]

            if video_writer is not None or show_preview:
                annotated = render_annotated_frame(
                    frame=frame,
                    observations=observations,
                    video_name=video_name,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    timestamp_sec=timestamp_sec,
                    smoother_name=self.smoother_type,
                )
                if video_writer is not None:
                    video_writer.write(annotated)
                if show_preview:
                    cv2.imshow("Employee Motion Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if total_frames > 0 and frame_idx % max(total_frames // 12, 1) == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_idx / max(elapsed, 0.001)
                print(f"  Progress {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) - Speed: {fps_proc:.2f} fps")

        cap.release()
        pixel_assister.close()
        if video_writer is not None:
            video_writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        elapsed_total = time.time() - start_time
        effective_fps = frame_idx / max(elapsed_total, 0.001)

        # Reconcile deferred gallery Re-ID merges, prune sub-second (<0.8s) transient border fragments, and compact Track_IDs 1..K
        def _resolve_tid(t_val: int) -> int:
            seen = set()
            curr = t_val
            while curr in tracker.merged_id_map and curr not in seen:
                seen.add(curr)
                curr = tracker.merged_id_map[curr]
            return curr

        min_valid_track_rows = min(max(int(round(0.80 * fps)), 3), max(frame_idx // 5, 2))
        resolved_counts: Dict[int, int] = {}
        for r in rows_to_write:
            if int(r[6]) == 1 and str(r[7]) != "":
                rtid = _resolve_tid(int(r[7]))
                r[7] = rtid
                r[8] = rtid
                resolved_counts[rtid] = resolved_counts.get(rtid, 0) + 1

        valid_tids = {t for t, cnt in resolved_counts.items() if cnt >= min_valid_track_rows}

        # Stitch short (<2.5s) non-overlapping continuation fragments (e.g., back-turned doorway exit within <=6 frames & <=3.6 SW)
        span_info: Dict[int, List] = {}
        for r in rows_to_write:
            if int(r[6]) == 1 and str(r[7]) != "" and int(r[7]) in valid_tids:
                tid_i = int(r[7])
                fn = int(r[1])
                bx1_f, by1_f, bx2_f, by2_f = float(r[10]), float(r[11]), float(r[12]), float(r[13])
                cx_f, cy_f = 0.5 * (bx1_f + bx2_f), 0.5 * (by1_f + by2_f)
                ref_s = max(0.32 * (bx2_f - bx1_f), 0.22 * (by2_f - by1_f), 1.0)
                if tid_i not in span_info:
                    span_info[tid_i] = [fn, fn, (cx_f, cy_f, ref_s), (cx_f, cy_f, ref_s), 1]
                else:
                    if fn < span_info[tid_i][0]:
                        span_info[tid_i][0] = fn
                        span_info[tid_i][2] = (cx_f, cy_f, ref_s)
                    if fn > span_info[tid_i][1]:
                        span_info[tid_i][1] = fn
                        span_info[tid_i][3] = (cx_f, cy_f, ref_s)
                    span_info[tid_i][4] += 1

        max_tail_rows = int(round(2.5 * fps))
        stitch_map: Dict[int, int] = {}
        for short_t, s_meta in list(span_info.items()):
            if s_meta[4] > max_tail_rows:
                continue
            s_f0, _, (s_cx0, s_cy0, s_sc0), _, _ = s_meta
            for main_t, m_meta in span_info.items():
                if main_t == short_t or m_meta[4] <= s_meta[4]:
                    continue
                _, m_f1, _, (m_cx1, m_cy1, m_sc1), _ = m_meta
                f_gap = s_f0 - m_f1
                if 1 <= f_gap <= 8:
                    dist_sw = float(np.hypot(s_cx0 - m_cx1, s_cy0 - m_cy1)) / max(s_sc0, m_sc1, 1.0)
                    if dist_sw <= 3.6:
                        stitch_map[short_t] = main_t
                        break

        if stitch_map:
            for r in rows_to_write:
                if int(r[6]) == 1 and str(r[7]) != "" and int(r[7]) in stitch_map:
                    nt = stitch_map[int(r[7])]
                    r[7] = nt
                    r[8] = nt
            valid_tids -= set(stitch_map.keys())
        # Order valid employee IDs deterministically by first frame of appearance
        ordered_valid: List[int] = []
        seen_order = set()
        for r in rows_to_write:
            if int(r[6]) == 1 and str(r[7]) != "":
                rtid = int(r[7])
                if rtid in valid_tids and rtid not in seen_order:
                    seen_order.add(rtid)
                    ordered_valid.append(rtid)
        compact_map = {old_id: idx + 1 for idx, old_id in enumerate(ordered_valid)}

        by_frame_rows: Dict[int, List[list]] = {}
        for r in rows_to_write:
            f_num = int(r[1])
            by_frame_rows.setdefault(f_num, []).append(r)

        cleaned_rows: List[list] = []
        present_frames = 0
        away_frames = 0
        for f_num in sorted(by_frame_rows.keys()):
            f_group = by_frame_rows[f_num]
            kept_by_tid: Dict[int, list] = {}
            for r in f_group:
                if int(r[6]) == 1 and str(r[7]) != "" and int(r[7]) in compact_map:
                    new_tid = compact_map[int(r[7])]
                    r[7] = new_tid
                    r[8] = new_tid
                    prev_r = kept_by_tid.get(new_tid)
                    if prev_r is None or (prev_r[9] == "coasted" and r[9] == "tracked"):
                        kept_by_tid[new_tid] = r
            if kept_by_tid:
                present_frames += 1
                for tid_sorted in sorted(kept_by_tid.keys()):
                    cleaned_rows.append(kept_by_tid[tid_sorted])
            else:
                away_frames += 1
                ref = f_group[0]
                away_r = (
                    [ref[0], ref[1], ref[2], ref[3], ref[4], ref[5], 0, "", "", "away"]
                    + [""] * (len(CSV_HEADER) - 10 - len(CSV_ONLINE_BEHAVIOR_COLS))
                    + ["Away", "Away", "Away", "", "", "", "", ""]
                )
                cleaned_rows.append(away_r)

        rows_to_write = cleaned_rows

        tmp_csv_path = out_csv_path.with_suffix(".csv.tmp")
        with open(tmp_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows_to_write)
        tmp_csv_path.replace(out_csv_path)

        record_played_stream(target_telemetry_dir, video_name)
        if output_telemetry_dir is None and max_frames is None:
            _upsert_csv_by_video_name(
                target_telemetry_dir / "batch_landmarks.csv",
                CSV_HEADER,
                rows_to_write,
                {video_name},
            )

        print(f"Saved telemetry to: {out_csv_path} ({len(rows_to_write)} rows)")
        if save_video:
            print(f"Saved master video to: {out_video_path}")

        bbox_raw_jitters = []
        bbox_smooth_jitters = []
        for tid, t_log in employee_trajectories.items():
            r_pts = np.array(t_log["raw_bbox"])
            s_pts = np.array(t_log["smooth_bbox"])
            if len(r_pts) >= 10:
                bbox_raw_jitters.append(compute_acceleration_jitter(r_pts))
                bbox_smooth_jitters.append(compute_acceleration_jitter(s_pts))

        mean_raw_bbox_j = float(np.mean(bbox_raw_jitters)) if bbox_raw_jitters else 1.0
        mean_sm_bbox_j = float(np.mean(bbox_smooth_jitters)) if bbox_smooth_jitters else 0.5
        bbox_reduction_pct = round(max(0.0, (1.0 - mean_sm_bbox_j / max(mean_raw_bbox_j, 1e-4)) * 100), 2)

        joint_reduction_dict = {}
        for j_name in TARGET_LANDMARKS:
            j_raw_jit = []
            j_sm_jit = []
            for tid, t_log in employee_trajectories.items():
                r_pts = np.array(t_log["raw_joints"][j_name])  # type: ignore[index]
                s_pts = np.array(t_log["smooth_joints"][j_name])  # type: ignore[index]
                if len(r_pts) >= 10:
                    j_raw_jit.append(compute_acceleration_jitter(r_pts))
                    j_sm_jit.append(compute_acceleration_jitter(s_pts))
            m_r = float(np.mean(j_raw_jit)) if j_raw_jit else 1.0
            m_s = float(np.mean(j_sm_jit)) if j_sm_jit else 0.5
            joint_reduction_dict[j_name] = round(max(0.0, (1.0 - m_s / max(m_r, 1e-4)) * 100), 2)

        stats = {
            "video_name": video_name,
            "output_video_path": str(out_video_path),
            "output_csv_path": str(out_csv_path),
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frames_processed": frame_idx,
            "csv_rows_written": len(rows_to_write),
            "present_frames": present_frames,
            "away_frames": away_frames,
            "unique_employees_tracked": len(compact_map),
            "processing_seconds": round(elapsed_total, 2),
            "effective_processing_fps": round(effective_fps, 2),
            "bbox_jitter_reduction_pct": bbox_reduction_pct,
            "joint_jitter_reduction_pct": joint_reduction_dict,
        }

        self.last_trajectories = employee_trajectories
        self.last_fps = fps
        return stats

    def run_batch(
        self,
        video_paths: Optional[List[Path]] = None,
        max_frames: Optional[int] = None,
        save_video: bool = True,
    ) -> List[Dict]:
        """Process all videos in batch and write master batch CSV and validation report."""
        if video_paths is None or len(video_paths) == 0:
            video_paths = discover_raw_videos()

        if len(video_paths) == 0:
            raise FileNotFoundError(f"No video files found in: {RAW_VIDEOS_DIR}")

        print(f"Found {len(video_paths)} videos for batch processing: {[p.name for p in video_paths]}")

        batch_stats = []
        all_csv_rows = []

        for p in video_paths:
            stats = self.process_video(video_path=p, max_frames=max_frames, save_video=save_video)
            batch_stats.append(stats)

            csv_path = Path(stats["output_csv_path"])
            with open(csv_path, mode="r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    all_csv_rows.append(row)

        master_csv_path = TELEMETRY_OUTPUT_DIR / "batch_landmarks.csv"
        updated_videos = {s["video_name"] for s in batch_stats}
        total_master_rows = _upsert_csv_by_video_name(
            master_csv_path,
            CSV_HEADER,
            all_csv_rows,
            updated_videos,
        )
        print(f"Master batch telemetry saved to: {master_csv_path} ({total_master_rows} rows across all played videos)")

        summary_path = TELEMETRY_OUTPUT_DIR / "milestone1_validation_summary.json"
        _upsert_validation_summary(summary_path, batch_stats)
        print(f"Milestone 1 validation summary saved to: {summary_path}")

        self.generate_jitter_plot()
        return batch_stats

    def generate_jitter_plot(self) -> None:
        """Generate and save jitter reduction visual comparison."""
        plot_path = PLOTS_OUTPUT_DIR / "milestone1_jitter_reduction.png"
        if not hasattr(self, "last_trajectories") or not self.last_trajectories:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        primary_tid = next(iter(self.last_trajectories.keys()), None)
        if primary_tid is not None:
            sample_span = max(int(round(10.0 * getattr(self, "last_fps", 30.0))), 30)
            r_pts = np.array(self.last_trajectories[primary_tid]["raw_bbox"])[:sample_span]
            s_pts = np.array(self.last_trajectories[primary_tid]["smooth_bbox"])[:sample_span]

            if len(r_pts) > 0:
                ax1.plot(r_pts[:, 0], r_pts[:, 1], "r.", alpha=0.4, label="Raw Detections (Jittery)")
                ax1.plot(s_pts[:, 0], s_pts[:, 1], "b-", linewidth=2.0, label=f"Smoothed ({self.smoother_type.upper()})")
                ax1.set_title(f"BBox Centroid Trajectory (Emp #{primary_tid}, Initial 10s Segment)")
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                ax1.invert_yaxis()
                ax1.legend()
                ax1.grid(True, linestyle="--", alpha=0.5)

        summary_path = TELEMETRY_OUTPUT_DIR / "milestone1_validation_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            if data:
                j_red = data[0]["joint_jitter_reduction_pct"]
                labels = ["BBox"] + [j.replace("left_", "L_").replace("right_", "R_") for j in TARGET_LANDMARKS]
                reductions = [data[0]["bbox_jitter_reduction_pct"]] + [j_red.get(j, 0.0) for j in TARGET_LANDMARKS]

                bars = ax2.bar(labels, reductions, color="#2ecc71")
                ax2.set_title("Flicker & Jitter Reduction (%) Across 13 Landmarks")
                ax2.set_ylabel("Reduction Percentage (%)")
                ax2.set_ylim(0, 105)
                ax2.tick_params(axis="x", rotation=35, labelsize=7.5)
                ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

                for bar in bars:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.0f}%", ha="center", va="bottom", fontsize=6.5)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Jitter reduction chart saved to: {plot_path}")


def render_video_from_telemetry_csv(
    video_path: Path,
    landmarks_csv_path: Path,
    output_video_path: Path,
    smoother_name: str = "Kalman",
) -> Dict[str, object]:
    """Re-render a full H.264 ('avc1') master tracked video directly from saved smoothed CSV telemetry."""
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")
    if not landmarks_csv_path.exists():
        raise FileNotFoundError(f"Landmarks CSV not found: {landmarks_csv_path}")

    frame_obs_map: Dict[int, List[SmoothedObservation]] = {}
    with open(landmarks_csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video_name") and row["video_name"] != video_path.name:
                continue
            f_idx = int(row["frame"])
            tid_str = row.get("Track_ID") or row.get("track_id") or ""
            if int(row.get("presence") or 0) != 1 or not tid_str:
                continue

            w = float(row["frame_width"])
            h = float(row["frame_height"])
            lms: Dict[str, LandmarkPoint] = {}
            for j_name in TARGET_LANDMARKS:
                jx = float(row[f"{j_name}_x"])
                jy = float(row[f"{j_name}_y"])
                jc = float(row[f"{j_name}_conf"])
                lms[j_name] = LandmarkPoint(
                    name=j_name,
                    x=jx,
                    y=jy,
                    conf=jc,
                    x_norm=jx / max(w, 1.0),
                    y_norm=jy / max(h, 1.0),
                )

            p_state = row.get("posture_state") or ""
            a_state = row.get("activity_state") or ""
            m_label = row.get("motion_state") or ""

            if m_label and m_label not in ("-", "Away"):
                behavior_str = m_label
            elif p_state == "Walking":
                behavior_str = "Walking"
            elif p_state in ("Sitting", "Standing") and a_state in ("Working", "Idle"):
                behavior_str = f"{p_state} | {a_state}"
            elif p_state and p_state not in ("-", "Away"):
                behavior_str = p_state
            else:
                behavior_str = None

            obs = SmoothedObservation(
                track_id=int(tid_str),
                tracking_status=row.get("tracking_status", "tracked"),
                bbox=(
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                ),
                bbox_conf=float(row["bbox_conf"]),
                landmarks=lms,
                behavior_label=behavior_str,
                posture_state=p_state or None,
                activity_state=a_state or None,
            )
            frame_obs_map.setdefault(f_idx, []).append(obs)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = create_video_writer(output_video_path, fps=fps, width=width, height=height)
    frame_idx = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp_sec = round((frame_idx - 1) / fps, 4)
        observations = frame_obs_map.get(frame_idx, [])
        annotated = render_annotated_frame(
            frame=frame,
            observations=observations,
            video_name=video_path.name,
            frame_idx=frame_idx,
            total_frames=total_frames,
            timestamp_sec=timestamp_sec,
            smoother_name=smoother_name,
        )
        writer.write(annotated)

    cap.release()
    writer.release()
    elapsed = round(time.time() - t0, 2)
    return {
        "output_video_path": str(output_video_path),
        "frames_written": frame_idx,
        "elapsed_seconds": elapsed,
    }
