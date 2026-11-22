"""Legacy Stage-1 Baseline (Refactored to Scale-Invariant Shoulder-Width Math with Track_ID).

Note: Production multi-person tracking & classification lives in `src/tracking/` and
`src/calibration/` using YOLOv8-Pose `model.track(persist=True)` + `ScaleInvariantMotionEngine`.
"""

from collections import deque
import csv
import math
from pathlib import Path
import sys
from typing import Dict

import cv2
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.online_engine import HysteresisStateMachine
from src.config import (
    DEFAULT_INPUT_VIDEO,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
    ensure_directories,
)
from src.tracking.visualizer import create_video_writer


def main() -> None:
    ensure_directories()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    input_video_path = DEFAULT_INPUT_VIDEO
    output_video_path = VIDEOS_OUTPUT_DIR / "mediapipe_baseline_output.mp4"
    csv_report_path = TELEMETRY_OUTPUT_DIR / "mediapipe_baseline_report.csv"

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    out = create_video_writer(output_video_path, fps, width, height)

    # Dictionary-based state management with 30-frame rolling deque & 15-frame debouncers
    employee_states: Dict[int, Dict[str, object]] = {
        1: {
            "wrist_vel_sw_window": deque(maxlen=30),
            "prev_wrists": None,
            "posture_fsm": HysteresisStateMachine(initial_state="Sitting", min_dwell_frames=15),
            "activity_fsm": HysteresisStateMachine(initial_state="Idle", min_dwell_frames=15),
        }
    }

    with open(csv_report_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Timestamp_sec", "Track_ID", "Presence", "Posture", "Activity"])

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = round((frame_count - 1) / fps, 4)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            track_id_out = ""
            presence_state = "Away"
            posture_state = "Away"
            activity_state = "Away"

            if results.pose_landmarks:
                tid = 1
                track_id_out = str(tid)
                presence_state = "Present"
                st = employee_states[tid]
                landmarks = results.pose_landmarks.landmark

                ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                bio_scale = max(math.hypot(ls.x - rs.x, ls.y - rs.y), 1e-3)
                elev_norm = 1.0 - 0.5 * (ls.y + rs.y)
                prop_posture = "Standing" if elev_norm >= 0.45 else "Sitting"
                posture_state = st["posture_fsm"].step(prop_posture)  # type: ignore[attr-defined]

                curr_wrists = ((lw.x, lw.y), (rw.x, rw.y))
                if st["prev_wrists"] is not None:
                    plw, prw = st["prev_wrists"]  # type: ignore[misc]
                    dl_sw = math.hypot(lw.x - plw[0], lw.y - plw[1]) / bio_scale
                    dr_sw = math.hypot(rw.x - prw[0], rw.y - prw[1]) / bio_scale
                    st["wrist_vel_sw_window"].append(max(dl_sw, dr_sw) * fps)  # type: ignore[attr-defined]
                st["prev_wrists"] = curr_wrists

                vel_win = st["wrist_vel_sw_window"]  # type: ignore[assignment]
                mean_sw_s = (sum(vel_win) / len(vel_win)) if len(vel_win) > 0 else 0.0
                prop_act = "Working" if mean_sw_s >= 0.24 else "Idle"
                activity_state = st["activity_fsm"].step(prop_act)  # type: ignore[attr-defined]

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            writer.writerow([frame_count, timestamp, track_id_out, presence_state, posture_state, activity_state])
            out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
