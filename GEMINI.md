# GEMINI.md — Employee Motion Tracker Instructions

Project-level instructions and engineering constraints for developing, testing, and running the **Employee Motion Tracker** pipeline.

---

## 1. Core Generalization & Computer Vision Constraints (Mandatory)

1. **Zero Hardcoded Frames or Timestamps**
   - Never write logic tied to specific frame numbers, global indices, video filenames, or static timestamps (e.g., no `if frame_idx > 200:` or per-video branches).
   - Every algorithm must generalize cleanly to any unseen video uploaded to `data/raw_videos/`.

2. **Zero Absolute Pixel Thresholds (Scale-Invariant Math)**
   - Never use hardcoded pixel values for distances, velocities, or posture (e.g., no `if velocity > 50:`).
   - Normalize all spatial coordinates to `[0, 1]` frame proportions (`x / W`, `y / H`, `(H - sh_y) / H`) and all kinematic distances/velocities against each employee's dynamic biological shoulder scale $S_i$ (`SW` and `SW/s`, via `estimate_biological_scale()`).

3. **Multi-Person Tracking & Unique `Track_ID` Persistence**
   - Fuse Ultralytics `model.track(persist=True, tracker="bytetrack.yaml")` with `PoseMultiTracker` (`src/tracking/multi_tracker.py`), Kalman smoothing, and the HSV + CIE-Lab Torso Appearance Re-ID Gallery (`EmployeeGalleryProfile`).
   - Always log `Track_ID` in CSV telemetry (`src/tracking/pipeline.py`).

4. **Inter-Frame Pixel-Change Discovery, Non-Human Rejection & Stationary Persistence**
   - Use `PixelMotionPoseAssister` (`src/tracking/pixel_motion.py`) to detect moving employees missed by full-frame YOLO via illumination-compensated inter-frame pixel changes (`|I_t - I_{t-1} - median(ΔI)|`) and adaptive MAD noise thresholding.
   - Verify every candidate region anatomically (`mp.solutions.pose.Pose` + `is_plausible_human_pose()`) so non-employee moving pixels (monitor flicker, shadows, curtains, empty chairs) are rejected.
   - Retain motionless employees via torso patch Zero-Mean Normalized Cross-Correlation (`ZNCC`), Pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`), and periodic cropped pose re-anchoring—never drop a motionless seated employee to `Away`.
   - Suppress pose-network keypoint jitter on static limbs using `wrist_pixel_motion_ratio` and `lk_wrist_flow_sw_s`.

5. **Dictionary-Based State, Rolling Windows & Hysteresis**
   - Maintain isolated per-employee state dictionaries (`employee_states = {track_id: ...}`) in `src/calibration/online_engine.py`, `src/calibration/features.py`, and `src/calibration/rules_engine.py`.
   - Compute rolling kinematics over `collections.deque(maxlen=30)` windows.
   - Enforce minimum-dwell debouncing (`HysteresisStateMachine`, `min_dwell_frames=15`) across all 6 motion classes: `Sitting`, `Standing`, `Walking`, `Away`, `Idle`, `Working`.
   - Calibrate thresholds **per video** (`video_name`) and evaluate desk posture **per employee** so mixed camera angles (eye-level vs. overhead CCTV) never pollute each other.

---

## 2. Offline Environment & Model Rules

- **Python Virtual Environment**: Always use `./venv/bin/python` (never global `python`).
- **Offline `lap` Shim**: `src/config.py` provides a SciPy-backed `lap.lapjv` shim (`_ensure_lap_shim()`) so Ultralytics ByteTrack runs offline without C-extension builds. Always import `src.config` before `ultralytics`.
- **MediaPipe Model Complexity**: **Always use `model_complexity=1`** (`pose_landmark_full.tflite` is bundled locally in `venv`). Never use `model_complexity=0` or `2` (they attempt online downloads and fail offline).
- **Video Encoding**: Always encode output `.mp4` videos with H.264 (`avc1` preferred in `create_video_writer()`).

---

## 3. Standard Commands & Verification Workflow

- **Run Milestone 1 (Single Video or Full `data/raw_videos/` Batch)**:
  ```bash
  ./venv/bin/python scripts/run_milestone1.py --video data/raw_videos/<video>.mp4 --no-video
  # Or full batch:
  ./venv/bin/python scripts/run_milestone1.py --no-video
  ```
- **Run Milestone 2 (Calibration, 6-Class Rules Engine & Plots)**:
  ```bash
  ./venv/bin/python scripts/run_milestone2.py --input-csv outputs/telemetry/<stem>_landmarks.csv
  ```
- **Render Annotated H.264 Video & Verification Grid**:
  ```bash
  ./venv/bin/python scripts/render_classified_video.py --video data/raw_videos/<video>.mp4 --csv outputs/telemetry/<stem>_classified.csv
  ```
- **Generate Milestone 3 Executive Reports (CSV, JSON, PNG, PDF)**:
  ```bash
  ./venv/bin/python scripts/generate_report.py
  ```
- **Launch Interactive Streamlit Dashboard**:
  ```bash
  ./venv/bin/streamlit run app.py
  ```
- **Run Full Test Suite (Must Pass 100% After Any Code Edit)**:
  ```bash
  ./venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
  ```
  Keep unit tests isolated using `tempfile.TemporaryDirectory()` so production files in `outputs/` are never overwritten by tests.
