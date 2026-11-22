# Employee Motion Tracker

Privacy-preserving computer vision system that extracts skeletal keypoints from office footage using a fine-tuned **YOLOv8n-Pose** model, applies confidence-weighted Kalman coordinate smoothing and data-calibrated geometric heuristics, and surfaces operational & ergonomic KPIs without storing raw video in telemetry reports.

## Project Structure

```text
employee-motion-tracker/
├── models/
│   └── best_yolov8n_pose.pt        # Fine-tuned YOLOv8n-Pose weights (COCO 17 keypoints)
├── data/
│   └── raw_videos/                 # Standardized input test videos (e.g., Input_1.mp4)
├── outputs/
│   ├── videos/                     # H.264 (avc1) annotated & smoothed tracking videos
│   ├── telemetry/                  # Landmark CSVs, temporal_features.csv & calibrated JSONs
│   ├── plots/                      # Jitter & threshold calibration charts
│   └── reports/                    # Executive KPI exports & standalone reports (Milestone 3)
├── src/
│   ├── config.py                   # Centralized paths & COCO 17-keypoint index map
│   ├── tracking/                   # Milestone 1: Batch tracking, landmark extraction & smoothing
│   ├── calibration/                # Milestone 2: Posture/activity calibration & rules engine
│   └── dashboard/                  # Milestone 3: Interactive executive KPI dashboard
├── scripts/
│   ├── run_milestone1.py           # CLI runner for Milestone 1 batch tracking & smoothing
│   ├── run_milestone2.py           # CLI runner for Milestone 2 calibration & state labeling
│   ├── test_yolo.py                # Local visual inference check for YOLOv8-Pose
│   └── legacy_mediapipe_poc.py     # Initial MediaPipe baseline proof-of-concept
├── tests/
│   ├── test_milestone1_tracking.py     # Unit & smoke tests for tracking & Kalman smoothing
│   ├── test_milestone2_calibration.py  # Unit & integration tests for calibration & rules engine
│   └── test_outputs_integrity.py       # Structural & output file integrity validation suite
├── requirements.txt
└── .gitignore
```

## Quick Start

Activate the local virtual environment:

```bash
source venv/bin/activate
```

Run Milestone 1 (spatial tracking & Kalman smoothing):

```bash
python scripts/run_milestone1.py --smoother kalman
```

Run Milestone 2 (posture & wrist-velocity calibration + behavior classification):

```bash
python scripts/run_milestone2.py
```

Run the complete test & output integrity suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
