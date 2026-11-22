"""CLI runner for Milestone 2: Heuristic Logic & Temporal Fine-Tuning."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.rules_engine import run_milestone2_pipeline
from src.config import PLOTS_OUTPUT_DIR, TELEMETRY_OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Milestone 2: Data-driven posture & wrist-velocity calibration and behavior rules engine."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(TELEMETRY_OUTPUT_DIR / "batch_landmarks.csv"),
        help="Input Milestone 1 landmark CSV file (default: outputs/telemetry/batch_landmarks.csv).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window length in frames for Euclidean wrist velocity (default: 30).",
    )
    parser.add_argument(
        "--dwell-frames",
        type=int,
        default=15,
        help="Minimum contiguous state dwell length in frames to suppress false-positive flicker (default: 15).",
    )
    parser.add_argument(
        "--thresholds-json",
        type=str,
        default=None,
        help="Optional pre-calibrated JSON file to apply instead of re-calibrating from input CSV.",
    )
    parser.add_argument(
        "--no-perspective-zones",
        action="store_true",
        help="Disable automatic camera-perspective zone posture calibration and use global threshold only.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input landmark CSV not found at {input_path}")
        sys.exit(1)

    override_path = Path(args.thresholds_json) if args.thresholds_json else None

    print("=================================================================")
    print("  Milestone 2: Heuristic Logic & Temporal Fine-Tuning")
    print(f"  Input CSV:          {input_path}")
    print(f"  Wrist Window:       {args.window} frames")
    print(f"  Anti-Flicker Dwell: {args.dwell_frames} frames")
    print("=================================================================\n")

    summary = run_milestone2_pipeline(
        input_csv_path=input_path,
        output_telemetry_dir=TELEMETRY_OUTPUT_DIR,
        output_plots_dir=PLOTS_OUTPUT_DIR,
        wrist_window_frames=args.window,
        min_dwell_frames=args.dwell_frames,
        thresholds_override_path=override_path,
        enable_perspective_zones=not args.no_perspective_zones,
    )

    ct = summary["calibrated_thresholds"]
    cm = summary["classification_metrics"]

    print("Calibrated Scale-Invariant Thresholds:")
    print(f"  - Global Normalized Posture Cutoff ((H - Avg_Sh_Y)/H): {ct['global_posture_threshold_norm']:.4f} H")
    if ct["use_perspective_zones"]:
        print(f"  - Perspective Zone Split (X / W):                      {ct['perspective_split_x_norm']:.4f} W")
        for z_name, z_cfg in ct["posture_zones"].items():
            if z_name == "global":
                continue
            print(
                f"    * {z_name.capitalize():10s} Zone: {z_cfg['threshold_norm']:.4f} H "
                f"(Seated med={z_cfg['seated_median_norm']:.4f}, Standing med={z_cfg['standing_median_norm']:.4f})"
            )
    print(
        f"  - Rolling {ct['wrist_window_frames']}f Wrist Speed Cutoff (Shoulder-Widths/s): "
        f"{ct['wrist_velocity_threshold_sw_s']:.4f} SW/s"
    )
    print(
        f"    * Schmitt Hysteresis Band:                           "
        f"[{ct['exit_to_idle_vel_sw_s']:.4f}, {ct['enter_working_vel_sw_s']:.4f}] SW/s"
    )

    print("\nBehavior Classification Summary (6-Class Taxonomy):")
    print(f"  - Total Telemetry Rows Labeled:  {cm['total_rows']}")
    print(f"  - Unique Employees Identified:   {cm['unique_tracks']}")
    print(f"  - 6-Class Counts:                {cm['six_class_summary']}")
    print(f"  - Spatial Posture Breakdown:     {cm['posture_distribution_rows']}")
    print(f"  - Workstation Activity Breakdown:{cm['activity_distribution_rows']}")
    print(f"  - Combined Motion Labels:        {cm['motion_state_distribution_rows']}")
    print(f"  - Posture Flickers Suppressed:   {cm['posture_flickers_suppressed']}")
    print(f"  - Activity Flickers Suppressed:  {cm['activity_flickers_suppressed']}")

    print("\nGenerated Deliverables:")
    print(f"  - Master Temporal Features CSV:  {summary['temporal_features_csv_path']}")
    print(f"  - Calibrated Thresholds JSON:    {summary['calibrated_thresholds_json_path']}")
    print(f"  - Posture Calibration Plot:      {summary['plots']['posture_calibration']}")
    print(f"  - Wrist Speed Calibration Plot:  {summary['plots']['wrist_velocity_calibration']}")
    print(f"  - Behavior State Timeline Plot:  {summary['plots']['state_timelines']}")
    print(f"  - 6-Class Summary Plot:          {summary['plots']['motion_categories_breakdown']}")


if __name__ == "__main__":
    main()
