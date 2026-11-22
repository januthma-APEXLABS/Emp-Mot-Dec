"""Milestone 2: Heuristic Logic & Temporal Fine-Tuning package."""

from src.calibration.calibrator import (
    CalibratedThresholds,
    ZonePostureThreshold,
    calibrate_thresholds_from_features,
    generate_calibration_plots,
    kde_valley_or_otsu_1d,
    otsu_threshold_1d,
)
from src.calibration.features import compute_temporal_features, load_and_extract_features
from src.calibration.online_engine import (
    EmployeeMotionState,
    HysteresisStateMachine,
    ScaleInvariantMotionEngine,
)
from src.calibration.rules_engine import (
    BehaviorRulesEngine,
    apply_hysteresis_and_dwell,
    generate_state_timeline_plot,
    run_milestone2_pipeline,
)

__all__ = [
    "compute_temporal_features",
    "load_and_extract_features",
    "ZonePostureThreshold",
    "CalibratedThresholds",
    "otsu_threshold_1d",
    "kde_valley_or_otsu_1d",
    "calibrate_thresholds_from_features",
    "generate_calibration_plots",
    "apply_hysteresis_and_dwell",
    "BehaviorRulesEngine",
    "generate_state_timeline_plot",
    "run_milestone2_pipeline",
]
