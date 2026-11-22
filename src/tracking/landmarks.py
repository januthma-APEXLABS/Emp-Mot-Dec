"""Scale-invariant keypoint and bounding box extraction with pose-aware NMS deduplication.

Zero hardcoded pixel thresholds: all anatomical and spatial checks scale dynamically
with biological inter-shoulder distance and normalized frame geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import KEYPOINT_INDICES, TARGET_LANDMARKS


@dataclass
class LandmarkPoint:
    """Single landmark point in pixel and normalized [0, 1] coordinates."""
    name: str
    x: float
    y: float
    conf: float
    x_norm: float = 0.0
    y_norm: float = 0.0


@dataclass
class DetectionRaw:
    """Raw bounding box, optional ByteTrack ID, and keypoints detected by YOLOv8-Pose or Pixel-Motion Pose."""
    bbox: Tuple[float, float, float, float]
    bbox_conf: float
    class_id: int
    landmarks: Dict[str, LandmarkPoint] = field(default_factory=dict)
    raw_track_id: Optional[int] = None
    source: str = "yolo"
    target_track_id: Optional[int] = None


@dataclass
class SmoothedObservation:
    """Smoothed bounding box, target landmarks, pixel-motion telemetry, and live behavior tags."""
    track_id: int
    tracking_status: str  # "tracked" or "coasted"
    bbox: Tuple[float, float, float, float]
    bbox_conf: float
    landmarks: Dict[str, LandmarkPoint] = field(default_factory=dict)
    raw_bbox: Optional[Tuple[float, float, float, float]] = None
    raw_landmarks: Optional[Dict[str, LandmarkPoint]] = None
    behavior_label: Optional[str] = None
    posture_state: Optional[str] = None
    activity_state: Optional[str] = None
    wrist_pixel_motion_ratio: Optional[float] = None
    lk_wrist_flow_sw_s: Optional[float] = None


def estimate_biological_scale(
    bbox: Tuple[float, float, float, float],
    landmarks: Dict[str, LandmarkPoint],
) -> float:
    """Estimate instantaneous biological reference scale (shoulder breadth) without fixed pixel floors."""
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1e-3)
    bh = max(y2 - y1, 1e-3)

    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    sh_w = 0.0
    if ls is not None and rs is not None and ls.conf >= 0.20 and rs.conf >= 0.20:
        sh_w = float(np.hypot(ls.x - rs.x, ls.y - rs.y))

    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")
    spine_l = 0.0
    if (
        ls is not None
        and rs is not None
        and lh is not None
        and rh is not None
        and (ls.conf >= 0.20 or rs.conf >= 0.20)
        and (lh.conf >= 0.20 or rh.conf >= 0.20)
    ):
        sh_mx = 0.5 * (ls.x + rs.x)
        sh_my = 0.5 * (ls.y + rs.y)
        hp_mx = 0.5 * (lh.x + rh.x)
        hp_my = 0.5 * (lh.y + rh.y)
        spine_l = float(np.hypot(hp_mx - sh_mx, hp_my - sh_my))

    return max(sh_w, 0.68 * spine_l, 0.22 * bh, 0.32 * bw, 1e-3)


def _box_iou_and_containment(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Return (IoU, max_intersection_over_smaller_area) between two boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), 1e-6)
    area_b = max((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]), 1e-6)
    iou = inter / max(area_a + area_b - inter, 1e-6)
    contain = inter / min(area_a, area_b)
    return iou, contain


def is_plausible_human_pose(
    bbox: Tuple[float, float, float, float],
    landmarks: Dict[str, LandmarkPoint],
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> bool:
    """Reject degenerate non-human or border-fragment detections using scale-invariant ratios."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    fw = max(float(frame_width), 1.0)
    fh = max(float(frame_height), 1.0)

    if (bw / fw) < 0.022 or (bh / fh) < 0.11:
        return False

    if (y1 / fh) > 0.86:
        return False

    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    if ls is None or rs is None:
        return True

    if ls.conf < 0.20 and rs.conf < 0.20:
        return False

    sh_y = 0.5 * (ls.y + rs.y)
    if (sh_y / fh) > 0.977:
        return False

    bio_scale = estimate_biological_scale(bbox, landmarks)
    nose = landmarks.get("nose")
    if nose and nose.conf > 0.30 and (ls.conf > 0.25 or rs.conf > 0.25):
        if nose.y > sh_y + 0.28 * bio_scale:
            return False

    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")
    if lh and rh and lh.conf > 0.28 and rh.conf > 0.28:
        hip_x = 0.5 * (lh.x + rh.x)
        hip_y = 0.5 * (lh.y + rh.y)
        sh_x = 0.5 * (ls.x + rs.x)
        # Reject inverted or sideways (>54-deg horizontal) non-human chair/furniture hallucinations
        if (hip_y - sh_y) < 0.22 * bio_scale:
            return False
        if abs(hip_x - sh_x) > 1.38 * max(hip_y - sh_y, 1.0):
            return False
        if nose and nose.conf > 0.35 and nose.y > hip_y + 0.12 * bio_scale:
            return False

    return True


def deduplicate_detections(detections: List[DetectionRaw]) -> List[DetectionRaw]:
    """Suppress duplicate skeletons on the same person using IoU, containment, and shoulder-scale proximity."""
    if len(detections) <= 1:
        return detections

    # Prefer high-confidence, unclipped full-torso detections over small partial fragments
    ordered = sorted(
        detections,
        key=lambda d: d.bbox_conf * (1.0 + 0.15 * min(((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])) ** 0.5 / 120.0, 1.5)),
        reverse=True,
    )
    kept: List[DetectionRaw] = []

    for cand in ordered:
        is_dup = False
        c_ls = cand.landmarks.get("left_shoulder")
        c_rs = cand.landmarks.get("right_shoulder")
        c_has_sh = c_ls is not None and c_rs is not None and (c_ls.conf > 0.25 or c_rs.conf > 0.25)
        c_sh_x = 0.5 * (c_ls.x + c_rs.x) if c_has_sh else 0.5 * (cand.bbox[0] + cand.bbox[2])
        c_sh_y = 0.5 * (c_ls.y + c_rs.y) if c_has_sh else 0.5 * (cand.bbox[1] + cand.bbox[3])
        c_scale = estimate_biological_scale(cand.bbox, cand.landmarks)

        for acc in kept:
            iou, contain = _box_iou_and_containment(cand.bbox, acc.bbox)
            if iou >= 0.38 or contain >= 0.54:
                is_dup = True
                break

            a_ls = acc.landmarks.get("left_shoulder")
            a_rs = acc.landmarks.get("right_shoulder")
            a_has_sh = a_ls is not None and a_rs is not None and (a_ls.conf > 0.25 or a_rs.conf > 0.25)
            if c_has_sh and a_has_sh:
                a_sh_x = 0.5 * (a_ls.x + a_rs.x)
                a_sh_y = 0.5 * (a_ls.y + a_rs.y)
                a_scale = estimate_biological_scale(acc.bbox, acc.landmarks)
                ref_scale = max(c_scale, a_scale, 1e-3)
                if (float(np.hypot(c_sh_x - a_sh_x, c_sh_y - a_sh_y)) / ref_scale) < 0.65:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(cand)

    return kept


def extract_detections_from_results(
    results,
    frame_width: int,
    frame_height: int,
    min_bbox_conf: float = 0.22,
) -> List[DetectionRaw]:
    """Extract deduplicated person boxes, ByteTrack IDs, and 13 landmarks from Ultralytics Results."""
    detections: List[DetectionRaw] = []
    if results is None or len(results) == 0:
        return detections

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return detections

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()
    raw_ids = res.boxes.id.int().cpu().numpy() if res.boxes.id is not None else None

    has_kpts = res.keypoints is not None and len(res.keypoints.data) > 0
    kpts_data = res.keypoints.data.cpu().numpy() if has_kpts else None

    fw = max(float(frame_width), 1.0)
    fh = max(float(frame_height), 1.0)

    for i in range(len(boxes)):
        bbox_conf = float(confs[i])
        if bbox_conf < min_bbox_conf:
            continue

        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        cls_id = int(clss[i])
        bt_id = int(raw_ids[i]) if (raw_ids is not None and i < len(raw_ids)) else None

        landmarks_dict: Dict[str, LandmarkPoint] = {}
        if kpts_data is not None and i < len(kpts_data):
            inst_kpts = kpts_data[i]
            for name in TARGET_LANDMARKS:
                idx = KEYPOINT_INDICES[name]
                if idx < len(inst_kpts):
                    kx = float(inst_kpts[idx, 0])
                    ky = float(inst_kpts[idx, 1])
                    kc = float(inst_kpts[idx, 2])
                    kx_norm = min(max(kx / fw, 0.0), 1.0)
                    ky_norm = min(max(ky / fh, 0.0), 1.0)
                    landmarks_dict[name] = LandmarkPoint(
                        name=name, x=kx, y=ky, conf=kc, x_norm=kx_norm, y_norm=ky_norm
                    )
                else:
                    landmarks_dict[name] = LandmarkPoint(name=name, x=0.0, y=0.0, conf=0.0)
        else:
            for name in TARGET_LANDMARKS:
                landmarks_dict[name] = LandmarkPoint(name=name, x=0.0, y=0.0, conf=0.0)

        if not is_plausible_human_pose(
            (x1, y1, x2, y2), landmarks_dict, frame_width=frame_width, frame_height=frame_height
        ):
            continue

        detections.append(
            DetectionRaw(
                bbox=(x1, y1, x2, y2),
                bbox_conf=bbox_conf,
                class_id=cls_id,
                landmarks=landmarks_dict,
                raw_track_id=bt_id,
            )
        )

    return deduplicate_detections(detections)
