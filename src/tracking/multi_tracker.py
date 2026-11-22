"""Multi-person persistent tracking fusing YOLOv8 ByteTrack IDs (`model.track(persist=True)`),
Kalman velocity-predicted Hungarian matching, and a scale-invariant Torso Appearance Re-ID Gallery.

Zero hardcoded pixel distances: all spatial association gates are normalized against
biological shoulder scale and frame diagonal geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import TARGET_LANDMARKS
from src.tracking.landmarks import (
    DetectionRaw,
    LandmarkPoint,
    SmoothedObservation,
    estimate_biological_scale,
)
from src.tracking.smoother import (
    KalmanBBoxSmoother,
    KalmanPoint2DSmoother,
    RollingAverageSmoother,
)


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    box_a_area = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union_area = box_a_area + box_b_area - inter_area
    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


def extract_torso_appearance_embedding(
    frame_bgr: Optional[np.ndarray],
    det: DetectionRaw,
) -> Optional[np.ndarray]:
    """Extract L2-normalized illumination-resilient HSV + CIE-Lab torso color histogram."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None

    x1, y1, x2, y2 = det.bbox
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    # Skip appearance only on thin border-clipped slivers (<6.5% width at frame edge) or tiny specks (<2% width)
    touches_lr_edge = (x1 / float(w)) < 0.02 or (x2 / float(w)) > 0.98
    if (bw / float(w)) < 0.020 or (touches_lr_edge and (bw / float(w)) < 0.065):
        return None
    bio_s = estimate_biological_scale(det.bbox, det.landmarks)
    pad = 0.07 * bio_s
    lms = det.landmarks

    ls = lms.get("left_shoulder")
    rs = lms.get("right_shoulder")
    lh = lms.get("left_hip")
    rh = lms.get("right_hip")

    if (
        ls is not None
        and rs is not None
        and lh is not None
        and rh is not None
        and ls.conf > 0.30
        and rs.conf > 0.30
        and lh.conf > 0.25
        and rh.conf > 0.25
    ):
        tx1 = int(max(0.0, min(ls.x, rs.x, lh.x, rh.x) - pad))
        tx2 = int(min(float(w), max(ls.x, rs.x, lh.x, rh.x) + pad))
        ty1 = int(max(0.0, min(ls.y, rs.y)))
        ty2 = int(min(float(h), max(lh.y, rh.y)))
    else:
        tx1 = int(max(0.0, x1 + 0.22 * bw))
        tx2 = int(min(float(w), x2 - 0.22 * bw))
        ty1 = int(max(0.0, y1 + 0.22 * bh))
        ty2 = int(min(float(h), y1 + 0.68 * bh))

    if (tx2 - tx1) < max(int(0.012 * w), 2) or (ty2 - ty1) < max(int(0.015 * h), 2):
        return None

    patch = cv2.resize(frame_bgr[ty1:ty2, tx1:tx2], (32, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

    h_hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256]).flatten()
    l_hist = cv2.calcHist([lab], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]).flatten()

    vec = np.concatenate([h_hist * 1.25, l_hist]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return None
    return vec / norm


@dataclass
class EmployeeGalleryProfile:
    """Persistent appearance & normalized spatial memory for an identified employee across occlusions."""
    track_id: int
    ema_embedding: Optional[np.ndarray] = None
    exemplars: List[np.ndarray] = field(default_factory=list)
    last_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_bio_scale: float = 1.0
    last_raw_bt_id: Optional[int] = None
    last_seen_step: int = 0
    total_hits: int = 0

    def update_appearance(
        self,
        emb: Optional[np.ndarray],
        bbox: Tuple[float, float, float, float],
        bio_scale: float,
        step: int,
        conf: float,
        raw_bt_id: Optional[int] = None,
    ) -> None:
        self.last_bbox = bbox
        self.last_bio_scale = max(bio_scale, 1e-3)
        if raw_bt_id is not None:
            self.last_raw_bt_id = raw_bt_id
        self.last_seen_step = step
        self.total_hits += 1
        if emb is None or conf < 0.50:
            return

        if self.ema_embedding is None:
            self.ema_embedding = emb.copy()
            self.exemplars.append(emb.copy())
            return

        merged = 0.90 * self.ema_embedding + 0.10 * emb
        nrm = float(np.linalg.norm(merged))
        if nrm > 1e-6:
            self.ema_embedding = merged / nrm

        if len(self.exemplars) < 5:
            max_ex_sim = max(float(np.dot(emb, ex)) for ex in self.exemplars)
            if max_ex_sim < 0.92:
                self.exemplars.append(emb.copy())

    def appearance_similarity(self, emb: Optional[np.ndarray]) -> float:
        if emb is None or self.ema_embedding is None:
            return 0.0
        sims = [float(np.dot(emb, self.ema_embedding))]
        for ex in self.exemplars:
            sims.append(float(np.dot(emb, ex)))
        return max(sims)


class SinglePersonTrack:
    """Represents a single persistent employee track across time."""

    def __init__(
        self,
        track_id: int,
        initial_detection: DetectionRaw,
        frame_width: int,
        frame_height: int,
        smoother_type: str = "kalman",
        dt: float = 1.0 / 30.0,
    ) -> None:
        self.track_id = track_id
        self.smoother_type = smoother_type
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.dt = dt

        self.time_since_update: int = 0
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 1

        self.last_bbox = initial_detection.bbox
        self.last_bbox_conf = initial_detection.bbox_conf
        self.last_landmarks = initial_detection.landmarks
        self.last_raw_bt_id: Optional[int] = initial_detection.raw_track_id

        if smoother_type == "kalman":
            self.bbox_smoother = KalmanBBoxSmoother(dt=dt)
            self.point_smoothers: Dict[str, KalmanPoint2DSmoother] = {
                name: KalmanPoint2DSmoother(dt=dt) for name in TARGET_LANDMARKS
            }
            self.bbox_smoother.update(*initial_detection.bbox, conf=initial_detection.bbox_conf)
            for name, lm in initial_detection.landmarks.items():
                if name in self.point_smoothers:
                    self.point_smoothers[name].update(lm.x, lm.y, lm.conf)
        else:
            self.rolling_smoother = RollingAverageSmoother(window_size=5)
            self.rolling_smoother.update_bbox(*initial_detection.bbox, conf=initial_detection.bbox_conf)
            for name, lm in initial_detection.landmarks.items():
                self.rolling_smoother.update_point(name, lm.x, lm.y, lm.conf)

    def get_predicted_matching_box(self) -> Tuple[float, float, float, float]:
        if self.smoother_type == "kalman" and self.time_since_update <= 6:
            pred = self.bbox_smoother.predict_bbox()
            if pred is not None:
                return pred
        return self.last_bbox

    def update(self, detection: DetectionRaw) -> None:
        was_long_occluded = self.time_since_update > 6
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.age += 1

        self.last_bbox = detection.bbox
        self.last_bbox_conf = detection.bbox_conf
        self.last_landmarks = detection.landmarks
        if detection.raw_track_id is not None:
            self.last_raw_bt_id = detection.raw_track_id

        if was_long_occluded:
            if self.smoother_type == "kalman":
                self.bbox_smoother.reset()
                self.bbox_smoother.update(*detection.bbox, conf=detection.bbox_conf)
                for name, sm in self.point_smoothers.items():
                    sm.reset()
                    if name in detection.landmarks:
                        lm = detection.landmarks[name]
                        sm.update(lm.x, lm.y, lm.conf)
            else:
                self.rolling_smoother.reset()
                self.rolling_smoother.update_bbox(*detection.bbox, conf=detection.bbox_conf)
                for name, lm in detection.landmarks.items():
                    self.rolling_smoother.update_point(name, lm.x, lm.y, lm.conf)

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.hit_streak = 0
        self.age += 1

    def get_smoothed_observation(self, is_coasted: bool = False) -> SmoothedObservation:
        status = "coasted" if is_coasted else "tracked"
        orig_raw_bbox = self.last_bbox

        if self.smoother_type == "kalman":
            if not is_coasted:
                smooth_x1, smooth_y1, smooth_x2, smooth_y2 = self.bbox_smoother.update(
                    *self.last_bbox, conf=self.last_bbox_conf
                )
            else:
                smooth_x1, smooth_y1, smooth_x2, smooth_y2 = self.bbox_smoother.update(
                    *self.last_bbox, conf=0.0
                )

            smooth_x1 = max(0.0, min(smooth_x1, self.frame_width))
            smooth_y1 = max(0.0, min(smooth_y1, self.frame_height))
            smooth_x2 = max(0.0, min(smooth_x2, self.frame_width))
            smooth_y2 = max(0.0, min(smooth_y2, self.frame_height))
            smoothed_bbox = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
            self.last_bbox = smoothed_bbox

            smoothed_landmarks: Dict[str, LandmarkPoint] = {}
            for name in TARGET_LANDMARKS:
                smoother = self.point_smoothers[name]
                if name in self.last_landmarks and not is_coasted:
                    raw_lm = self.last_landmarks[name]
                    sx, sy, sc = smoother.update(raw_lm.x, raw_lm.y, raw_lm.conf)
                else:
                    sx, sy, sc = smoother.update(0.0, 0.0, 0.0)

                sx = max(0.0, min(sx, self.frame_width))
                sy = max(0.0, min(sy, self.frame_height))
                nx = sx / self.frame_width if self.frame_width > 0 else 0.0
                ny = sy / self.frame_height if self.frame_height > 0 else 0.0
                smoothed_landmarks[name] = LandmarkPoint(
                    name=name, x=sx, y=sy, conf=sc, x_norm=nx, y_norm=ny
                )
        else:
            if not is_coasted:
                sb = self.rolling_smoother.update_bbox(*self.last_bbox, conf=self.last_bbox_conf)
            else:
                sb = self.last_bbox

            smoothed_bbox = (
                max(0.0, min(sb[0], self.frame_width)),
                max(0.0, min(sb[1], self.frame_height)),
                max(0.0, min(sb[2], self.frame_width)),
                max(0.0, min(sb[3], self.frame_height)),
            )
            smoothed_landmarks = {}
            for name in TARGET_LANDMARKS:
                if name in self.last_landmarks:
                    raw_lm = self.last_landmarks[name]
                    sx, sy, sc = self.rolling_smoother.update_point(name, raw_lm.x, raw_lm.y, raw_lm.conf)
                else:
                    sx, sy, sc = 0.0, 0.0, 0.0
                sx = max(0.0, min(sx, self.frame_width))
                sy = max(0.0, min(sy, self.frame_height))
                nx = sx / self.frame_width if self.frame_width > 0 else 0.0
                ny = sy / self.frame_height if self.frame_height > 0 else 0.0
                smoothed_landmarks[name] = LandmarkPoint(
                    name=name, x=sx, y=sy, conf=sc, x_norm=nx, y_norm=ny
                )

        return SmoothedObservation(
            track_id=self.track_id,
            tracking_status=status,
            bbox=smoothed_bbox,
            bbox_conf=self.last_bbox_conf,
            landmarks=smoothed_landmarks,
            raw_bbox=orig_raw_bbox,
            raw_landmarks=self.last_landmarks,
        )


class PoseMultiTracker:
    """Dictionary-backed multi-person tracker combining ByteTrack ID persistence,
    scale-invariant kinematic Hungarian matching, and Torso Appearance Re-ID.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = 30.0,
        smoother_type: str = "kalman",
        max_age_frames: int = 40,
        max_coast_render_frames: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.18,
        new_track_min_conf: float = 0.42,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.smoother_type = smoother_type
        self.max_age_frames = max_age_frames
        self.max_coast_render_frames = max_coast_render_frames
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.new_track_min_conf = new_track_min_conf

        self.step_idx: int = 0
        self.next_track_id: int = 1
        self._next_temp_id: int = -1
        self.tracks_by_id: Dict[int, SinglePersonTrack] = {}
        self.gallery: Dict[int, EmployeeGalleryProfile] = {}
        self.merged_id_map: Dict[int, int] = {}

    @property
    def active_tracks(self) -> List[SinglePersonTrack]:
        return list(self.tracks_by_id.values())

    def _is_clean_isolated_detection(self, det_idx: int, detections: List[DetectionRaw]) -> bool:
        """Ensure appearance gallery templates only update on unoccluded, non-border-clipped poses."""
        box_i = detections[det_idx].bbox
        fw = max(float(self.frame_width), 1.0)
        if (box_i[0] / fw) < 0.015 or (box_i[2] / fw) > 0.985:
            return False
        for j, other in enumerate(detections):
            if j != det_idx and compute_iou(box_i, other.bbox) > 0.12:
                return False
        return True

    def _match_gallery_reid(
        self,
        det: DetectionRaw,
        emb: Optional[np.ndarray],
        excluded_ids: set,
    ) -> Optional[int]:
        if not self.gallery:
            return None

        diag_len = max(float(np.hypot(self.frame_width, self.frame_height)), 1.0)
        d_cx = 0.5 * (det.bbox[0] + det.bbox[2])
        d_cy = 0.5 * (det.bbox[1] + det.bbox[3])
        det_scale = estimate_biological_scale(det.bbox, det.landmarks)

        best_tid: Optional[int] = None
        best_score = -1.0

        for tid, prof in self.gallery.items():
            if tid in excluded_ids:
                continue
            if prof.total_hits < 3:
                continue

            p_cx = 0.5 * (prof.last_bbox[0] + prof.last_bbox[2])
            p_cy = 0.5 * (prof.last_bbox[1] + prof.last_bbox[3])
            dist_raw = float(np.hypot(d_cx - p_cx, d_cy - p_cy))
            dist_diag_norm = dist_raw / diag_len
            ref_bio = max(det_scale, prof.last_bio_scale, 1e-3)
            dist_sw = dist_raw / ref_bio
            frames_absent = max(self.step_idx - prof.last_seen_step, 1)

            app_sim = prof.appearance_similarity(emb)
            bt_bonus = 0.08 if (det.raw_track_id is not None and det.raw_track_id == prof.last_raw_bt_id) else 0.0
            spatial_prox = float(np.exp(-dist_diag_norm / 0.45))
            recency_bonus = 0.06 * float(np.exp(-frames_absent / 240.0))

            # If this track is still actively alive (`time_since_update <= 25`), Stage 1 already evaluated it:
            # only reconnect in Stage 2 if appearance strongly matches (`>= 0.76`) so a nearby newcomer never steals its ID!
            live_trk = self.tracks_by_id.get(tid)
            if live_trk is not None and live_trk.time_since_update <= 25:
                if emb is not None and prof.ema_embedding is not None and app_sim >= 0.76 and dist_sw <= 2.5:
                    score = 0.82 * app_sim + 0.18 * spatial_prox + recency_bonus + bt_bonus
                    if score > best_score:
                        best_score = score
                        best_tid = tid
                continue

            if emb is not None and prof.ema_embedding is not None:
                if (
                    app_sim >= 0.72
                    or (frames_absent <= 120 and dist_sw <= 2.6 and app_sim >= 0.62)
                    or (frames_absent <= 750 and dist_sw <= 1.65 and app_sim >= 0.54)
                ):
                    score = 0.78 * app_sim + 0.20 * spatial_prox + recency_bonus + bt_bonus
                    if score > best_score:
                        best_score = score
                        best_tid = tid
            else:
                if frames_absent <= 45 and dist_sw <= 2.2:
                    score = spatial_prox + recency_bonus + bt_bonus
                    if score > best_score:
                        best_score = score
                        best_tid = tid

        return best_tid

    def update(
        self,
        detections: List[DetectionRaw],
        frame: Optional[np.ndarray] = None,
    ) -> List[SmoothedObservation]:
        self.step_idx += 1
        track_list = list(self.tracks_by_id.values())
        num_tracks = len(track_list)
        num_dets = len(detections)

        det_embeddings: List[Optional[np.ndarray]] = [
            extract_torso_appearance_embedding(frame, det) for det in detections
        ]
        det_scales: List[float] = [
            estimate_biological_scale(det.bbox, det.landmarks) for det in detections
        ]

        matched_track_indices = set()
        matched_det_indices = set()
        diag_len = max(float(np.hypot(self.frame_width, self.frame_height)), 1.0)

        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.full((num_tracks, num_dets), 10.0, dtype=np.float32)
            pred_boxes = [t.get_predicted_matching_box() for t in track_list]

            for t_idx, track in enumerate(track_list):
                t_box = pred_boxes[t_idx]
                t_cx = 0.5 * (t_box[0] + t_box[2])
                t_cy = 0.5 * (t_box[1] + t_box[3])
                prof = self.gallery.get(track.track_id)

                for d_idx, det in enumerate(detections):
                    d_box = det.bbox
                    d_cx = 0.5 * (d_box[0] + d_box[2])
                    d_cy = 0.5 * (d_box[1] + d_box[3])

                    iou = compute_iou(t_box, d_box)
                    dist_norm = float(np.hypot(t_cx - d_cx, t_cy - d_cy)) / diag_len
                    bt_match = (
                        det.raw_track_id is not None
                        and track.last_raw_bt_id is not None
                        and det.raw_track_id == track.last_raw_bt_id
                    )
                    target_match = det.target_track_id is not None and det.target_track_id == track.track_id
                    if det.source == "lk_hold" and not target_match:
                        continue
                    geom_cost = (
                        (1.0 - iou) * 0.60
                        + dist_norm * 0.40
                        - (0.18 if bt_match else 0.0)
                        - (0.35 if target_match else 0.0)
                    )

                    if prof is not None and det_embeddings[d_idx] is not None and prof.ema_embedding is not None:
                        app_sim = prof.appearance_similarity(det_embeddings[d_idx])
                        cost_matrix[t_idx, d_idx] = max(0.0, 0.65 * geom_cost + 0.35 * (1.0 - app_sim))
                    else:
                        cost_matrix[t_idx, d_idx] = max(0.0, geom_cost)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                track = track_list[r]
                det = detections[c]
                if det.source == "lk_hold" and det.target_track_id != track.track_id:
                    continue
                t_box = pred_boxes[r]
                iou = compute_iou(t_box, det.bbox)
                t_cx = 0.5 * (t_box[0] + t_box[2])
                t_cy = 0.5 * (t_box[1] + t_box[3])
                d_cx = 0.5 * (det.bbox[0] + det.bbox[2])
                d_cy = 0.5 * (det.bbox[1] + det.bbox[3])
                center_dist = float(np.hypot(t_cx - d_cx, t_cy - d_cy))
                center_diag_norm = center_dist / diag_len
                center_sw = center_dist / max(det_scales[c], 1e-3)

                prof = self.gallery.get(track.track_id)
                app_sim = prof.appearance_similarity(det_embeddings[c]) if prof else 0.0
                bt_same = (
                    det.raw_track_id is not None
                    and track.last_raw_bt_id is not None
                    and det.raw_track_id == track.last_raw_bt_id
                )
                target_same = det.target_track_id is not None and det.target_track_id == track.track_id

                has_app_check = (
                    prof is not None
                    and prof.ema_embedding is not None
                    and prof.total_hits >= 5
                    and det_embeddings[c] is not None
                    and det.source != "lk_hold"
                )
                # Prevent cross-occlusion ID hijack when a differently dressed person enters/crosses a track
                app_veto = False
                if has_app_check:
                    if app_sim < 0.58 and not (iou >= 0.75 and track.time_since_update == 0):
                        app_veto = True
                    elif (track.time_since_update >= 1 or iou < 0.35) and app_sim < 0.65:
                        app_veto = True

                # Physically bounded per-frame shift gate in biological shoulder-widths (prevents teleport jumps!)
                max_shift_sw = 0.65 + min(track.time_since_update * 0.18, 1.85)
                if not app_veto and (
                    iou >= self.iou_threshold
                    or target_same
                    or (bt_same and center_diag_norm < 0.15)
                    or center_sw < max_shift_sw
                    or (app_sim >= 0.78 and center_diag_norm < 0.22)
                ):
                    track.update(det)
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)
                    if track.track_id > 0:
                        if track.track_id not in self.gallery:
                            self.gallery[track.track_id] = EmployeeGalleryProfile(track_id=track.track_id)
                        emb_to_store = (
                            det_embeddings[c]
                            if (det.source != "lk_hold" and self._is_clean_isolated_detection(c, detections))
                            else None
                        )
                        self.gallery[track.track_id].update_appearance(
                            emb_to_store,
                            det.bbox,
                            det_scales[c],
                            self.step_idx,
                            det.bbox_conf,
                            raw_bt_id=det.raw_track_id,
                        )

        for r, trk in enumerate(track_list):
            if r not in matched_track_indices:
                trk.mark_missed()

        # Only exclude tracks ALREADY matched on this current frame so occluded/coasting tracks can immediately Re-ID!
        occupied_ids = {track_list[r].track_id for r in matched_track_indices if track_list[r].track_id > 0}

        for c in range(num_dets):
            if c in matched_det_indices:
                continue
            det = detections[c]
            if det.source == "lk_hold":
                continue
            emb = det_embeddings[c]

            reid_tid = self._match_gallery_reid(det, emb, excluded_ids=occupied_ids)
            if reid_tid is not None:
                occupied_ids.add(reid_tid)
                if reid_tid in self.tracks_by_id:
                    existing_trk = self.tracks_by_id[reid_tid]
                    existing_trk.update(det)
                else:
                    revived_track = SinglePersonTrack(
                        track_id=reid_tid,
                        initial_detection=det,
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                        smoother_type=self.smoother_type,
                        dt=1.0 / max(self.fps, 1.0),
                    )
                    revived_track.hits = self.gallery[reid_tid].total_hits + 1
                    self.tracks_by_id[reid_tid] = revived_track
                emb_to_store = emb if self._is_clean_isolated_detection(c, detections) else None
                self.gallery[reid_tid].update_appearance(
                    emb_to_store,
                    det.bbox,
                    det_scales[c],
                    self.step_idx,
                    det.bbox_conf,
                    raw_bt_id=det.raw_track_id,
                )
            elif det.bbox_conf >= self.new_track_min_conf:
                fw_f = max(float(self.frame_width), 1.0)
                fh_f = max(float(self.frame_height), 1.0)
                dx1, dy1, dx2, dy2 = det.bbox
                is_edge_sliver = (
                    ((dx1 / fw_f) < 0.02 and (dx2 / fw_f) < 0.15)
                    or ((dx2 / fw_f) > 0.98 and (dx1 / fw_f) > 0.85)
                    or ((dy1 / fh_f) > 0.65 and (dx2 / fw_f) > 0.85)
                )
                has_established_gallery = any(
                    p.total_hits >= 10 and p.ema_embedding is not None for p in self.gallery.values()
                )
                # Wait until torso enters frame enough to extract appearance before spawning a new ID
                if is_edge_sliver or (frame is not None and emb is None and has_established_gallery):
                    continue
                # Allocate temporary negative ID on frame > 1 when `frame` is present; promote to permanent ID on 2nd hit!
                use_probation = frame is not None and self.step_idx > 1
                if use_probation:
                    alloc_tid = self._next_temp_id
                    self._next_temp_id -= 1
                else:
                    alloc_tid = self.next_track_id
                    self.next_track_id += 1
                    occupied_ids.add(alloc_tid)
                new_track = SinglePersonTrack(
                    track_id=alloc_tid,
                    initial_detection=det,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                    smoother_type=self.smoother_type,
                    dt=1.0 / max(self.fps, 1.0),
                )
                self.tracks_by_id[alloc_tid] = new_track
                if alloc_tid > 0:
                    self.gallery[alloc_tid] = EmployeeGalleryProfile(track_id=alloc_tid)
                    emb_to_store = emb if self._is_clean_isolated_detection(c, detections) else None
                    self.gallery[alloc_tid].update_appearance(
                        emb_to_store,
                        det.bbox,
                        det_scales[c],
                        self.step_idx,
                        det.bbox_conf,
                        raw_bt_id=det.raw_track_id,
                    )

        # Promote probationary tracks once confirmed (`hits >= req_hits`), matching gallery first before minting a new ID!
        req_hits = max(self.min_hits, 4)
        for t_key in list(self.tracks_by_id.keys()):
            if t_key >= 0:
                continue
            p_trk = self.tracks_by_id[t_key]
            if p_trk.hits >= req_hits and p_trk.hit_streak >= min(req_hits - 1, 4) and p_trk.time_since_update == 0:
                p_det = DetectionRaw(
                    bbox=p_trk.last_bbox,
                    bbox_conf=p_trk.last_bbox_conf,
                    class_id=0,
                    landmarks=p_trk.last_landmarks,
                    raw_track_id=p_trk.last_raw_bt_id,
                )
                p_emb = extract_torso_appearance_embedding(frame, p_det)
                p_scale = estimate_biological_scale(p_det.bbox, p_det.landmarks)
                # Exclude ONLY tracks updated on this current step (`time_since_update == 0`)
                active_matched_ids = {k for k, v in self.tracks_by_id.items() if k > 0 and v.time_since_update == 0}
                reid_match = self._match_gallery_reid(p_det, p_emb, excluded_ids=active_matched_ids)
                if reid_match is not None:
                    perm_id = reid_match
                    p_trk.hits += self.gallery[perm_id].total_hits
                else:
                    perm_id = self.next_track_id
                    self.next_track_id += 1
                    self.gallery[perm_id] = EmployeeGalleryProfile(track_id=perm_id)
                p_trk.track_id = perm_id
                self.tracks_by_id[perm_id] = p_trk
                del self.tracks_by_id[t_key]
                self.gallery[perm_id].update_appearance(
                    p_emb,
                    p_det.bbox,
                    p_scale,
                    self.step_idx,
                    p_det.bbox_conf,
                    raw_bt_id=p_det.raw_track_id,
                )

        # Deferred gallery merge: if a newer track reveals a torso matching an un-matched veteran ID, merge!
        matched_ids_now = {k for k, v in self.tracks_by_id.items() if k > 0 and v.time_since_update == 0}
        for y_tid in list(matched_ids_now):
            y_trk = self.tracks_by_id.get(y_tid)
            y_prof = self.gallery.get(y_tid)
            if y_trk is None or y_prof is None or y_trk.hits > 90 or y_prof.ema_embedding is None:
                continue
            y_cx = 0.5 * (y_trk.last_bbox[0] + y_trk.last_bbox[2])
            y_cy = 0.5 * (y_trk.last_bbox[1] + y_trk.last_bbox[3])
            best_vet_tid: Optional[int] = None
            best_vet_score = 0.0
            for vet_tid, vet_prof in self.gallery.items():
                if vet_tid == y_tid or vet_tid in matched_ids_now:
                    continue
                if vet_prof.total_hits < 15 or vet_prof.ema_embedding is None or vet_tid > y_tid:
                    continue
                sim = vet_prof.appearance_similarity(y_prof.ema_embedding)
                v_cx = 0.5 * (vet_prof.last_bbox[0] + vet_prof.last_bbox[2])
                v_cy = 0.5 * (vet_prof.last_bbox[1] + vet_prof.last_bbox[3])
                d_sw = float(np.hypot(y_cx - v_cx, y_cy - v_cy)) / max(y_prof.last_bio_scale, vet_prof.last_bio_scale, 1e-3)
                vet_still_alive = vet_tid in self.tracks_by_id
                if (not vet_still_alive and sim >= 0.80 and d_sw <= 2.2) or (sim >= 0.65 and d_sw <= 1.15):
                    cand_sc = sim + (0.08 if d_sw <= 1.15 else 0.0)
                    if cand_sc > best_vet_score:
                        best_vet_score = cand_sc
                        best_vet_tid = vet_tid
            if best_vet_tid is not None:
                self.merged_id_map[y_tid] = best_vet_tid
                y_trk.track_id = best_vet_tid
                y_trk.hits += self.gallery[best_vet_tid].total_hits
                self.gallery[best_vet_tid].update_appearance(
                    y_prof.ema_embedding,
                    y_trk.last_bbox,
                    y_prof.last_bio_scale,
                    self.step_idx,
                    y_trk.last_bbox_conf,
                    raw_bt_id=y_trk.last_raw_bt_id,
                )
                self.tracks_by_id[best_vet_tid] = y_trk
                del self.tracks_by_id[y_tid]
                del self.gallery[y_tid]
                matched_ids_now.discard(y_tid)
                matched_ids_now.add(best_vet_tid)
                if self.next_track_id == y_tid + 1:
                    self.next_track_id = y_tid

        surviving: Dict[int, SinglePersonTrack] = {}
        observations: List[SmoothedObservation] = []

        for tid, track in self.tracks_by_id.items():
            max_allowed_miss = (2 if tid < 0 else 5) if track.hits < self.min_hits else self.max_age_frames
            if track.time_since_update <= max_allowed_miss:
                surviving[tid] = track
                if tid < 0:
                    if track.time_since_update == 0:
                        _ = track.get_smoothed_observation(is_coasted=False)
                    continue
                is_coasting = track.time_since_update > 0
                if not is_coasting:
                    observations.append(track.get_smoothed_observation(is_coasted=False))
                elif track.hits >= self.min_hits and track.time_since_update <= self.max_coast_render_frames:
                    observations.append(track.get_smoothed_observation(is_coasted=True))

        self.tracks_by_id = surviving
        observations.sort(key=lambda o: o.track_id)
        return observations
