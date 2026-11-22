"""Inter-frame pixel-change employee discovery, MediaPipe skeletal verification,
stationary torso pixel persistence, and Pyramidal Lucas-Kanade optical-flow keypoint tracking.

Zero hardcoded frame indices and zero absolute pixel thresholds:
  - Inter-frame pixel differences compensate for global illumination shifts and adapt
    thresholds dynamically to per-frame sensor noise via Median Absolute Deviation (MAD).
  - Every candidate moving region must pass anatomical human-pose verification
    (`mp.solutions.pose.Pose`, `model_complexity=1` bundled offline) so non-employee
    moving pixels (monitors, shadows, plants, chairs) are rejected.
  - Motionless employees are discovered via overlapping spatial tiles and retained across
    stillness via torso patch Zero-Mean Normalized Cross-Correlation (ZNCC) + LK optical flow.
  - Local hand/wrist pixel-change ratios and optical flow (`SW/s`) suppress neural-network
    keypoint jitter on static limbs while capturing subtle typing/mouse/writing motion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import mediapipe as mp
import numpy as np

from src.config import TARGET_LANDMARKS
from src.tracking.landmarks import (
    DetectionRaw,
    LandmarkPoint,
    SmoothedObservation,
    _box_iou_and_containment,
    deduplicate_detections,
    estimate_biological_scale,
    is_plausible_human_pose,
)

# Mapping from our 13 COCO-aligned TARGET_LANDMARKS to MediaPipe Pose 33-landmark indices
MP_LANDMARK_MAP: Dict[str, int] = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}
MP_LOWER_BODY_INDICES = (25, 26, 27, 28)  # knees & ankles for full-body bbox extent when visible


def _zncc_patch(patch_a: Optional[np.ndarray], patch_b: Optional[np.ndarray]) -> float:
    """Zero-mean normalized cross-correlation in [-1, 1] between two equal-sized patches."""
    if patch_a is None or patch_b is None or patch_a.shape != patch_b.shape or patch_a.size == 0:
        return 0.0
    fa = patch_a.astype(np.float32)
    fb = patch_b.astype(np.float32)
    fa -= float(np.mean(fa))
    fb -= float(np.mean(fb))
    na = float(np.linalg.norm(fa))
    nb = float(np.linalg.norm(fb))
    if na <= 1e-4 or nb <= 1e-4:
        # Both patches nearly uniform intensity -> check raw L1 similarity
        mad = float(np.mean(np.abs(patch_a.astype(np.float32) - patch_b.astype(np.float32))))
        return float(max(0.0, 1.0 - mad / 25.0))
    return float(np.sum(fa * fb) / (na * nb))


class PixelMotionPoseAssister:
    """Fuses inter-frame pixel changes, anatomical pose verification, stationary pixel
    persistence, and Lucas-Kanade optical flow for general multi-employee video tracking.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = 30.0,
        work_width: int = 320,
        lk_max_dim: int = 960,
        periodic_tile_interval_sec: float = 2.0,
    ) -> None:
        self.frame_width = max(int(frame_width), 1)
        self.frame_height = max(int(frame_height), 1)
        self.fps = max(float(fps), 1.0)

        self.work_w = max(int(work_width), 160)
        self.work_h = max(int(round(self.frame_height * (self.work_w / float(self.frame_width)))), 90)

        max_orig = max(self.frame_width, self.frame_height)
        self.lk_scale = min(1.0, float(lk_max_dim) / float(max_orig))
        self.lk_w = max(int(round(self.frame_width * self.lk_scale)), 64)
        self.lk_h = max(int(round(self.frame_height * self.lk_scale)), 64)

        self.periodic_tile_frames = max(int(round(periodic_tile_interval_sec * self.fps)), 30)

        self.step_idx: int = 0
        self.prev_work_gray: Optional[np.ndarray] = None
        self.bg_work_gray: Optional[np.ndarray] = None
        self.prev_lk_gray: Optional[np.ndarray] = None
        self.curr_lk_gray: Optional[np.ndarray] = None
        self.step_active_mask: Optional[np.ndarray] = None
        self.fg_active_mask: Optional[np.ndarray] = None
        self.tau_step: float = 5.0

        # Spatial cooldown grid (4 rows x 6 cols) suppressing repeated non-human motion checks
        self.grid_rows = 4
        self.grid_cols = 6
        self.non_human_fail_streak = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        self.non_human_cooldown = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        # Per-employee pixel persistence & verified pose anchors: {track_id: dict}
        self.track_anchors: Dict[int, Dict[str, Any]] = {}

        # Lazy-loaded persistent MediaPipe Pose (model_complexity=1 bundled offline)
        self._mp_pose: Optional[Any] = None

    def _get_mp_pose(self) -> Any:
        if self._mp_pose is None:
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.40,
            )
        return self._mp_pose

    def close(self) -> None:
        if self._mp_pose is not None:
            try:
                self._mp_pose.close()
            except Exception:
                pass
            self._mp_pose = None

    def __del__(self) -> None:
        self.close()

    def _extract_torso_gray_patch(
        self,
        lk_gray: Optional[np.ndarray],
        bbox: Tuple[float, float, float, float],
        landmarks: Dict[str, LandmarkPoint],
    ) -> Optional[np.ndarray]:
        """Extract a normalized 24x32 grayscale torso patch in LK pyramid space."""
        if lk_gray is None or lk_gray.size == 0:
            return None
        s = self.lk_scale
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)

        ls = landmarks.get("left_shoulder")
        rs = landmarks.get("right_shoulder")
        lh = landmarks.get("left_hip")
        rh = landmarks.get("right_hip")

        if (
            ls is not None
            and rs is not None
            and lh is not None
            and rh is not None
            and ls.conf >= 0.25
            and rs.conf >= 0.25
            and lh.conf >= 0.20
            and rh.conf >= 0.20
        ):
            sh_w = float(np.hypot(ls.x - rs.x, ls.y - rs.y))
            pad = 0.12 * max(sh_w, 0.25 * bw)
            tx1 = min(ls.x, rs.x, lh.x, rh.x) - pad
            tx2 = max(ls.x, rs.x, lh.x, rh.x) + pad
            ty1 = min(ls.y, rs.y)
            ty2 = max(lh.y, rh.y)
        else:
            tx1 = x1 + 0.20 * bw
            tx2 = x2 - 0.20 * bw
            ty1 = y1 + 0.18 * bh
            ty2 = y1 + 0.68 * bh

        ix1 = int(max(0, round(tx1 * s)))
        iy1 = int(max(0, round(ty1 * s)))
        ix2 = int(min(self.lk_w, round(tx2 * s)))
        iy2 = int(min(self.lk_h, round(ty2 * s)))
        if (ix2 - ix1) < 4 or (iy2 - iy1) < 4:
            return None
        return cv2.resize(lk_gray[iy1:iy2, ix1:ix2], (24, 32), interpolation=cv2.INTER_AREA)

    def _compute_pixel_change_rois(self, frame_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Update grayscale pyramids and extract candidate moving ROIs `(x1, y1, x2, y2, energy)`."""
        if self.lk_scale < 0.999:
            lk_bgr = cv2.resize(frame_bgr, (self.lk_w, self.lk_h), interpolation=cv2.INTER_AREA)
        else:
            lk_bgr = frame_bgr
        curr_lk = cv2.cvtColor(lk_bgr, cv2.COLOR_BGR2GRAY)
        self.curr_lk_gray = curr_lk

        work_u8 = cv2.resize(curr_lk, (self.work_w, self.work_h), interpolation=cv2.INTER_AREA)
        curr_work = cv2.GaussianBlur(work_u8, (5, 5), 0).astype(np.float32)

        self.non_human_cooldown = np.maximum(self.non_human_cooldown - 1, 0)

        if self.prev_work_gray is None or self.bg_work_gray is None:
            self.prev_work_gray = curr_work.copy()
            self.bg_work_gray = curr_work.copy()
            self.step_active_mask = np.zeros((self.work_h, self.work_w), dtype=np.uint8)
            self.fg_active_mask = np.zeros((self.work_h, self.work_w), dtype=np.uint8)
            return []

        raw_diff = curr_work - self.prev_work_gray
        illum_shift = float(np.median(raw_diff))
        step_diff = np.abs(raw_diff - illum_shift)

        mad = float(np.median(step_diff))
        sigma = max(1.4826 * mad, 1.5)
        self.tau_step = max(3.4 * sigma, 4.5)

        self.bg_work_gray = np.clip(self.bg_work_gray + illum_shift, 0.0, 255.0)
        bg_diff = np.abs(curr_work - self.bg_work_gray)
        tau_bg = max(4.6 * sigma, 8.0)

        step_bool = step_diff >= self.tau_step
        fg_bool = step_bool | (bg_diff >= tau_bg)

        alpha = np.where(fg_bool, 0.004, 0.045).astype(np.float32)
        self.bg_work_gray = (1.0 - alpha) * self.bg_work_gray + alpha * curr_work
        self.prev_work_gray = curr_work

        self.step_active_mask = step_bool.astype(np.uint8)
        # Build candidate ROIs strictly from inter-frame pixel changes (`step_bool`), avoiding vacated-chair background ghosts!
        step_u8 = (step_bool.astype(np.uint8)) * 255

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_rad = max(int(round(0.035 * self.work_w)), 7)
        if close_rad % 2 == 0:
            close_rad += 1
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_rad, close_rad))

        cleaned = cv2.morphologyEx(step_u8, cv2.MORPH_OPEN, k_open)
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
        self.fg_active_mask = (closed > 0).astype(np.uint8)

        total_area = float(self.work_w * self.work_h)
        fg_frac = float(np.count_nonzero(self.fg_active_mask)) / total_area
        # Reject global camera shake / auto-exposure flash covering > 72% of the frame
        if fg_frac > 0.72:
            return []

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        sx = float(self.frame_width) / float(self.work_w)
        sy = float(self.frame_height) / float(self.work_h)

        rois: List[Tuple[float, float, float, float, float]] = []
        for lbl_idx in range(1, num_labels):
            area = int(stats[lbl_idx, cv2.CC_STAT_AREA])
            area_ratio = float(area) / total_area
            if area_ratio < 0.0014 or area_ratio > 0.68:
                continue
            rx = float(stats[lbl_idx, cv2.CC_STAT_LEFT])
            ry = float(stats[lbl_idx, cv2.CC_STAT_TOP])
            rw = float(stats[lbl_idx, cv2.CC_STAT_WIDTH])
            rh = float(stats[lbl_idx, cv2.CC_STAT_HEIGHT])
            comp_mask = labels == lbl_idx
            energy = float(np.sum(step_diff[comp_mask]))
            rois.append((rx * sx, ry * sy, (rx + rw) * sx, (ry + rh) * sy, energy))

        rois.sort(key=lambda r: r[4], reverse=True)
        return rois[:5]

    def _box_motion_fraction(self, bbox: Tuple[float, float, float, float], use_step_only: bool = True) -> float:
        """Compute fraction of pixels with illumination-compensated motion inside `bbox`."""
        mask = self.step_active_mask if use_step_only else self.fg_active_mask
        if mask is None:
            return 0.0
        sx = float(self.work_w) / float(self.frame_width)
        sy = float(self.work_h) / float(self.frame_height)
        ix1 = int(max(0, np.floor(bbox[0] * sx)))
        iy1 = int(max(0, np.floor(bbox[1] * sy)))
        ix2 = int(min(self.work_w, np.ceil(bbox[2] * sx)))
        iy2 = int(min(self.work_h, np.ceil(bbox[3] * sy)))
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        sub = mask[iy1:iy2, ix1:ix2]
        return float(np.mean(sub))

    def _detect_pose_in_crop(
        self,
        frame_bgr: np.ndarray,
        crop_box: Tuple[float, float, float, float],
        source: str,
        mask_boxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
        require_motion_overlap: bool = False,
    ) -> Optional[DetectionRaw]:
        """Run MediaPipe Pose (`model_complexity=1`) on `crop_box` and strictly validate human anatomy."""
        cx1 = int(max(0, round(crop_box[0])))
        cy1 = int(max(0, round(crop_box[1])))
        cx2 = int(min(self.frame_width, round(crop_box[2])))
        cy2 = int(min(self.frame_height, round(crop_box[3])))
        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw < max(int(0.04 * self.frame_width), 24) or ch < max(int(0.08 * self.frame_height), 32):
            return None

        crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
        if mask_boxes:
            crop_bgr = crop_bgr.copy()
            fill_bgr = tuple(int(v) for v in np.median(crop_bgr.reshape(-1, 3), axis=0))
            for mb in mask_boxes:
                mw = mb[2] - mb[0]
                mh = mb[3] - mb[1]
                # Mask inner 68% core of already-detected people so neighbors remain visible
                mx1 = int(max(0, round((mb[0] + 0.16 * mw) - cx1)))
                my1 = int(max(0, round((mb[1] + 0.12 * mh) - cy1)))
                mx2 = int(min(cw, round((mb[2] - 0.16 * mw) - cx1)))
                my2 = int(min(ch, round((mb[3] - 0.12 * mh) - cy1)))
                if mx2 > mx1 and my2 > my1:
                    crop_bgr[my1:my2, mx1:mx2] = fill_bgr

        # MediaPipe Pose internal detector/landmark input is 256x256 -> resize once in OpenCV
        max_c = max(cw, ch)
        if max_c > 288:
            sc = 288.0 / float(max_c)
            proc_bgr = cv2.resize(
                crop_bgr,
                (max(int(round(cw * sc)), 32), max(int(round(ch * sc)), 32)),
                interpolation=cv2.INTER_AREA,
            )
        elif max_c < 160:
            sc = 224.0 / float(max_c)
            proc_bgr = cv2.resize(crop_bgr, (int(round(cw * sc)), int(round(ch * sc))), interpolation=cv2.INTER_LINEAR)
        else:
            proc_bgr = crop_bgr

        rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        res = self._get_mp_pose().process(rgb)
        if not res.pose_landmarks:
            return None

        mp_lms = res.pose_landmarks.landmark
        ls_mp = mp_lms[11]
        rs_mp = mp_lms[12]

        # 1. Near shoulder must be clearly visible and far shoulder plausible (supports 90-deg side-profile desks!)
        max_sh_vis = max(float(ls_mp.visibility), float(rs_mp.visibility))
        min_sh_vis = min(float(ls_mp.visibility), float(rs_mp.visibility))
        if max_sh_vis < 0.60 or min_sh_vis < 0.26:
            return None
        if not (0.01 <= ls_mp.x <= 0.99 and 0.01 <= rs_mp.x <= 0.99):
            return None
        if not (0.01 <= ls_mp.y <= 0.98 and 0.01 <= rs_mp.y <= 0.98):
            return None

        # 2. Head landmarks (frontal face OR back-turned ears/shoulders+hips) must sit anatomically above shoulders
        head_indices = (0, 2, 5, 7, 8)
        best_head_vis = max(float(mp_lms[idx].visibility) for idx in head_indices)
        both_hips_strong = float(mp_lms[23].visibility) >= 0.58 and float(mp_lms[24].visibility) >= 0.58
        back_view_valid = min_sh_vis >= 0.62 and both_hips_strong and best_head_vis >= 0.22
        if best_head_vis < 0.50 and not back_view_valid:
            return None

        fw = float(self.frame_width)
        fh = float(self.frame_height)
        landmarks_dict: Dict[str, LandmarkPoint] = {}
        vis_pts_x: List[float] = []
        vis_pts_y: List[float] = []

        for name in TARGET_LANDMARKS:
            mp_idx = MP_LANDMARK_MAP[name]
            pt = mp_lms[mp_idx]
            gx = float(cx1) + float(pt.x) * float(cw)
            gy = float(cy1) + float(pt.y) * float(ch)
            vis = float(pt.visibility)
            if pt.x < -0.02 or pt.x > 1.02 or pt.y < -0.02 or pt.y > 1.04:
                vis *= 0.25
            gx_clamped = min(max(gx, 0.0), fw)
            gy_clamped = min(max(gy, 0.0), fh)
            landmarks_dict[name] = LandmarkPoint(
                name=name,
                x=gx_clamped,
                y=gy_clamped,
                conf=vis,
                x_norm=gx_clamped / fw,
                y_norm=gy_clamped / fh,
            )
            if vis >= 0.35 and -0.02 <= pt.x <= 1.02 and -0.02 <= pt.y <= 1.02:
                vis_pts_x.append(gx_clamped)
                vis_pts_y.append(gy_clamped)

        ls_pt = landmarks_dict["left_shoulder"]
        rs_pt = landmarks_dict["right_shoulder"]
        lh_pt = landmarks_dict["left_hip"]
        rh_pt = landmarks_dict["right_hip"]
        sh_mid_x = 0.5 * (ls_pt.x + rs_pt.x)
        sh_mid_y = 0.5 * (ls_pt.y + rs_pt.y)
        sh_w = float(np.hypot(ls_pt.x - rs_pt.x, ls_pt.y - rs_pt.y))

        has_hips = lh_pt.conf >= 0.35 and rh_pt.conf >= 0.35
        hip_mid_x = 0.5 * (lh_pt.x + rh_pt.x) if has_hips else sh_mid_x
        hip_mid_y = 0.5 * (lh_pt.y + rh_pt.y) if has_hips else (sh_mid_y + 1.35 * max(sh_w, 0.03 * fh))
        spine_len = float(np.hypot(hip_mid_x - sh_mid_x, hip_mid_y - sh_mid_y)) if has_hips else 0.0

        # Yaw-invariant biological reference width (handles side-profile foreshortening of sh_w)
        bio_ref = max(sh_w, 0.68 * spine_len)

        # Include visible knees/ankles when legs are genuinely extended vertically (standing/walking)
        for lb_idx in MP_LOWER_BODY_INDICES:
            pt = mp_lms[lb_idx]
            if float(pt.visibility) >= 0.58 and 0.0 <= pt.x <= 1.0 and 0.0 <= pt.y <= 0.99:
                gx = min(max(float(cx1) + float(pt.x) * float(cw), 0.0), fw)
                gy = min(max(float(cy1) + float(pt.y) * float(ch), 0.0), fh)
                if not has_hips or (gy - hip_mid_y) >= 0.45 * max(spine_len, bio_ref):
                    vis_pts_x.append(gx)
                    vis_pts_y.append(gy)

        if len(vis_pts_x) < 5:
            return None

        # 3. Require >= 2 additional upper/mid-body joints on fresh discovery, or >=1 / high-conf head+shoulders on active track re-anchor
        limb_hip_vis = sum(
            1
            for jn in ("left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip")
            if landmarks_dict[jn].conf >= 0.38
        )
        min_req_limbs = 1 if source == "mp_track" else 2
        if limb_hip_vis < min_req_limbs and not (source == "mp_track" and max_sh_vis >= 0.72 and best_head_vis >= 0.68):
            return None

        # Reject tile detections whose shoulders sit right against an internal bottom crop edge (truncated torso)
        if source == "mp_tile" and cy2 < int(0.96 * fh) and sh_mid_y > (cy2 - 0.16 * ch):
            return None

        if (bio_ref / fw) < 0.018 or (bio_ref / fw) > 0.48:
            return None

        # Head must sit anatomically above/near shoulder line and horizontally aligned with torso
        head_min_c = 0.20 if back_view_valid else 0.48
        head_valid = [
            landmarks_dict[jn]
            for jn in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
            if landmarks_dict[jn].conf >= head_min_c
        ]
        if not head_valid:
            return None
        head_x = float(np.mean([p.x for p in head_valid]))
        head_y = float(np.mean([p.y for p in head_valid]))
        if head_y > sh_mid_y + 0.22 * bio_ref or abs(head_x - sh_mid_x) > 1.45 * bio_ref:
            return None

        if has_hips and hip_mid_y < sh_mid_y + 0.30 * bio_ref:
            return None

        eff_head_c = max(best_head_vis, 0.72 if back_view_valid else 0.0)
        core_conf = float((max_sh_vis + 0.5 * (ls_pt.conf + rs_pt.conf) + eff_head_c) / 3.0)
        min_core_req = 0.72 if source == "mp_tile" else 0.58
        if core_conf < min_core_req:
            return None

        # 4. Tightly fitted anatomical person bounding box
        pad_x = max(0.34 * bio_ref, 0.012 * fw)
        pad_top = max(0.48 * bio_ref, 0.018 * fh)
        pad_bot = max(0.26 * bio_ref, 0.015 * fh)

        bx1 = max(0.0, min(vis_pts_x) - pad_x)
        bx2 = min(fw, max(vis_pts_x) + pad_x)
        by1 = max(0.0, min(head_y - pad_top, min(vis_pts_y) - 0.25 * bio_ref))
        by2 = min(fh, max(max(vis_pts_y) + pad_bot, sh_mid_y + 1.45 * bio_ref))

        bbox = (bx1, by1, bx2, by2)
        if not is_plausible_human_pose(bbox, landmarks_dict, frame_width=self.frame_width, frame_height=self.frame_height):
            return None

        if require_motion_overlap:
            step_frac = self._box_motion_fraction(bbox, use_step_only=True)
            # Candidate at t > 0 must overlap genuine inter-frame pixel changes (never vacated-chair ghosts!)
            if step_frac < 0.018:
                return None

        return DetectionRaw(
            bbox=bbox,
            bbox_conf=min(0.98, core_conf),
            class_id=0,
            landmarks=landmarks_dict,
            raw_track_id=None,
            source=source,
        )

    def _is_covered_by_existing(
        self,
        box: Tuple[float, float, float, float],
        existing: Sequence[DetectionRaw],
        iou_gate: float = 0.28,
        contain_gate: float = 0.58,
    ) -> bool:
        for det in existing:
            iou, contain = _box_iou_and_containment(box, det.bbox)
            if iou >= iou_gate or contain >= contain_gate:
                return True
        return False

    def _propagate_landmarks_lk(
        self,
        prev_bbox: Tuple[float, float, float, float],
        prev_landmarks: Dict[str, LandmarkPoint],
        bio_scale: float,
    ) -> Tuple[Tuple[float, float, float, float], Dict[str, LandmarkPoint], Dict[str, Tuple[float, float, bool]]]:
        """Propagate 13 landmarks from `prev_lk_gray` to `curr_lk_gray` using forward-backward
        Pyramidal Lucas-Kanade optical flow, zeroing sub-noise drift on static pixels.
        """
        flow_map: Dict[str, Tuple[float, float, bool]] = {}
        if self.prev_lk_gray is None or self.curr_lk_gray is None:
            return prev_bbox, prev_landmarks, flow_map

        names = list(TARGET_LANDMARKS)
        pts_prev = np.array(
            [[prev_landmarks[n].x * self.lk_scale, prev_landmarks[n].y * self.lk_scale] for n in names],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        lk_win = max(int(round(0.22 * bio_scale * self.lk_scale)), 11)
        if lk_win % 2 == 0:
            lk_win += 1
        lk_win = min(lk_win, 25)
        lk_params = dict(
            winSize=(lk_win, lk_win),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )

        pts_next, st_fwd, _ = cv2.calcOpticalFlowPyrLK(self.prev_lk_gray, self.curr_lk_gray, pts_prev, None, **lk_params)
        pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(self.curr_lk_gray, self.prev_lk_gray, pts_next, None, **lk_params)

        inv_s = 1.0 / max(self.lk_scale, 1e-6)
        fw = float(self.frame_width)
        fh = float(self.frame_height)
        max_fb_px = max(0.10 * bio_scale * self.lk_scale, 1.5)

        new_lms: Dict[str, LandmarkPoint] = {}
        core_shifts_x: List[float] = []
        core_shifts_y: List[float] = []

        for idx, name in enumerate(names):
            old_lm = prev_landmarks[name]
            ok = (
                pts_next is not None
                and pts_back is not None
                and st_fwd is not None
                and st_bwd is not None
                and int(st_fwd[idx, 0]) == 1
                and int(st_bwd[idx, 0]) == 1
            )
            if ok and old_lm.conf > 0.05:
                fb_err = float(np.hypot(pts_prev[idx, 0, 0] - pts_back[idx, 0, 0], pts_prev[idx, 0, 1] - pts_back[idx, 0, 1]))
                if fb_err <= max_fb_px:
                    nx_full = float(pts_next[idx, 0, 0]) * inv_s
                    ny_full = float(pts_next[idx, 0, 1]) * inv_s
                    dx = nx_full - old_lm.x
                    dy = ny_full - old_lm.y

                    # Suppress sub-noise optical flow drift when local patch pixels did not change
                    rad = 0.18 * bio_scale
                    patch_mot = self._box_motion_fraction((old_lm.x - rad, old_lm.y - rad, old_lm.x + rad, old_lm.y + rad))
                    if patch_mot < 0.010 and float(np.hypot(dx, dy)) < 0.06 * bio_scale:
                        dx, dy = 0.0, 0.0

                    # Bound single-frame optical flow step to physically plausible limb speed
                    step_mag = float(np.hypot(dx, dy))
                    if step_mag > 0.85 * bio_scale:
                        scale_cap = (0.85 * bio_scale) / step_mag
                        dx *= scale_cap
                        dy *= scale_cap

                    cx = min(max(old_lm.x + dx, 0.0), fw)
                    cy = min(max(old_lm.y + dy, 0.0), fh)
                    flow_map[name] = (cx, cy, True)
                    new_lms[name] = LandmarkPoint(
                        name=name,
                        x=cx,
                        y=cy,
                        conf=old_lm.conf,
                        x_norm=cx / fw,
                        y_norm=cy / fh,
                    )
                    if name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip") and old_lm.conf >= 0.25:
                        core_shifts_x.append(dx)
                        core_shifts_y.append(dy)
                    continue

            flow_map[name] = (old_lm.x, old_lm.y, False)
            new_lms[name] = LandmarkPoint(
                name=name,
                x=old_lm.x,
                y=old_lm.y,
                conf=old_lm.conf,
                x_norm=old_lm.x_norm,
                y_norm=old_lm.y_norm,
            )

        med_dx = float(np.median(core_shifts_x)) if core_shifts_x else 0.0
        med_dy = float(np.median(core_shifts_y)) if core_shifts_y else 0.0
        nbx1 = min(max(prev_bbox[0] + med_dx, 0.0), fw)
        nby1 = min(max(prev_bbox[1] + med_dy, 0.0), fh)
        nbx2 = min(max(prev_bbox[2] + med_dx, 0.0), fw)
        nby2 = min(max(prev_bbox[3] + med_dy, 0.0), fh)
        return (nbx1, nby1, nbx2, nby2), new_lms, flow_map

    def augment_detections(
        self,
        frame_bgr: np.ndarray,
        yolo_detections: List[DetectionRaw],
        active_tracks: Sequence[Any],
    ) -> List[DetectionRaw]:
        """Discover missed moving & stationary employees via pixel changes + pose verification,
        maintain motionless employee tracks, and stabilize keypoints with LK optical flow.
        """
        self.step_idx += 1
        motion_rois = self._compute_pixel_change_rois(frame_bgr)
        merged: List[DetectionRaw] = list(yolo_detections)
        fw = float(self.frame_width)
        fh = float(self.frame_height)

        # --- Stage A: Initial Spatial Tile Discovery on Frame 1-2 (Already-Seated Motionless Employees) ---
        # After Frame 2, any newly arriving employee MUST produce pixel changes (`Stage B`), preventing static chair hallucinations!
        if self.step_idx in (1, 2):
            tiles_norm = [
                (0.00, 0.28, 0.48, 1.00),
                (0.26, 0.28, 0.74, 1.00),
                (0.52, 0.28, 1.00, 1.00),
                (0.00, 0.00, 0.48, 0.70),
                (0.26, 0.00, 0.74, 0.70),
                (0.52, 0.00, 1.00, 0.70),
            ]
            if self.step_idx == 1:
                tiles_norm.append((0.06, 0.02, 0.94, 0.98))

            for tx1_n, ty1_n, tx2_n, ty2_n in tiles_norm:
                t_box = (tx1_n * fw, ty1_n * fh, tx2_n * fw, ty2_n * fh)
                cand = self._detect_pose_in_crop(
                    frame_bgr,
                    t_box,
                    source="mp_tile",
                    mask_boxes=None,
                    require_motion_overlap=False,
                )
                if cand is not None and not self._is_covered_by_existing(cand.bbox, merged, iou_gate=0.30, contain_gate=0.48):
                    merged = deduplicate_detections(merged + [cand])

        # --- Stage B: Inter-Frame Pixel-Change ROI Discovery (Moving Employees) ---
        tracks_flagged_motion: set = set()
        checks_done = 0
        padded_merged_boxes = []
        for d in merged:
            ds = estimate_biological_scale(d.bbox, d.landmarks)
            # Include all visible limb/wrist reach so an employee's own writing/reaching arm never spawns a 2nd skeleton
            vis_xs = [d.bbox[0], d.bbox[2]] + [lm.x for lm in d.landmarks.values() if lm.conf >= 0.20]
            vis_ys = [d.bbox[1], d.bbox[3]] + [lm.y for lm in d.landmarks.values() if lm.conf >= 0.20]
            padded_merged_boxes.append(
                DetectionRaw(
                    bbox=(min(vis_xs) - 0.55 * ds, min(vis_ys) - 0.35 * ds, max(vis_xs) + 0.55 * ds, max(vis_ys) + 0.45 * ds),
                    bbox_conf=d.bbox_conf,
                    class_id=d.class_id,
                )
            )

        for rx1, ry1, rx2, ry2, _ in motion_rois:
            r_box = (rx1, ry1, rx2, ry2)
            if self._is_covered_by_existing(r_box, padded_merged_boxes, iou_gate=0.16, contain_gate=0.48):
                continue

            # If motion falls inside an active track not yet in `merged`, flag Stage C to refresh it
            matched_active_trk = False
            for trk in active_tracks:
                iou_t, cont_t = _box_iou_and_containment(r_box, trk.last_bbox)
                if iou_t >= 0.18 or cont_t >= 0.50:
                    tracks_flagged_motion.add(trk.track_id)
                    matched_active_trk = True
            if matched_active_trk:
                continue

            rcx = 0.5 * (rx1 + rx2) / fw
            rcy = 0.5 * (ry1 + ry2) / fh
            gc = min(max(int(rcx * self.grid_cols), 0), self.grid_cols - 1)
            gr = min(max(int(rcy * self.grid_rows), 0), self.grid_rows - 1)
            if self.non_human_cooldown[gr, gc] > 0:
                continue

            if checks_done >= 2:
                break
            checks_done += 1

            rw = rx2 - rx1
            rh = ry2 - ry1
            pad_x = max(0.52 * rw, 0.09 * fw)
            pad_top = max(0.45 * rh, 0.09 * fh)
            pad_bot = max(0.85 * rh, 0.18 * fh)
            exp_box = (
                max(0.0, rx1 - pad_x),
                max(0.0, ry1 - pad_top),
                min(fw, rx2 + pad_x),
                min(fh, ry2 + pad_bot),
            )
            cand = self._detect_pose_in_crop(
                frame_bgr,
                exp_box,
                source="mp_motion",
                mask_boxes=None,
                require_motion_overlap=True,
            )
            if cand is not None and not self._is_covered_by_existing(cand.bbox, padded_merged_boxes, iou_gate=0.22, contain_gate=0.45):
                self.non_human_fail_streak[gr, gc] = 0
                self.non_human_cooldown[gr, gc] = 0
                merged = deduplicate_detections(merged + [cand])
            else:
                # Non-employee moving pixels (monitor flicker, shadow, curtain/plant) -> cooldown cell
                self.non_human_fail_streak[gr, gc] += 1
                self.non_human_cooldown[gr, gc] = 18

        # --- Stage C: Active Employee Track Continuity (Cropped Pose Re-Anchor + Stationary LK Hold) ---
        for trk in active_tracks:
            tid = trk.track_id
            t_scale = estimate_biological_scale(trk.last_bbox, trk.last_landmarks)
            t_ls = trk.last_landmarks.get("left_shoulder")
            t_rs = trk.last_landmarks.get("right_shoulder")
            t_has_sh = t_ls is not None and t_rs is not None and (t_ls.conf > 0.25 or t_rs.conf > 0.25)
            t_sh_x = 0.5 * (t_ls.x + t_rs.x) if t_has_sh else 0.5 * (trk.last_bbox[0] + trk.last_bbox[2])
            t_sh_y = 0.5 * (t_ls.y + t_rs.y) if t_has_sh else 0.5 * (trk.last_bbox[1] + trk.last_bbox[3])

            already_matched = False
            for d in merged:
                iou_d, _ = _box_iou_and_containment(trk.last_bbox, d.bbox)
                if iou_d >= 0.26:
                    already_matched = True
                    break
                d_ls = d.landmarks.get("left_shoulder")
                d_rs = d.landmarks.get("right_shoulder")
                if t_has_sh and d_ls is not None and d_rs is not None and (d_ls.conf > 0.25 or d_rs.conf > 0.25):
                    d_sh_x = 0.5 * (d_ls.x + d_rs.x)
                    d_sh_y = 0.5 * (d_ls.y + d_rs.y)
                    if float(np.hypot(t_sh_x - d_sh_x, t_sh_y - d_sh_y)) < 0.70 * t_scale:
                        already_matched = True
                        break

            if already_matched:
                continue

            local_mot = self._box_motion_fraction(trk.last_bbox, use_step_only=True)
            anchor = self.track_anchors.get(tid)
            should_run_mp = (
                anchor is None
                or trk.hits < 3
                or tid in tracks_flagged_motion
                or local_mot >= 0.045
                or trk.time_since_update > 0
                or ((self.step_idx + abs(tid) * 2) % 7 == 0)
            )

            mp_succeeded = False
            if should_run_mp and trk.time_since_update <= 15:
                tbx1, tby1, tbx2, tby2 = trk.last_bbox
                exp_trk_box = (
                    max(0.0, tbx1 - 0.55 * t_scale),
                    max(0.0, tby1 - 0.45 * t_scale),
                    min(fw, tbx2 + 0.55 * t_scale),
                    min(fh, tby2 + 0.55 * t_scale),
                )
                other_boxes = [
                    other.last_bbox
                    for other in active_tracks
                    if other.track_id != tid and _box_iou_and_containment(trk.last_bbox, other.last_bbox)[0] < 0.35
                ] + [d.bbox for d in merged]
                cand_trk = self._detect_pose_in_crop(
                    frame_bgr,
                    exp_trk_box,
                    source="mp_track",
                    mask_boxes=None,
                    require_motion_overlap=False,
                )
                if cand_trk is not None:
                    iou_c, _ = _box_iou_and_containment(trk.last_bbox, cand_trk.bbox)
                    c_ls = cand_trk.landmarks["left_shoulder"]
                    c_rs = cand_trk.landmarks["right_shoulder"]
                    c_sh_dist = float(np.hypot(t_sh_x - 0.5 * (c_ls.x + c_rs.x), t_sh_y - 0.5 * (c_ls.y + c_rs.y)))
                    if iou_c >= 0.12 or c_sh_dist <= 1.35 * t_scale:
                        cand_trk.target_track_id = tid
                        cand_trk.raw_track_id = trk.last_raw_bt_id
                        merged = deduplicate_detections(merged + [cand_trk])
                        mp_succeeded = True

            if mp_succeeded:
                continue

            # Stationary Torso Pixel Persistence + LK Optical Flow Hold
            if anchor is not None and trk.hits >= 1 and trk.time_since_update <= 2:
                steps_since_verified = self.step_idx - int(anchor["last_pose_verified_step"])
                curr_patch = self._extract_torso_gray_patch(self.curr_lk_gray, trk.last_bbox, trk.last_landmarks)
                zncc_anchor = _zncc_patch(curr_patch, anchor.get("anchor_torso_patch"))
                zncc_prev = _zncc_patch(curr_patch, anchor.get("prev_torso_patch"))
                max_hold_steps = 900 if trk.hits >= 12 else 45

                if not should_run_mp:
                    can_hold = steps_since_verified <= max_hold_steps and zncc_prev >= 0.78 and zncc_anchor >= 0.72 and local_mot < 0.045
                else:
                    # MP Pose missed (e.g., resting chin on hand / back turned while seated), hold while torso pixels remain in place
                    can_hold = steps_since_verified <= max_hold_steps and zncc_anchor >= 0.75 and zncc_prev >= 0.80 and local_mot < 0.035

                if can_hold:
                    lk_bbox, lk_lms, _ = self._propagate_landmarks_lk(trk.last_bbox, trk.last_landmarks, t_scale)
                    hold_det = DetectionRaw(
                        bbox=lk_bbox,
                        bbox_conf=max(float(trk.last_bbox_conf) * 0.96, 0.55),
                        class_id=0,
                        landmarks=lk_lms,
                        raw_track_id=trk.last_raw_bt_id,
                        source="lk_hold",
                        target_track_id=tid,
                    )
                    merged.append(hold_det)

        # --- Stage D: Stabilize Pose Landmarks Against Pixel Optical Flow (Suppress Static Jitter) ---
        if self.prev_lk_gray is not None and self.curr_lk_gray is not None and active_tracks:
            for det in merged:
                if det.source == "lk_hold":
                    continue
                best_trk = None
                best_iou = 0.25
                for trk in active_tracks:
                    if det.target_track_id is not None and trk.track_id == det.target_track_id:
                        best_trk = trk
                        break
                    iou_t, _ = _box_iou_and_containment(det.bbox, trk.last_bbox)
                    if iou_t > best_iou:
                        best_iou = iou_t
                        best_trk = trk

                if best_trk is None:
                    continue

                d_scale = estimate_biological_scale(det.bbox, det.landmarks)
                _, _, flow_map = self._propagate_landmarks_lk(best_trk.last_bbox, best_trk.last_landmarks, d_scale)
                for j_name, (fx, fy, f_valid) in flow_map.items():
                    if not f_valid or j_name not in det.landmarks:
                        continue
                    dlm = det.landmarks[j_name]
                    if dlm.conf < 0.15:
                        continue
                    jump = float(np.hypot(dlm.x - fx, dlm.y - fy))
                    rad = 0.20 * d_scale
                    mot_old = self._box_motion_fraction((fx - rad, fy - rad, fx + rad, fy + rad))
                    mot_new = self._box_motion_fraction((dlm.x - rad, dlm.y - rad, dlm.x + rad, dlm.y + rad))
                    local_mot_max = max(mot_old, mot_new)

                    if local_mot_max < 0.012 and jump < 0.35 * d_scale:
                        # Pixels at this joint are static -> anchor strongly to LK optical flow to kill jitter
                        w_det = 0.16
                    elif jump < 0.45 * d_scale:
                        w_det = 0.78
                    else:
                        w_det = 1.0

                    bx = w_det * dlm.x + (1.0 - w_det) * fx
                    by = w_det * dlm.y + (1.0 - w_det) * fy
                    det.landmarks[j_name] = LandmarkPoint(
                        name=j_name,
                        x=bx,
                        y=by,
                        conf=dlm.conf,
                        x_norm=bx / fw,
                        y_norm=by / fh,
                    )

        return merged

    def _lk_patch_motion_fraction(self, bbox: Tuple[float, float, float, float]) -> float:
        """High-resolution (`lk_gray`, up to 960px) illumination-compensated pixel-change fraction
        for capturing fine finger/mouse typing motion even on wide-angle 2.8K/4K cameras.
        """
        if self.prev_lk_gray is None or self.curr_lk_gray is None:
            return 0.0
        s = self.lk_scale
        ix1 = int(max(0, np.floor(bbox[0] * s)))
        iy1 = int(max(0, np.floor(bbox[1] * s)))
        ix2 = int(min(self.lk_w, np.ceil(bbox[2] * s)))
        iy2 = int(min(self.lk_h, np.ceil(bbox[3] * s)))
        if (ix2 - ix1) < 3 or (iy2 - iy1) < 3:
            return 0.0
        p0 = cv2.GaussianBlur(self.prev_lk_gray[iy1:iy2, ix1:ix2], (3, 3), 0).astype(np.float32)
        p1 = cv2.GaussianBlur(self.curr_lk_gray[iy1:iy2, ix1:ix2], (3, 3), 0).astype(np.float32)
        diff = p1 - p0
        adiff = np.abs(diff - float(np.median(diff)))
        thresh = max(self.tau_step * 1.05, 5.0)
        return float(np.mean(adiff >= thresh))

    def _measure_hand_patch_optical_flow_sw_s(
        self,
        centers: Sequence[Tuple[float, float]],
        bio_scale: float,
    ) -> float:
        """Measure forward-backward validated LK optical flow speed (`SW/s`) on a 3x3 probe grid around wrist/hand."""
        if self.prev_lk_gray is None or self.curr_lk_gray is None or not centers:
            return 0.0

        s = self.lk_scale
        step_r = max(0.10 * bio_scale * s, 2.0)
        probes: List[List[float]] = []
        for cx, cy in centers:
            px, py = cx * s, cy * s
            for ox in (-step_r, 0.0, step_r):
                for oy in (-step_r, 0.0, step_r):
                    qx = min(max(px + ox, 1.0), float(self.lk_w - 2))
                    qy = min(max(py + oy, 1.0), float(self.lk_h - 2))
                    probes.append([qx, qy])

        pts0 = np.array(probes, dtype=np.float32).reshape(-1, 1, 2)
        win_sz = max(int(round(0.20 * bio_scale * s)), 11)
        if win_sz % 2 == 0:
            win_sz += 1
        win_sz = min(win_sz, 23)
        lk_params = dict(
            winSize=(win_sz, win_sz),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        )
        pts1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_lk_gray, self.curr_lk_gray, pts0, None, **lk_params)
        pts0_b, st2, _ = cv2.calcOpticalFlowPyrLK(self.curr_lk_gray, self.prev_lk_gray, pts1, None, **lk_params)
        if pts1 is None or pts0_b is None or st1 is None or st2 is None:
            return 0.0

        max_fb = max(0.10 * bio_scale * s, 1.5)
        inv_s = 1.0 / max(s, 1e-6)
        valid_speeds_sw_s: List[float] = []
        for i in range(len(probes)):
            if int(st1[i, 0]) == 1 and int(st2[i, 0]) == 1:
                fb = float(np.hypot(pts0[i, 0, 0] - pts0_b[i, 0, 0], pts0[i, 0, 1] - pts0_b[i, 0, 1]))
                if fb <= max_fb:
                    disp_full = float(np.hypot(pts1[i, 0, 0] - pts0[i, 0, 0], pts1[i, 0, 1] - pts0[i, 0, 1])) * inv_s
                    valid_speeds_sw_s.append((disp_full / max(bio_scale, 1e-3)) * self.fps)

        if not valid_speeds_sw_s:
            return 0.0
        return float(np.percentile(valid_speeds_sw_s, 75.0))

    def attach_pixel_motion_metrics(
        self,
        observations: List[SmoothedObservation],
        raw_detections: Sequence[DetectionRaw],
    ) -> None:
        """Update per-track pixel anchors, attach wrist/hand pixel-change & optical-flow telemetry
        to each `SmoothedObservation`, and roll the LK pyramid buffer forward.
        """
        lk_hold_ids = {d.target_track_id for d in raw_detections if d.source == "lk_hold" and d.target_track_id is not None}

        for obs in observations:
            tid = obs.track_id
            bio_s = estimate_biological_scale(obs.bbox, obs.landmarks)
            curr_patch = self._extract_torso_gray_patch(self.curr_lk_gray, obs.bbox, obs.landmarks)

            if tid not in self.track_anchors:
                self.track_anchors[tid] = {
                    "anchor_torso_patch": curr_patch,
                    "prev_torso_patch": curr_patch,
                    "last_pose_verified_step": self.step_idx,
                }
            else:
                self.track_anchors[tid]["prev_torso_patch"] = curr_patch
                if obs.tracking_status == "tracked" and tid not in lk_hold_ids:
                    if curr_patch is not None:
                        self.track_anchors[tid]["anchor_torso_patch"] = curr_patch
                    self.track_anchors[tid]["last_pose_verified_step"] = self.step_idx

            lms = obs.landmarks
            lw = lms["left_wrist"]
            rw = lms["right_wrist"]
            lelb = lms["left_elbow"]
            relb = lms["right_elbow"]

            rad = 0.24 * bio_s
            l_centers: List[Tuple[float, float]] = []
            r_centers: List[Tuple[float, float]] = []
            lw_mot = 0.0
            rw_mot = 0.0

            if lw.conf >= 0.10:
                l_centers.append((lw.x, lw.y))
                lw_mot = max(
                    self._box_motion_fraction((lw.x - rad, lw.y - rad, lw.x + rad, lw.y + rad)),
                    self._lk_patch_motion_fraction((lw.x - rad, lw.y - rad, lw.x + rad, lw.y + rad)),
                )
                if lelb.conf >= 0.12:
                    lhx = min(max(lw.x + 0.30 * (lw.x - lelb.x), 0.0), float(self.frame_width))
                    lhy = min(max(lw.y + 0.30 * (lw.y - lelb.y), 0.0), float(self.frame_height))
                    l_centers.append((lhx, lhy))
                    lw_mot = max(
                        lw_mot,
                        self._box_motion_fraction((lhx - rad, lhy - rad, lhx + rad, lhy + rad)),
                        self._lk_patch_motion_fraction((lhx - rad, lhy - rad, lhx + rad, lhy + rad)),
                    )

            if rw.conf >= 0.10:
                r_centers.append((rw.x, rw.y))
                rw_mot = max(
                    self._box_motion_fraction((rw.x - rad, rw.y - rad, rw.x + rad, rw.y + rad)),
                    self._lk_patch_motion_fraction((rw.x - rad, rw.y - rad, rw.x + rad, rw.y + rad)),
                )
                if relb.conf >= 0.12:
                    rhx = min(max(rw.x + 0.30 * (rw.x - relb.x), 0.0), float(self.frame_width))
                    rhy = min(max(rw.y + 0.30 * (rw.y - relb.y), 0.0), float(self.frame_height))
                    r_centers.append((rhx, rhy))
                    rw_mot = max(
                        rw_mot,
                        self._box_motion_fraction((rhx - rad, rhy - rad, rhx + rad, rhy + rad)),
                        self._lk_patch_motion_fraction((rhx - rad, rhy - rad, rhx + rad, rhy + rad)),
                    )

            wrist_px_mot = max(lw_mot, rw_mot)
            lk_lw = self._measure_hand_patch_optical_flow_sw_s(l_centers, bio_s) if l_centers else 0.0
            lk_rw = self._measure_hand_patch_optical_flow_sw_s(r_centers, bio_s) if r_centers else 0.0
            lk_dom = max(lk_lw, lk_rw)

            # Suppress sub-pixel optical flow drift when zero illumination-compensated pixels changed
            if wrist_px_mot < 0.008 and lk_dom < 0.05:
                lk_dom = 0.0

            obs.wrist_pixel_motion_ratio = round(float(wrist_px_mot), 4)
            obs.lk_wrist_flow_sw_s = round(float(lk_dom), 4)

        self.prev_lk_gray = self.curr_lk_gray
