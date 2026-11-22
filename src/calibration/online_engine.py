"""Scale-invariant, dictionary-backed online multi-employee motion engine using
`collections.deque(maxlen=30)` rolling windows and minimum-dwell hysteresis state machines.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from src.tracking.landmarks import SmoothedObservation


class HysteresisStateMachine:
    """Debounced state machine requiring a new state condition to hold continuously
    for at least `min_dwell_frames` consecutive observations before flipping state.
    """

    def __init__(
        self,
        initial_state: str,
        min_dwell_frames: int = 15,
        custom_dwell_overrides: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> None:
        self.current_state: str = initial_state
        self.min_dwell_frames: int = max(int(min_dwell_frames), 1)
        self.custom_dwell_overrides: Dict[Tuple[str, str], int] = custom_dwell_overrides or {}
        self.initialized: bool = False
        self.candidate_state: Optional[str] = None
        self.candidate_streak: int = 0
        self.flickers_suppressed: int = 0

    def step(self, proposed_state: str, immediate_override: bool = False) -> str:
        """Advance state machine by one frame with debouncing / minimum-dwell enforcement."""
        if not self.initialized:
            self.current_state = proposed_state
            self.initialized = True
            self.candidate_state = None
            self.candidate_streak = 0
            return self.current_state

        if proposed_state == self.current_state:
            if self.candidate_streak > 0:
                self.flickers_suppressed += 1
            self.candidate_state = None
            self.candidate_streak = 0
            return self.current_state

        if immediate_override:
            self.current_state = proposed_state
            self.candidate_state = None
            self.candidate_streak = 0
            return self.current_state

        if proposed_state == self.candidate_state:
            self.candidate_streak += 1
        else:
            if self.candidate_streak > 0:
                self.flickers_suppressed += 1
            self.candidate_state = proposed_state
            self.candidate_streak = 1

        required_dwell = self.custom_dwell_overrides.get(
            (self.current_state, proposed_state),
            self.min_dwell_frames,
        )
        if self.candidate_streak >= max(int(required_dwell), 1):
            self.current_state = proposed_state
            self.candidate_state = None
            self.candidate_streak = 0

        return self.current_state


@dataclass
class FrameKeypointSnapshot:
    """Single-frame scale-normalized keypoint snapshot stored in an employee's 30-frame deque."""
    timestamp_sec: float
    bio_scale_px: float
    torso_x_px: float
    torso_y_px: float
    sh_x_norm: float
    sh_dist_bottom_norm: float
    spine_to_shoulder_ratio: float
    bbox_to_shoulder_ratio: float
    bbox_aspect_h_w: float
    lw_rel_sw: Tuple[float, float]
    rw_rel_sw: Tuple[float, float]
    lw_vel_sw_s: float
    rw_vel_sw_s: float
    dom_wrist_vel_sw_s: float
    wrist_pixel_motion_ratio: float
    lk_wrist_flow_sw_s: float
    nose_rel_sw: Tuple[float, float]
    nose_valid: bool
    min_ear_dist_sw: float
    min_face_dist_sw: float
    head_pitch_sw: float
    head_sym_sw: float


@dataclass
class EmployeeMotionState:
    """Isolated temporal history, rolling 30-frame deques, and debounced state machines for one `track_id`."""
    track_id: int
    window_size: int = 30
    min_dwell_frames: int = 15
    fps: float = 30.0

    history: Deque[FrameKeypointSnapshot] = field(init=False)
    bio_scale_window: Deque[float] = field(init=False)
    elev_baseline_window: Deque[float] = field(init=False)
    wrist_vert_window_lw: Deque[float] = field(init=False)
    wrist_vert_window_rw: Deque[float] = field(init=False)

    posture_fsm: HysteresisStateMachine = field(init=False)
    walking_fsm: HysteresisStateMachine = field(init=False)
    activity_fsm: HysteresisStateMachine = field(init=False)
    phone_streak: int = 0
    focus_streak: int = 0
    drink_cooldown: int = 0

    latest_features: Dict[str, float] = field(default_factory=dict)
    posture_state: str = "Sitting"
    activity_state: str = "Idle"
    motion_state: str = "Sitting | Idle"

    def __post_init__(self) -> None:
        w = max(int(self.window_size), 2)
        dwell = max(int(self.min_dwell_frames), 1)
        idle_confirm_frames = max(int(round(5.0 * max(self.fps, 1.0))), dwell)

        self.history = deque(maxlen=w)
        self.bio_scale_window = deque(maxlen=w)
        self.elev_baseline_window = deque(maxlen=max(w * 6, 180))
        self.wrist_vert_window_lw = deque(maxlen=max(w * 2, 60))
        self.wrist_vert_window_rw = deque(maxlen=max(w * 2, 60))

        self.posture_fsm = HysteresisStateMachine(
            initial_state="Sitting",
            min_dwell_frames=dwell,
        )
        self.walking_fsm = HysteresisStateMachine(
            initial_state="Stationary",
            min_dwell_frames=max(dwell // 2, 4),
        )
        self.activity_fsm = HysteresisStateMachine(
            initial_state="Idle",
            min_dwell_frames=dwell,
            custom_dwell_overrides={("Working", "Idle"): idle_confirm_frames},
        )


class ScaleInvariantMotionEngine:
    """Real-time multi-person feature & 6-class behavioral classification engine."""

    def __init__(
        self,
        window_size: int = 30,
        min_dwell_frames: int = 15,
        posture_threshold_norm: Optional[float] = None,
        enter_working_vel_sw_s: float = 0.25,
        exit_idle_vel_sw_s: float = 0.20,
        walking_vel_sw_s: float = 0.68,
    ) -> None:
        self.window_size = max(int(window_size), 2)
        self.min_dwell_frames = max(int(min_dwell_frames), 1)
        self.posture_threshold_norm = posture_threshold_norm
        self.enter_working_vel_sw_s = enter_working_vel_sw_s
        self.exit_idle_vel_sw_s = exit_idle_vel_sw_s
        self.walking_vel_sw_s = walking_vel_sw_s

        # Dictionary-based per-employee state management: {track_id: EmployeeMotionState}
        self.employee_states: Dict[int, EmployeeMotionState] = {}

    def reset(self) -> None:
        self.employee_states.clear()

    def update_observation(
        self,
        obs: SmoothedObservation,
        timestamp_sec: float,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        tid = int(obs.track_id)
        fps_safe = max(float(fps), 1.0)
        fw = max(float(frame_width), 1.0)
        fh = max(float(frame_height), 1.0)

        if tid not in self.employee_states:
            self.employee_states[tid] = EmployeeMotionState(
                track_id=tid,
                window_size=self.window_size,
                min_dwell_frames=self.min_dwell_frames,
                fps=fps_safe,
            )
        state = self.employee_states[tid]

        lms = obs.landmarks
        ls = lms["left_shoulder"]
        rs = lms["right_shoulder"]
        lh = lms["left_hip"]
        rh = lms["right_hip"]
        lw = lms["left_wrist"]
        rw = lms["right_wrist"]
        nose = lms["nose"]
        le = lms["left_eye"]
        re = lms["right_eye"]
        lear = lms["left_ear"]
        rear = lms["right_ear"]

        sh_x = 0.5 * (ls.x + rs.x)
        sh_y = 0.5 * (ls.y + rs.y)
        has_hips = lh.conf >= 0.12 and rh.conf >= 0.12
        hip_x = 0.5 * (lh.x + rh.x) if has_hips else sh_x
        hip_y = 0.5 * (lh.y + rh.y) if has_hips else (sh_y + 0.22 * fh)

        raw_sh_w = float(np.hypot(ls.x - rs.x, ls.y - rs.y))
        spine_len = float(np.hypot(hip_x - sh_x, hip_y - sh_y))
        bx1, by1, bx2, by2 = obs.bbox
        bh = max(by2 - by1, 1e-3)
        bw = max(bx2 - bx1, 1e-3)

        if ls.conf >= 0.25 and rs.conf >= 0.25 and raw_sh_w > 0.0:
            state.bio_scale_window.append(raw_sh_w)

        med_sh = float(np.median(state.bio_scale_window)) if state.bio_scale_window else raw_sh_w
        bio_s = max(med_sh, raw_sh_w, 0.68 * spine_len, 0.22 * bh, 0.30 * bw, 1e-3)

        sh_x_norm = sh_x / fw
        sh_dist_bot_norm = (fh - sh_y) / fh
        spine_ratio = spine_len / bio_s
        bbox_ratio = bh / bio_s
        bbox_aspect = bh / max(bw, 1e-3)
        torso_x = 0.5 * (sh_x + hip_x) if has_hips else sh_x
        torso_y = 0.5 * (sh_y + hip_y) if has_hips else sh_y

        lw_rel = ((lw.x - sh_x) / bio_s, (lw.y - sh_y) / bio_s)
        rw_rel = ((rw.x - sh_x) / bio_s, (rw.y - sh_y) / bio_s)
        nose_rel = ((nose.x - sh_x) / bio_s, (nose.y - sh_y) / bio_s)

        lw_vel_sw_s = 0.0
        rw_vel_sw_s = 0.0
        if state.history:
            prev = state.history[-1]
            dt = timestamp_sec - prev.timestamp_sec
            if 0.0 < dt <= (3.5 / fps_safe):
                inst_torso_sw_s = (float(np.hypot(torso_x - prev.torso_x_px, torso_y - prev.torso_y_px)) / bio_s) / dt
                walk_blend = float(np.clip((inst_torso_sw_s - 0.30) / 0.30, 0.0, 1.0))

                if lw.conf >= 0.10:
                    prev_lw_x = prev.lw_rel_sw[0] * prev.bio_scale_px + (prev.sh_x_norm * fw)
                    prev_lw_y = prev.lw_rel_sw[1] * prev.bio_scale_px + (fh * (1.0 - prev.sh_dist_bottom_norm))
                    dl_raw_sw = float(np.hypot(lw.x - prev_lw_x, lw.y - prev_lw_y)) / bio_s
                    dl_rel_sw = float(np.hypot(lw_rel[0] - prev.lw_rel_sw[0], lw_rel[1] - prev.lw_rel_sw[1]))
                    lw_vel_sw_s = ((1.0 - walk_blend) * dl_raw_sw + walk_blend * dl_rel_sw) / dt

                if rw.conf >= 0.10:
                    prev_rw_x = prev.rw_rel_sw[0] * prev.bio_scale_px + (prev.sh_x_norm * fw)
                    prev_rw_y = prev.rw_rel_sw[1] * prev.bio_scale_px + (fh * (1.0 - prev.sh_dist_bottom_norm))
                    dr_raw_sw = float(np.hypot(rw.x - prev_rw_x, rw.y - prev_rw_y)) / bio_s
                    dr_rel_sw = float(np.hypot(rw_rel[0] - prev.rw_rel_sw[0], rw_rel[1] - prev.rw_rel_sw[1]))
                    rw_vel_sw_s = ((1.0 - walk_blend) * dr_raw_sw + walk_blend * dr_rel_sw) / dt

        px_mot = obs.wrist_pixel_motion_ratio
        lk_flow = obs.lk_wrist_flow_sw_s
        if px_mot is not None and lk_flow is not None:
            px_mot_f = float(px_mot)
            lk_flow_f = float(lk_flow)
            pixel_gate = float(np.clip(max(px_mot_f / 0.026, lk_flow_f / 0.13), 0.0, 1.0))
            lw_vel_sw_s *= pixel_gate
            rw_vel_sw_s *= pixel_gate
            px_boost = max(0.85 * lk_flow_f, 3.0 * px_mot_f if px_mot_f >= 0.042 else 0.0)
            dom_vel_sw_s = max(lw_vel_sw_s, rw_vel_sw_s, pixel_gate * px_boost)
        else:
            px_mot_f = 0.0
            lk_flow_f = 0.0
            dom_vel_sw_s = max(lw_vel_sw_s, rw_vel_sw_s)

        ear_candidates = [99.0]
        if lw.conf >= 0.12 and lear.conf >= 0.12:
            ear_candidates.append(float(np.hypot(lw.x - lear.x, lw.y - lear.y)) / bio_s)
        if lw.conf >= 0.12 and rear.conf >= 0.12:
            ear_candidates.append(float(np.hypot(lw.x - rear.x, lw.y - rear.y)) / bio_s)
        if rw.conf >= 0.12 and rear.conf >= 0.12:
            ear_candidates.append(float(np.hypot(rw.x - rear.x, rw.y - rear.y)) / bio_s)
        if rw.conf >= 0.12 and lear.conf >= 0.12:
            ear_candidates.append(float(np.hypot(rw.x - lear.x, rw.y - lear.y)) / bio_s)
        if nose.conf >= 0.15:
            if lw.conf >= 0.12:
                ear_candidates.append(1.15 * float(np.hypot(lw.x - nose.x, lw.y - nose.y)) / bio_s)
            if rw.conf >= 0.12:
                ear_candidates.append(1.15 * float(np.hypot(rw.x - nose.x, rw.y - nose.y)) / bio_s)
        min_ear_sw = min(ear_candidates)

        mouth_x, mouth_y = nose.x, nose.y + 0.16 * bio_s
        face_candidates = [99.0]
        if nose.conf >= 0.15 and lw.conf >= 0.12:
            face_candidates.append(float(np.hypot(lw.x - mouth_x, lw.y - mouth_y)) / bio_s)
        if nose.conf >= 0.15 and rw.conf >= 0.12:
            face_candidates.append(float(np.hypot(rw.x - mouth_x, rw.y - mouth_y)) / bio_s)
        min_face_sw = min(face_candidates)

        has_eyes = le.conf >= 0.15 and re.conf >= 0.15
        has_ears = lear.conf >= 0.15 and rear.conf >= 0.15
        eye_mx = 0.5 * (le.x + re.x) if has_eyes else sh_x
        eye_my = (
            0.5 * (le.y + re.y)
            if has_eyes
            else (0.5 * (lear.y + rear.y) if has_ears else (nose.y - 0.12 * bio_s))
        )
        head_pitch_sw = ((nose.y - eye_my) / bio_s) if nose.conf >= 0.18 else float("nan")
        head_sym_sw = (abs(nose.x - eye_mx) / bio_s) if nose.conf >= 0.18 else float("nan")

        snap = FrameKeypointSnapshot(
            timestamp_sec=timestamp_sec,
            bio_scale_px=bio_s,
            torso_x_px=torso_x,
            torso_y_px=torso_y,
            sh_x_norm=sh_x_norm,
            sh_dist_bottom_norm=sh_dist_bot_norm,
            spine_to_shoulder_ratio=spine_ratio,
            bbox_to_shoulder_ratio=bbox_ratio,
            bbox_aspect_h_w=bbox_aspect,
            lw_rel_sw=lw_rel,
            rw_rel_sw=rw_rel,
            lw_vel_sw_s=lw_vel_sw_s,
            rw_vel_sw_s=rw_vel_sw_s,
            dom_wrist_vel_sw_s=dom_vel_sw_s,
            wrist_pixel_motion_ratio=px_mot_f,
            lk_wrist_flow_sw_s=lk_flow_f,
            nose_rel_sw=nose_rel,
            nose_valid=(nose.conf >= 0.18),
            min_ear_dist_sw=min_ear_sw,
            min_face_dist_sw=min_face_sw,
            head_pitch_sw=head_pitch_sw,
            head_sym_sw=head_sym_sw,
        )
        state.history.append(snap)
        state.elev_baseline_window.append(sh_dist_bot_norm)
        state.wrist_vert_window_lw.append(lw_rel[1])
        state.wrist_vert_window_rw.append(rw_rel[1])

        win_len = len(state.history)
        mean_lw_sw_s = float(np.mean([s.lw_vel_sw_s for s in state.history]))
        mean_rw_sw_s = float(np.mean([s.rw_vel_sw_s for s in state.history]))
        mean_dom_sw_s = float(np.mean([s.dom_wrist_vel_sw_s for s in state.history]))

        lw_xs = np.array([s.lw_rel_sw[0] for s in state.history], dtype=np.float64)
        lw_ys = np.array([s.lw_rel_sw[1] for s in state.history], dtype=np.float64)
        rw_xs = np.array([s.rw_rel_sw[0] for s in state.history], dtype=np.float64)
        rw_ys = np.array([s.rw_rel_sw[1] for s in state.history], dtype=np.float64)

        lw_std_sw = float(np.sqrt(np.var(lw_xs) + np.var(lw_ys))) if win_len >= 2 else 0.0
        rw_std_sw = float(np.sqrt(np.var(rw_xs) + np.var(rw_ys))) if win_len >= 2 else 0.0
        dom_wrist_std_sw = max(lw_std_sw, rw_std_sw)

        if win_len >= 2:
            k_back = min(15, win_len - 1)
            s_old = state.history[-1 - k_back]
            dt_walk = max(snap.timestamp_sec - s_old.timestamp_sec, 1.0 / fps_safe)
            horiz_vel_sw_s = (abs(snap.torso_x_px - s_old.torso_x_px) / bio_s) / dt_walk
            torso_2d_vel_sw_s = (
                float(np.hypot(snap.torso_x_px - s_old.torso_x_px, snap.torso_y_px - s_old.torso_y_px)) / bio_s
            ) / dt_walk
            tx_arr = np.array([s.torso_x_px for s in state.history], dtype=np.float64)
            ty_arr = np.array([s.torso_y_px for s in state.history], dtype=np.float64)
            horiz_span_30f_sw = float(np.max(tx_arr) - np.min(tx_arr)) / bio_s
            torso_2d_span_sw = float(np.hypot(np.max(tx_arr) - np.min(tx_arr), np.max(ty_arr) - np.min(ty_arr))) / bio_s
        else:
            horiz_vel_sw_s = 0.0
            torso_2d_vel_sw_s = 0.0
            horiz_span_30f_sw = 0.0
            torso_2d_span_sw = 0.0

        nose_pts = [(s.nose_rel_sw[0], s.nose_rel_sw[1]) for s in state.history if s.nose_valid]
        if len(nose_pts) >= 3:
            nx_a = np.array([p[0] for p in nose_pts], dtype=np.float64)
            ny_a = np.array([p[1] for p in nose_pts], dtype=np.float64)
            gaze_std_sw = float(np.sqrt(np.var(nx_a) + np.var(ny_a)))
        else:
            gaze_std_sw = 0.0

        med_bbox_r = float(np.median([s.bbox_to_shoulder_ratio for s in state.history]))
        med_aspect = float(np.median([s.bbox_aspect_h_w for s in state.history]))
        is_full_upright = med_bbox_r >= 3.48 and med_aspect >= 2.25
        if self.posture_threshold_norm is not None:
            p_cut = self.posture_threshold_norm
        else:
            elev_arr = np.array(state.elev_baseline_window, dtype=np.float64)
            p_lo, p_hi = float(np.percentile(elev_arr, 15.0)), float(np.percentile(elev_arr, 90.0))
            if (p_hi - p_lo) >= 0.06:
                p_cut = 0.5 * (p_lo + p_hi)
            else:
                # Unimodal per-employee elevation: distinguish full-upright Standing vs desk Sitting anatomically
                if is_full_upright:
                    p_cut = max(sh_dist_bot_norm - 0.06, 0.15)
                else:
                    p_cut = min(sh_dist_bot_norm + 0.065, 0.95)

        enter_stand = p_cut * 1.03
        exit_sit = p_cut * 0.97
        curr_p = state.posture_fsm.current_state
        if not state.posture_fsm.initialized:
            prop_posture = "Standing" if (sh_dist_bot_norm >= p_cut or is_full_upright) else "Sitting"
        elif curr_p == "Sitting":
            prop_posture = "Standing" if (sh_dist_bot_norm >= enter_stand or is_full_upright) else "Sitting"
        else:
            prop_posture = "Sitting" if (sh_dist_bot_norm <= exit_sit and not is_full_upright) else "Standing"
        debounced_posture = state.posture_fsm.step(prop_posture)

        curr_w = state.walking_fsm.current_state
        walk_hi = self.walking_vel_sw_s
        walk_lo = 0.80 * walk_hi
        eff_loco_vel = max(horiz_vel_sw_s, torso_2d_vel_sw_s if horiz_span_30f_sw >= 0.35 else 0.0)
        if curr_w == "Stationary":
            prop_walk = "Walking" if eff_loco_vel >= walk_hi else "Stationary"
        else:
            prop_walk = "Stationary" if eff_loco_vel <= walk_lo else "Walking"
        debounced_walk = state.walking_fsm.step(prop_walk)

        is_walking = (
            (
                debounced_posture == "Standing"
                and debounced_walk == "Walking"
                and (horiz_span_30f_sw >= 0.42 or torso_2d_span_sw >= 1.55)
            )
            or (horiz_vel_sw_s >= 1.25 and horiz_span_30f_sw >= 1.35)
        )
        final_posture = "Walking" if is_walking else debounced_posture

        ear_short_std = min(lw_std_sw, rw_std_sw)
        if not is_walking and min_ear_sw <= 0.58 and ear_short_std <= 0.09:
            state.phone_streak += 1
        else:
            state.phone_streak = 0
        is_phone = state.phone_streak >= max(int(round(1.2 * fps_safe)), 12)

        lw_ptp = (max(state.wrist_vert_window_lw) - min(state.wrist_vert_window_lw)) if len(state.wrist_vert_window_lw) >= 4 else 0.0
        rw_ptp = (max(state.wrist_vert_window_rw) - min(state.wrist_vert_window_rw)) if len(state.wrist_vert_window_rw) >= 4 else 0.0
        face_ptp = max(lw_ptp, rw_ptp)
        if not is_walking and not is_phone and min_face_sw <= 0.48 and face_ptp >= 0.55:
            state.drink_cooldown = 10
        elif state.drink_cooldown > 0:
            state.drink_cooldown -= 1
        is_drinking = state.drink_cooldown > 0

        wrist_active = (
            mean_dom_sw_s >= self.enter_working_vel_sw_s
            if state.activity_fsm.current_state == "Idle"
            else mean_dom_sw_s > self.exit_idle_vel_sw_s
        )

        tilted_down = np.isfinite(head_pitch_sw) and head_pitch_sw >= 0.22 and gaze_std_sw <= 0.09
        locked_fwd = (
            np.isfinite(head_pitch_sw)
            and head_pitch_sw >= 0.17
            and np.isfinite(head_sym_sw)
            and head_sym_sw <= 0.14
            and gaze_std_sw <= 0.05
        )
        if not is_walking and not is_phone and not is_drinking and not wrist_active and (tilted_down or locked_fwd):
            state.focus_streak += 1
        else:
            state.focus_streak = 0
        is_focused = state.focus_streak >= self.min_dwell_frames

        if is_drinking:
            proposed_act = "Idle"
        elif wrist_active or is_phone or is_focused:
            proposed_act = "Working"
        else:
            proposed_act = "Idle"

        debounced_act = state.activity_fsm.step(proposed_act, immediate_override=is_drinking)

        if final_posture == "Walking":
            final_act = "Walking"
            motion_lbl = "Walking"
        else:
            final_act = debounced_act
            motion_lbl = f"{final_posture} | {final_act}"

        state.posture_state = final_posture
        state.activity_state = final_act
        state.motion_state = motion_lbl
        state.latest_features = {
            "bio_shoulder_scale_px": round(bio_s, 3),
            "shoulder_dist_bottom_norm": round(sh_dist_bot_norm, 4),
            "spine_to_shoulder_ratio": round(spine_ratio, 4),
            "rolling_30f_left_wrist_vel_sw_s": round(mean_lw_sw_s, 4),
            "rolling_30f_right_wrist_vel_sw_s": round(mean_rw_sw_s, 4),
            "rolling_30f_wrist_vel_sw_s": round(mean_dom_sw_s, 4),
            "rolling_30f_wrist_std_sw": round(dom_wrist_std_sw, 4),
            "torso_horiz_vel_sw_s": round(horiz_vel_sw_s, 4),
            "torso_horiz_span_30f_sw": round(horiz_span_30f_sw, 4),
        }

        obs.posture_state = final_posture
        obs.activity_state = final_act
        obs.behavior_label = motion_lbl

        return {
            "Track_ID": tid,
            "posture_state": final_posture,
            "activity_state": final_act,
            "motion_state": motion_lbl,
            **state.latest_features,
        }
