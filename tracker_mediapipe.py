"""
tracker.py — Face mesh, EAR computation, gaze estimation, and blink detection.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from config import Config
from utils import compute_ear, get_iris_center, normalize_gaze


class TrackResult:
    """Data transfer object for one processed frame."""

    def __init__(self):
        self.ear_left: float = 0.0
        self.ear_right: float = 0.0
        self.gaze_x: float = 0.0       # normalised -1 (left) … +1 (right)
        self.gaze_y: float = 0.0       # normalised -1 (up)   … +1 (down)
        self.blink_left: bool = False
        self.blink_right: bool = False
        self.double_blink: bool = False
        self.landmarks = None          # raw NormalizedLandmarkList


class EyeTracker:
    # MediaPipe landmark indices for each eye
    # Left eye  (from user's perspective — mirrored = right side of image)
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

    # Iris centre landmarks (requires refine_landmarks=True)
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    # Eye corners for gaze normalisation
    LEFT_EYE_CORNERS = [362, 263]
    RIGHT_EYE_CORNERS = [33, 133]

    def __init__(self):
        self._mp_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_mesh.FaceMesh(
            max_num_faces=Config.MAX_FACES,
            refine_landmarks=Config.REFINE_LANDMARKS,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
        )

        # Blink state tracking
        self._left_blink_counter = 0
        self._right_blink_counter = 0
        self._last_blink_time = 0.0
        self._blink_pending = False

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> "TrackResult | None":
        """Process one BGR frame; returns TrackResult or None if no face found."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = self._face_mesh.process(rgb)

        if not mesh_result.multi_face_landmarks:
            return None

        lm = mesh_result.multi_face_landmarks[0].landmark
        result = TrackResult()
        result.landmarks = mesh_result.multi_face_landmarks[0]

        # ── EAR ─────────────────────────────────────────────────────────────
        result.ear_left = compute_ear(lm, self.LEFT_EYE_EAR, w, h)
        result.ear_right = compute_ear(lm, self.RIGHT_EYE_EAR, w, h)

        # ── Blink detection ─────────────────────────────────────────────────
        result.blink_left, self._left_blink_counter = self._detect_blink(
            result.ear_left, self._left_blink_counter
        )
        result.blink_right, self._right_blink_counter = self._detect_blink(
            result.ear_right, self._right_blink_counter
        )
        result.double_blink = self._detect_double_blink(result.blink_left or result.blink_right)

        # ── Gaze estimation (iris-based) ─────────────────────────────────────
        try:
            left_iris_center = get_iris_center(lm, self.LEFT_IRIS, w, h)
            right_iris_center = get_iris_center(lm, self.RIGHT_IRIS, w, h)

            left_corners = [
                (int(lm[self.LEFT_EYE_CORNERS[0]].x * w), int(lm[self.LEFT_EYE_CORNERS[0]].y * h)),
                (int(lm[self.LEFT_EYE_CORNERS[1]].x * w), int(lm[self.LEFT_EYE_CORNERS[1]].y * h)),
            ]
            right_corners = [
                (int(lm[self.RIGHT_EYE_CORNERS[0]].x * w), int(lm[self.RIGHT_EYE_CORNERS[0]].y * h)),
                (int(lm[self.RIGHT_EYE_CORNERS[1]].x * w), int(lm[self.RIGHT_EYE_CORNERS[1]].y * h)),
            ]

            gaze_l = normalize_gaze(left_iris_center, left_corners)
            gaze_r = normalize_gaze(right_iris_center, right_corners)

            # Average both eyes
            result.gaze_x = (gaze_l[0] + gaze_r[0]) / 2.0
            result.gaze_y = (gaze_l[1] + gaze_r[1]) / 2.0
        except Exception:
            pass  # Iris landmarks unavailable; gaze stays 0,0

        return result

    def draw_debug(self, frame: np.ndarray, result: "TrackResult"):
        if not Config.SHOW_LANDMARKS and not Config.SHOW_EAR_VALUE:
            return

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = Config.FONT_COLOR

        if Config.SHOW_EAR_VALUE:
            cv2.putText(frame, f"EAR L:{result.ear_left:.2f} R:{result.ear_right:.2f}",
                        (10, 20), font, Config.FONT_SCALE, color, Config.FONT_THICKNESS)
            cv2.putText(frame, f"Gaze X:{result.gaze_x:.2f} Y:{result.gaze_y:.2f}",
                        (10, 42), font, Config.FONT_SCALE, color, Config.FONT_THICKNESS)

        if result.blink_left:
            cv2.putText(frame, "LEFT BLINK", (10, 64), font, Config.FONT_SCALE, (0, 120, 255), 2)
        if result.blink_right:
            cv2.putText(frame, "RIGHT BLINK", (10, 86), font, Config.FONT_SCALE, (0, 120, 255), 2)
        if result.double_blink:
            cv2.putText(frame, "DOUBLE BLINK", (10, 108), font, 0.7, (0, 0, 255), 2)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _detect_blink(self, ear: float, counter: int) -> "tuple[bool, int]":
        """Returns (did_blink_this_frame, updated_counter)."""
        if ear < Config.EAR_BLINK_THRESHOLD:
            counter += 1
            return False, counter
        else:
            if counter >= Config.EAR_CONSECUTIVE_FRAMES:
                return True, 0
            return False, 0

    def _detect_double_blink(self, blinked: bool) -> bool:
        now = time.time()
        if blinked:
            if self._blink_pending and (now - self._last_blink_time) <= Config.DOUBLE_BLINK_INTERVAL:
                self._blink_pending = False
                return True
            self._last_blink_time = now
            self._blink_pending = True
        return False
