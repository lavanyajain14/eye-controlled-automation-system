"""
config.py — Centralized configuration for the Eye Control System.
Tune these values to adapt to different users and environments.
"""

import pyautogui

# Suppress PyAutoGUI fail-safe in debug mode (keep True in production)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # No delay between PyAutoGUI calls for real-time feel

TRACKER_TYPE = "mediapipe"
# options: mediapipe, haar, ear, pupil, cnn

class Config:
    # ── Camera ──────────────────────────────────────────────────────────────
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30

    # ── MediaPipe Face Mesh ──────────────────────────────────────────────────
    MAX_FACES = 1
    REFINE_LANDMARKS = True          # enables iris landmarks (468–477)
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7

    # ── EAR (Eye Aspect Ratio) ───────────────────────────────────────────────
    EAR_BLINK_THRESHOLD = 0.21       # below this → eye closed
    EAR_CONSECUTIVE_FRAMES = 2       # frames eye must be closed to register blink
    DOUBLE_BLINK_INTERVAL = 0.4      # seconds between blinks for double-blink detection

    # ── Gaze / Cursor ────────────────────────────────────────────────────────
    GAZE_SENSITIVITY_X = 6.0        # horizontal multiplier
    GAZE_SENSITIVITY_Y = 6.5       # vertical multiplier
    SMOOTHING_FACTOR = 0.12        # EMA weight (0=no movement, 1=raw movement)
    SCREEN_MARGIN = 20              # pixels from screen edge to stop

    # ── Scroll ───────────────────────────────────────────────────────────────
    SCROLL_GAZE_THRESHOLD = 0.60    # normalised gaze-y to trigger scroll
    SCROLL_AMOUNT = 2               # scroll clicks per trigger
    SCROLL_COOLDOWN = 1.0           # seconds between scrolls

    # ── Dwell Click ──────────────────────────────────────────────────────────
    DWELL_ENABLED = True
    DWELL_TIME = 1.2                # seconds to dwell before auto-click
    DWELL_RADIUS = 35               # pixels cursor must stay within to count as dwell

    # ── Debug / UI ───────────────────────────────────────────────────────────
    SHOW_LANDMARKS = True
    SHOW_EAR_VALUE = True
    SHOW_GAZE_VECTOR = True
    FONT = 0                        # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55
    FONT_COLOR = (0, 255, 120)
    FONT_THICKNESS = 1
