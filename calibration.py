"""
calibration.py — Optional per-user gaze calibration.

Run standalone:  python calibration.py
Saves offsets to calibration_data.json which main.py loads automatically.
"""

import cv2
import json
import time
import numpy as np
import mediapipe as mp
from pathlib import Path
from utils import compute_ear, get_iris_center, normalize_gaze
from config import Config

CALIB_FILE = Path("calibration_data.json")

# Screen positions to look at (normalised 0–1)
TARGETS = [
    (0.5, 0.5),  # centre
    (0.1, 0.1),  # top-left
    (0.9, 0.1),  # top-right
    (0.1, 0.9),  # bottom-left
    (0.9, 0.9),  # bottom-right
]

SAMPLES_PER_TARGET = 40
HOLD_SECONDS = 2


def run_calibration():
    import pyautogui
    sw, sh = pyautogui.size()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    mp_mesh = mp.solutions.face_mesh
    face_mesh = mp_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_CORNERS  = [362, 263]
    RIGHT_CORNERS = [33, 133]

    collected_gaze  = []
    collected_screen = []

    for tx, ty in TARGETS:
        screen_pt = (int(tx * sw), int(ty * sh))
        samples = []
        start = time.time()

        print(f"[CALIB] Look at ({tx:.0%}, {ty:.0%}) of screen — hold for {HOLD_SECONDS}s")

        while len(samples) < SAMPLES_PER_TARGET:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            # Draw target dot
            tx_px = int(tx * w)
            ty_px = int(ty * h)
            cv2.circle(frame, (tx_px, ty_px), 20, (0, 255, 0), -1)
            cv2.putText(frame, f"Hold {HOLD_SECONDS - int(time.time() - start)}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)

            if time.time() - start < HOLD_SECONDS:
                continue

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                try:
                    li = get_iris_center(lm, LEFT_IRIS, w, h)
                    ri = get_iris_center(lm, RIGHT_IRIS, w, h)
                    lc = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_CORNERS]
                    rc = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_CORNERS]
                    gl = normalize_gaze(li, lc)
                    gr = normalize_gaze(ri, rc)
                    gaze = ((gl[0] + gr[0]) / 2, (gl[1] + gr[1]) / 2)
                    samples.append(gaze)
                except Exception:
                    pass

        avg_gaze = np.mean(samples, axis=0).tolist()
        collected_gaze.append(avg_gaze)
        collected_screen.append([tx, ty])

    # Compute simple linear offsets
    gaze_arr = np.array(collected_gaze)
    screen_arr = np.array(collected_screen)
    offset_x = float(np.mean(screen_arr[:, 0] - gaze_arr[:, 0]))
    offset_y = float(np.mean(screen_arr[:, 1] - gaze_arr[:, 1]))

    data = {"offset_x": offset_x, "offset_y": offset_y}
    CALIB_FILE.write_text(json.dumps(data, indent=2))
    print(f"[CALIB] Done. Offsets saved to {CALIB_FILE}: {data}")

    cap.release()
    cv2.destroyAllWindows()


def load_calibration() -> dict:
    if CALIB_FILE.exists():
        return json.loads(CALIB_FILE.read_text())
    return {"offset_x": 0.0, "offset_y": 0.0}


if __name__ == "__main__":
    run_calibration()
