"""
Eye-Controlled Automation System
Entry point — initializes camera, tracker, and action dispatcher.
"""

import cv2
import sys
from actions import ActionDispatcher
from config import Config

if Config.TRACKER_TYPE == "mediapipe":
    from tracker_mediapipe import EyeTracker as Tracker
elif Config.TRACKER_TYPE == "haar":
    from tracker_haar import HaarTracker as Tracker
elif Config.TRACKER_TYPE == "ear":
    from tracker_ear import EyeTracker as Tracker
elif Config.TRACKER_TYPE == "pupil":
    from tracker_pupil import PupilTracker as Tracker
elif Config.TRACKER_TYPE == "cnn":
    from tracker_cnn import CNNTracker as Tracker

tracker = Tracker()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.FPS)

    tracker = Tracker()
    dispatcher = ActionDispatcher()

    print("[INFO] Eye Control System started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame.")
            continue

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        result = tracker.process(frame)

        if result:
            dispatcher.dispatch(result, frame)
            tracker.draw_debug(frame, result)

        cv2.imshow("Eye Control System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()