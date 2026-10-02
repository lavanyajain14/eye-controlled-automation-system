# Eye-Controlled Automation System

A real-time, hands-free human-computer interaction system that allows users to control their laptop using eye movements and blink gestures.

The system uses MediaPipe Face Mesh + Eye Aspect Ratio (EAR) to track gaze and detect blinks, enabling actions such as cursor movement, clicking, and scrolling — all using a standard webcam.

---

## Features

- Real-time eye tracking using webcam
- Cursor control using gaze direction
- Blink-based actions:
  - Right blink → Left click
  - Left blink → Right click
  - Double blink → Double click
- Scroll using vertical gaze
- Dwell-based auto click (optional)
- Smooth and stable cursor movement
- Multiple algorithm support:
  - MediaPipe (default)
  - EAR-based
  - Haar Cascade
  - Pupil detection
  - CNN (demo)

---

## How It Works

1. Webcam captures live video
2. Face landmarks detected using MediaPipe
3. Eye landmarks extracted
4. EAR (Eye Aspect Ratio) computed for blink detection :contentReference[oaicite:0]{index=0}
5. Iris position used for gaze estimation
6. Gaze mapped to screen coordinates
7. Actions executed using PyAutoGUI :contentReference[oaicite:1]{index=1}


