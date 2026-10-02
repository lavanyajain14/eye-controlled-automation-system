import numpy as np

class TrackResult:
    def __init__(self):
        self.gaze_x = 0
        self.gaze_y = 0
        self.blink_left = False
        self.blink_right = False
        self.double_blink = False

class CNNTracker:
    def process(self, frame):
        result = TrackResult()

        # Simulated output (for demo)
        result.gaze_x = np.random.uniform(-0.5, 0.5)
        result.gaze_y = np.random.uniform(-0.5, 0.5)

        return result