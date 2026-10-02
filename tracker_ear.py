from tracker_mediapipe import EyeTracker as BaseTracker

class EyeTracker(BaseTracker):
    def process(self, frame):
        result = super().process(frame)
        if result is None:
            return None

        # Ensure EAR exists
        if not hasattr(result, "ear_left") or not hasattr(result, "ear_right"):
            return result

        # Stable EAR-based movement
        result.gaze_x = (result.ear_left - result.ear_right) * 3
        result.gaze_y = ((result.ear_left + result.ear_right) / 2 - 0.25) * 3

        return result