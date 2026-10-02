import cv2

class TrackResult:
    def __init__(self):
        self.gaze_x = 0
        self.gaze_y = 0
        self.blink_left = False
        self.blink_right = False
        self.double_blink = False

def detect_pupil(eye):
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x + w//2, y + h//2

class PupilTracker:
    def __init__(self):
        self.eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eye.detectMultiScale(gray, 1.3, 5)

        if len(eyes) == 0:
            return None

        x, y, w, h = eyes[0]
        eye_img = frame[y:y+h, x:x+w]

        pupil = detect_pupil(eye_img)
        if pupil is None:
            return None

        px, py = pupil

        result = TrackResult()
        result.gaze_x = px/w - 0.5
        result.gaze_y = py/h - 0.5

        return result