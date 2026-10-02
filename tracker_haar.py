import cv2

class TrackResult:
    def __init__(self):
        self.gaze_x = 0
        self.gaze_y = 0
        self.blink_left = False
        self.blink_right = False
        self.double_blink = False

class HaarTracker:
    def __init__(self):
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        eyes = self.eyes.detectMultiScale(roi)

        if len(eyes) == 0:
            return None

        ex, ey, ew, eh = eyes[0]

        result = TrackResult()
        result.gaze_x = (ex + ew/2)/w - 0.5
        result.gaze_y = (ey + eh/2)/h - 0.5

        return result