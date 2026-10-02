import time
import pyautogui
from config import Config


class ActionDispatcher:
    def __init__(self):
        self._screen_w, self._screen_h = pyautogui.size()

        self._cursor_x = self._screen_w / 2
        self._cursor_y = self._screen_h / 2

        self._last_scroll_time = 0.0
        self._dwell_start = None
        self._dwell_anchor = None
        self._dwell_fired = False
        self._last_right_click = 0.0

    def dispatch(self, result, frame=None):
        self._move_cursor(result)
        self._handle_clicks(result)
        self._handle_scroll(result)

        if Config.DWELL_ENABLED:
            self._handle_dwell()

    # ── Cursor movement ─────────────────────────

    def _move_cursor(self, result):
        """Ultra-stable cursor"""

        target_x = self._screen_w / 2 -  result.gaze_x * self._screen_w * 0.8
        target_y = self._screen_h / 2 + result.gaze_y * self._screen_h * 0.8

        target_x = max(Config.SCREEN_MARGIN, min(self._screen_w - Config.SCREEN_MARGIN, target_x))
        target_y = max(Config.SCREEN_MARGIN, min(self._screen_h - Config.SCREEN_MARGIN, target_y))

        alpha = 0.12
        new_x = alpha * target_x + (1 - alpha) * self._cursor_x
        new_y = alpha * target_y + (1 - alpha) * self._cursor_y

        # Dead zone
        if abs(new_x - self._cursor_x) < 6:
            new_x = self._cursor_x
        if abs(new_y - self._cursor_y) < 6:
            new_y = self._cursor_y

        # Speed limit
        max_step = 40
        dx = max(-max_step, min(max_step, new_x - self._cursor_x))
        dy = max(-max_step, min(max_step, new_y - self._cursor_y))

        self._cursor_x += dx
        self._cursor_y += dy

        pyautogui.moveTo(int(self._cursor_x), int(self._cursor_y))

    # ── Click handling ─────────────────────────

    def _handle_clicks(self, result):
        now = time.time()

        if result.double_blink:
            pyautogui.doubleClick()
            return

        if result.blink_right:
            pyautogui.click()

        if result.blink_left:
            if now - self._last_right_click > 0.5:
                pyautogui.rightClick()
                self._last_right_click = now

    # ── Scroll handling ───────────────────────

    def _handle_scroll(self, result):
        now = time.time()

        if now - self._last_scroll_time < Config.SCROLL_COOLDOWN:
            return

        if result.gaze_y < -Config.SCROLL_GAZE_THRESHOLD:
            pyautogui.scroll(Config.SCROLL_AMOUNT)
            self._last_scroll_time = now

        elif result.gaze_y > Config.SCROLL_GAZE_THRESHOLD:
            pyautogui.scroll(-Config.SCROLL_AMOUNT)
            self._last_scroll_time = now

    # ── Dwell click ───────────────────────────

    def _handle_dwell(self):
        now = time.time()
        cx, cy = int(self._cursor_x), int(self._cursor_y)

        if self._dwell_anchor is None:
            self._dwell_anchor = (cx, cy)
            self._dwell_start = now
            self._dwell_fired = False
            return

        ax, ay = self._dwell_anchor
        dist = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5

        if dist > Config.DWELL_RADIUS:
            self._dwell_anchor = (cx, cy)
            self._dwell_start = now
            self._dwell_fired = False

        elif not self._dwell_fired and (now - self._dwell_start) >= Config.DWELL_TIME:
            pyautogui.click()
            self._dwell_fired = True