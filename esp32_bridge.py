"""
esp32_bridge.py — Optional serial bridge to send eye-gesture commands to an ESP32.

Wiring: USB cable from PC to ESP32.
Arduino side: read Serial, parse command strings, drive GPIO / servos.

Usage: pass an EyeSerialBridge instance to ActionDispatcher and call
       bridge.send(result) inside the dispatch loop.
"""

import serial
import serial.tools.list_ports
import time


class EyeSerialBridge:
    BAUD = 115200

    COMMANDS = {
        "blink_right":  "CLICK\n",
        "blink_left":   "RCLICK\n",
        "double_blink": "DCLICK\n",
        "gaze_up":      "SCROLL_UP\n",
        "gaze_down":    "SCROLL_DOWN\n",
        "gaze_left":    "MOVE_LEFT\n",
        "gaze_right":   "MOVE_RIGHT\n",
    }

    def __init__(self, port: str = None, auto_detect: bool = True):
        self._ser = None
        port = port or (self._auto_detect() if auto_detect else None)
        if port:
            try:
                self._ser = serial.Serial(port, self.BAUD, timeout=0.1)
                time.sleep(2)  # let ESP32 reset
                print(f"[ESP32] Connected on {port}")
            except serial.SerialException as e:
                print(f"[ESP32] Could not open {port}: {e}")

    def send(self, result):
        """Translate TrackResult into serial commands for the ESP32."""
        if not self._ser or not self._ser.is_open:
            return

        if result.double_blink:
            self._write("DCLICK\n")
        elif result.blink_right:
            self._write("CLICK\n")
        elif result.blink_left:
            self._write("RCLICK\n")

        THRESH = 0.35
        if result.gaze_y < -THRESH:
            self._write("SCROLL_UP\n")
        elif result.gaze_y > THRESH:
            self._write("SCROLL_DOWN\n")

        if result.gaze_x < -THRESH:
            self._write("MOVE_LEFT\n")
        elif result.gaze_x > THRESH:
            self._write("MOVE_RIGHT\n")

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _write(self, cmd: str):
        try:
            self._ser.write(cmd.encode())
        except serial.SerialException:
            pass

    @staticmethod
    def _auto_detect() -> "str | None":
        for p in serial.tools.list_ports.comports():
            desc = p.description.lower()
            if "esp32" in desc or "cp210" in desc or "ch340" in desc or "ftdi" in desc:
                return p.device
        return None
