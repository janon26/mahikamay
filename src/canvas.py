"""NumPy drawing layer that gets blended over the camera frame."""
import cv2
import numpy as np


class Canvas:
    def __init__(self, width, height):
        self.img = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev = None

    def draw_to(self, point, color=(255, 0, 255), thickness=8):
        if self.prev is None:
            self.prev = point
        cv2.line(self.img, self.prev, point, color, thickness)
        self.prev = point

    def lift(self):
        """Pen up: the next draw starts a new stroke."""
        self.prev = None

    def clear(self):
        self.img[:] = 0
        self.lift()

    def overlay(self, frame):
        """Paste non-black canvas pixels onto the frame."""
        mask = self.img.any(axis=2)
        out = frame.copy()
        out[mask] = self.img[mask]
        return out
