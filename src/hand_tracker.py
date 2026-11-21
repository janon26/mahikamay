import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

HAND_LINES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


class HandTracker:
    def __init__(self):
        if not MODEL.exists():
            MODEL.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(URL, MODEL)
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )

    def detect(self, frame, ms):
        """Return {"Left": points, "Right": points} for the hands seen (21 pixel (x, y) points each).

        Expects a mirrored (selfie-view) frame, so the labels match your real hands.
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        found = self.landmarker.detect_for_video(image, int(ms))
        return {
            side[0].category_name: [(int(p.x * w), int(p.y * h)) for p in pts]
            for side, pts in zip(found.handedness, found.hand_landmarks)
        }
