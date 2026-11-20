"""Thin wrapper around the MediaPipe Tasks HandLandmarker."""
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky


def ensure_model():
    """Download the hand landmark model on first run."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.results = None

    def process(self, frame_bgr, timestamp_ms):
        """Run detection. timestamp_ms must increase on every call."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.results = self._landmarker.detect_for_video(image, int(timestamp_ms))
        return self.results

    def landmarks_px(self, frame, hand_idx=0):
        """Return list of (x, y) pixel coords for one hand, or [] if none."""
        if not self.results or len(self.results.hand_landmarks) <= hand_idx:
            return []
        h, w = frame.shape[:2]
        return [(int(p.x * w), int(p.y * h)) for p in self.results.hand_landmarks[hand_idx]]

    def fingers_up(self, lm):
        """[thumb, index, middle, ring, pinky] -> 1 if up. Rough: tip above PIP joint."""
        if not lm:
            return [0] * 5
        # TODO: thumb check depends on handedness; rough x-compare for now (mirrored frame)
        thumb = int(lm[4][0] > lm[3][0])
        others = [int(lm[t][1] < lm[t - 2][1]) for t in TIP_IDS[1:]]
        return [thumb] + others

    def draw_landmarks(self, frame):
        if not self.results:
            return
        h, w = frame.shape[:2]
        for hand in self.results.hand_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in hand]
            for c in self._connections:
                cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 0, 255), cv2.FILLED)
