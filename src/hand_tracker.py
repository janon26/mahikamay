"""Thin wrapper around MediaPipe Hands."""
import cv2
import mediapipe as mp

TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.5):
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._draw = mp.solutions.drawing_utils
        self.results = None

    def process(self, frame_bgr):
        self.results = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return self.results

    def landmarks_px(self, frame, hand_idx=0):
        """Return list of (x, y) pixel coords for one hand, or [] if none."""
        if not self.results or not self.results.multi_hand_landmarks:
            return []
        h, w = frame.shape[:2]
        hand = self.results.multi_hand_landmarks[hand_idx]
        return [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]

    def fingers_up(self, lm):
        """[thumb, index, middle, ring, pinky] -> 1 if up. Rough: tip above PIP joint."""
        if not lm:
            return [0] * 5
        # TODO: thumb check depends on handedness; rough x-compare for now (mirrored frame)
        thumb = int(lm[4][0] > lm[3][0])
        others = [int(lm[t][1] < lm[t - 2][1]) for t in TIP_IDS[1:]]
        return [thumb] + others

    def draw_landmarks(self, frame):
        if self.results and self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                self._draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
