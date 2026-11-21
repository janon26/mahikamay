import math
import time

import cv2
import numpy as np

from hand_tracker import HAND_LINES, HandTracker

WINDOW, WINDOW_SIZE = "MahiKamay", (960, 720)
COLOR = (255, 0, 255)  # BGR
ERASE_RADIUS = 25
MIN_BRUSH, MAX_BRUSH = 2, 40
TOGGLE_SECS, CLEAR_SECS = 0.7, 1.0

since = {}  # when each gesture started being held


def up(hand, tip):
    return hand[tip][1] < hand[tip - 2][1]  # fingertip above its middle joint


def hold(name, active, secs):
    """0..1 progress of holding a gesture; 1 means it has been held long enough."""
    if not active:
        since.pop(name, None)
        return 0
    return min((time.time() - since.setdefault(name, time.time())) / secs, 1)


def ring(frame, center, progress):
    cv2.ellipse(frame, center, (35, 35), -90, 0, 360 * progress, (0, 255, 255), 4)


cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable: drag the corners to change the size
cv2.resizeWindow(WINDOW, *WINDOW_SIZE)
tracker = HandTracker()
canvas = None
prev = None
wand = False
latched = False  # stops one fist hold from toggling repeatedly
brush = 8.0
start = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    hands = tracker.detect(frame, (time.time() - start) * 1000)
    left, right = hands.get("Right"), hands.get("Left")  # swapped: MediaPipe's labels are mirrored here

    # Left hand: pinch thumb and index together or apart to set the brush size.
    # Only counts with the other three fingers folded; any other pose keeps the last size.
    if left and not any(up(left, t) for t in (12, 16, 20)):
        span = max(math.dist(left[0], left[9]), 1)  # hand size, so distance to camera matters less
        pinch = math.dist(left[4], left[8]) / span
        brush += 0.3 * (np.interp(pinch, (0.15, 1.0), (MIN_BRUSH, MAX_BRUSH)) - brush)

    # Right hand: all the gestures.
    index, middle, ring_, pinky = (right and up(right, t) for t in (8, 12, 16, 20))
    fist = bool(right) and not (index or middle or ring_ or pinky)
    both_palms = all(h and all(up(h, t) for t in (8, 12, 16, 20)) for h in (left, right))
    toggle = hold("fist", fist, TOGGLE_SECS)
    if toggle == 1 and not latched:
        wand, latched = not wand, True
    elif toggle == 0:
        latched = False
    clear = hold("clear", wand and both_palms, CLEAR_SECS)

    if clear == 1:
        canvas[:] = 0
    eraser = None
    if wand and index and middle and not (ring_ or pinky):  # point between the two fingertips
        eraser = ((right[8][0] + right[12][0]) // 2, (right[8][1] + right[12][1]) // 2)
        cv2.circle(canvas, eraser, ERASE_RADIUS, 0, cv2.FILLED)
    pt = right[8] if wand and index and not (middle or ring_ or pinky) else None
    if pt and prev:
        cv2.line(canvas, prev, pt, COLOR, int(brush))
    prev = pt

    drawn = canvas.any(axis=2)
    frame[drawn] = canvas[drawn]
    for pts in hands.values():
        for a, b in HAND_LINES:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), cv2.FILLED)
    if left:  # show the thumb-index line and the current brush size between the fingertips
        cv2.line(frame, left[4], left[8], (0, 255, 255), 2)
        mid = ((left[4][0] + left[8][0]) // 2, (left[4][1] + left[8][1]) // 2)
        cv2.circle(frame, mid, max(int(brush / 2), 1), COLOR, cv2.FILLED)
    if toggle:
        ring(frame, right[9], toggle)
    if 0 < clear < 1:
        for h in (left, right):
            ring(frame, h[9], clear)
    if eraser:
        cv2.circle(frame, eraser, ERASE_RADIUS, (255, 255, 255), 2)
    label, color = ("WAND ON", (0, 255, 0)) if wand else ("WAND OFF", (0, 0, 255))
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()
