import time

import cv2
import numpy as np

from hand_tracker import HAND_LINES, HandTracker

COLOR = (255, 0, 255)  # BGR
ERASE_RADIUS = 25
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
tracker = HandTracker()
canvas = None
prev = None
wand = False
latched = False  # stops one fist hold from toggling repeatedly
start = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    hands = tracker.detect(frame, (time.time() - start) * 1000)
    fists = [h for h in hands if not any(up(h, t) for t in (8, 12, 16, 20))]
    palms = [h for h in hands if all(up(h, t) for t in (8, 12, 16, 20))]

    toggle = hold("fist", bool(fists), TOGGLE_SECS)
    if toggle == 1 and not latched:
        wand, latched = not wand, True
    elif toggle == 0:
        latched = False
    clear = hold("clear", wand and len(palms) == 2, CLEAR_SECS)
    erasers = [  # index + middle up: erase at the point between the two fingertips
        ((h[8][0] + h[12][0]) // 2, (h[8][1] + h[12][1]) // 2)
        for h in hands
        if wand and up(h, 8) and up(h, 12) and not (up(h, 16) or up(h, 20))
    ]

    if clear == 1:
        canvas[:] = 0
    for e in erasers:
        cv2.circle(canvas, e, ERASE_RADIUS, 0, cv2.FILLED)
    pt = next((h[8] for h in hands if up(h, 8) and not any(up(h, t) for t in (12, 16, 20))), None)
    pt = pt if wand else None
    if pt and prev:
        cv2.line(canvas, prev, pt, COLOR, 8)
    prev = pt

    drawn = canvas.any(axis=2)
    frame[drawn] = canvas[drawn]
    for h in hands:
        for a, b in HAND_LINES:
            cv2.line(frame, h[a], h[b], (0, 255, 0), 2)
        for p in h:
            cv2.circle(frame, p, 4, (0, 0, 255), cv2.FILLED)
    if toggle:
        ring(frame, fists[0][9], toggle)
    if 0 < clear < 1:
        for h in palms:
            ring(frame, h[9], clear)
    for e in erasers:
        cv2.circle(frame, e, ERASE_RADIUS, (255, 255, 255), 2)
    label, color = ("WAND ON", (0, 255, 0)) if wand else ("WAND OFF", (0, 0, 255))
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("MahiKamay", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()
