import math
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from hand_tracker import HAND_LINES, HandTracker

WINDOW = "MahiKamay"
CAMERA_SIZE = (1280, 720)  # the most this camera delivers
COLORS = [  # BGR: red, orange, yellow, green, blue, violet
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 200, 0), (255, 0, 0), (211, 0, 148),
]
# Sizes below are in pixels for a 640-wide frame; px() scales them to the real frame width.
ERASE_RADIUS = 25
TOGETHER = 0.3  # erase only if index and middle fingertips are closer than this fraction of palm length
MIN_BRUSH, MAX_BRUSH = 2, 40
TOGGLE_SECS, CLEAR_SECS = 0.7, 1.0
SMOOTHING_POINTS = 5
BRUSH_CONFIRM_SECS = 0.5
BRUSH_SMOOTHING = 0.2
BRUSH_STABLE_TOLERANCE = 3.0
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"

since = {}  # when each gesture started being held


def up(hand, tip):
    return hand[tip][1] < hand[tip - 2][1]  # fingertip above its middle joint


def index_fully_extended(hand):
    """Require both index joints to be straight before arming drawing."""
    return up(hand, 8) and math.dist(hand[5], hand[8]) > 2.2 * max(math.dist(hand[5], hand[6]), 1)


def hold(name, active, secs):
    """0..1 progress of holding a gesture; 1 means it has been held long enough."""
    if not active:
        since.pop(name, None)
        return 0
    return min((time.time() - since.setdefault(name, time.time())) / secs, 1)


scale = 1.0


def px(n):
    return max(round(n * scale), 1)


def load_settings():
    """Load saved brush preferences, falling back to the defaults if needed."""
    defaults = {"brush_size": 8.0, "color_index": 5}
    try:
        with SETTINGS_FILE.open(encoding="utf-8") as file:
            saved = json.load(file)
        brush_size = float(saved.get("brush_size", defaults["brush_size"]))
        color_index = int(saved.get("color_index", defaults["color_index"]))
        return {
            "brush_size": min(max(brush_size, MIN_BRUSH), MAX_BRUSH),
            "color_index": min(max(color_index, 0), len(COLORS) - 1),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return defaults


def save_settings(brush_size, color_index):
    """Save preferences without putting machine-specific state into Git."""
    try:
        with SETTINGS_FILE.open("w", encoding="utf-8") as file:
            json.dump({"brush_size": round(brush_size, 2), "color_index": color_index}, file, indent=2)
            file.write("\n")
    except OSError:
        # A read-only directory should not stop the drawing app from running.
        pass


def redraw_canvas(canvas, strokes):
    """Rebuild the canvas from the remaining strokes after an undo."""
    canvas[:] = 0
    for stroke in strokes:
        for start, end, color, thickness in stroke:
            cv2.line(canvas, start, end, color, thickness)


def ring(frame, center, progress):
    cv2.ellipse(frame, center, (px(35), px(35)), -90, 0, 360 * progress, (0, 255, 255), px(4), cv2.LINE_AA)


def arc(r, a0, a1, h):
    """Points on a circle of radius r around the bottom-left corner, angles in degrees from the bottom edge."""
    return [(int(r * math.cos(math.radians(a))), int(h - r * math.sin(math.radians(a)))) for a in np.linspace(a0, a1, 8)]


def draw_menu(frame, selected):
    """Quarter ring in the bottom-left corner: red along the left edge down to violet along the bottom edge."""
    overlay = frame.copy()
    for i, color in enumerate(COLORS):
        top, bottom = 90 - 15 * i, 90 - 15 * (i + 1)
        outer = px(130 if i == selected else 120)  # the chosen slice sticks out a little
        slice_ = np.array(arc(outer, bottom, top, frame.shape[0]) + arc(px(50), top, bottom, frame.shape[0]), np.int32)
        cv2.fillPoly(overlay, [slice_], color, cv2.LINE_AA)
        if i == selected:
            cv2.polylines(overlay, [slice_], True, (255, 255, 255), px(2), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)


HELP = [
    "WAND on/off: right fist, hold 0.7s",
    "DRAW: right index finger up",
    "ERASE: right index + middle up, together",
    "CLEAR: both palms open, hold 1s",
    "UNDO: u = remove the last stroke",
    "BRUSH SIZE: left thumb + index pinch",
    "   (other 3 fingers folded)",
    "COLOR: left pinky only up,",
    "   tilt it left/right to pick",
    "q = quit    spacebar = clear",
]
mouse = (-1, -1)


def on_mouse(event, x, y, flags, param):
    global mouse
    mouse = (x, y)


def draw_help(frame):
    """A hollow ? icon in the bottom-left corner; hovering it shows the instructions."""
    k = frame.shape[1] / 640  # layout was designed for a 640-wide frame
    center, r = (int(20 * k), frame.shape[0] - int(20 * k)), int(12 * k)
    white = (255, 255, 255)
    cv2.circle(frame, center, r, white, max(int(1.5 * k), 1), cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize("?", cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, 2)
    cv2.putText(frame, "?", (center[0] - tw // 2, center[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, white, max(int(1.5 * k), 1), cv2.LINE_AA)
    if math.dist(mouse, center) <= r:
        x0, y1 = int(8 * k), center[1] - r - int(8 * k)
        y0 = y1 - int((len(HELP) * 22 + 16) * k)
        frame[y0:y1, x0 : x0 + int(330 * k)] //= 3  # darken the area behind the text
        for i, line in enumerate(HELP):
            cv2.putText(frame, line, (x0 + int(10 * k), y0 + int((28 + i * 22) * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, white, 1, cv2.LINE_AA)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = width / 640
settings = load_settings()
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable: drag the corners to change the size
cv2.resizeWindow(WINDOW, width, height)
cv2.setMouseCallback(WINDOW, on_mouse)
tracker = HandTracker()
canvas = None
prev = None
strokes = []
current_stroke = None
wand = False
latched = False  # stops one fist hold from toggling repeatedly
draw_armed = False
brush_size = settings["brush_size"]  # stored in 640-wide logical pixels
brush = brush_size * scale
brush_candidate = None
brush_candidate_since = None
brush_candidate_confirmed = False
pinch_size = None
selected_color = settings["color_index"]
ink = COLORS[selected_color]  # current colour
pinky_angle = None  # smoothed tilt of the left pinky while the colour menu is open
tip_history = deque(maxlen=SMOOTHING_POINTS)
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

    # Left pinky only up: open the colour menu and pick by pinky tilt (the pinky being up also pauses brush size).
    # The pinky counts as up when extended in any direction, so it can point sideways to reach violet.
    pinky_only = bool(left) and math.dist(left[0], left[20]) > math.dist(left[0], left[18]) and not any(up(left, t) for t in (8, 12, 16))
    if pinky_only:
        tilt = math.degrees(math.atan2(left[17][1] - left[20][1], left[20][0] - left[17][0]))  # 90 = parallel to the left edge, 0 = parallel to the bottom edge
        pinky_angle = tilt if pinky_angle is None else pinky_angle + 0.3 * (tilt - pinky_angle)
        selected = int(np.clip((90 - np.clip(pinky_angle, 0, 90)) / 15, 0, 5))  # same angle as the menu: 90 = red, 0 = violet
        selected_color = selected
        ink = COLORS[selected_color]
    else:
        pinky_angle = None

    # Left hand: pinch thumb and index together or apart to preview a brush size.
    # The size commits only after the pinch position stays steady for a moment.
    if left and not pinky_only and not any(up(left, t) for t in (12, 16, 20)):
        span = max(math.dist(left[0], left[9]), 1)  # hand size, so distance to camera matters less
        pinch = math.dist(left[4], left[8]) / span
        target_size = float(np.interp(pinch, (0.15, 1.0), (MIN_BRUSH, MAX_BRUSH)))
        pinch_size = target_size if pinch_size is None else pinch_size + BRUSH_SMOOTHING * (target_size - pinch_size)
        if brush_candidate is None or abs(pinch_size - brush_candidate) > BRUSH_STABLE_TOLERANCE:
            brush_candidate = pinch_size
            brush_candidate_since = time.time()
            brush_candidate_confirmed = False
        elif not brush_candidate_confirmed and time.time() - brush_candidate_since >= BRUSH_CONFIRM_SECS:
            brush_size = brush_candidate
            brush = brush_size * scale
            brush_candidate_confirmed = True
    else:
        brush_candidate = None
        brush_candidate_since = None
        brush_candidate_confirmed = False
        pinch_size = None

    # Right hand: all the gestures.
    index, middle, ring_, pinky = (right and up(right, t) for t in (8, 12, 16, 20))
    fist = bool(right) and not (index or middle or ring_ or pinky)
    both_palms = all(h and all(up(h, t) for t in (8, 12, 16, 20)) for h in (left, right))
    toggle = hold("fist", fist, TOGGLE_SECS)
    if toggle == 1 and not latched:
        wand, latched = not wand, True
        draw_armed = False
    elif toggle == 0:
        latched = False
    if wand and right and index and not (middle or ring_ or pinky) and index_fully_extended(right):
        draw_armed = True
    clear = hold("clear", wand and both_palms, CLEAR_SECS)

    if clear == 1:
        canvas[:] = 0
        strokes.clear()
        current_stroke = None
    eraser = None
    together = bool(right) and math.dist(right[8], right[12]) < TOGETHER * math.dist(right[0], right[9])
    if wand and index and middle and together and not (ring_ or pinky):  # point between the two fingertips
        eraser = ((right[8][0] + right[12][0]) // 2, (right[8][1] + right[12][1]) // 2)
        cv2.circle(canvas, eraser, px(ERASE_RADIUS), 0, cv2.FILLED)
    pt = right[8] if wand and draw_armed and index and not (middle or ring_ or pinky) else None
    cursor = None
    if pt:
        tip_history.append(pt)
        smooth_pt = tuple(np.mean(tip_history, axis=0).astype(int))
        if prev is None:
            current_stroke = []
        if prev:
            thickness = int(brush)
            cv2.line(canvas, prev, smooth_pt, ink, thickness)
            current_stroke.append((prev, smooth_pt, ink, thickness))
        prev = smooth_pt
        cursor = smooth_pt
    else:
        if current_stroke:
            strokes.append(current_stroke)
            current_stroke = None
        tip_history.clear()
        prev = None

    drawn = canvas.any(axis=2)
    frame[drawn] = canvas[drawn]
    for pts in hands.values():
        for a, b in HAND_LINES:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), px(2), cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, px(4), (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
    if left:  # show the thumb-index line and the current brush size between the fingertips
        cv2.line(frame, left[4], left[8], (0, 255, 255), px(2), cv2.LINE_AA)
        mid = ((left[4][0] + left[8][0]) // 2, (left[4][1] + left[8][1]) // 2)
        preview_brush = brush_candidate * scale if brush_candidate is not None else brush
        cv2.circle(frame, mid, max(int(preview_brush / 2), 1), ink, cv2.FILLED)
        if brush_candidate is not None and not brush_candidate_confirmed:
            progress = min((time.time() - brush_candidate_since) / BRUSH_CONFIRM_SECS, 1)
            ring(frame, mid, progress)
    if cursor:
        cv2.circle(frame, cursor, max(int(brush / 2), 1), ink, cv2.FILLED, cv2.LINE_AA)
    if toggle:
        ring(frame, right[9], toggle)
    if 0 < clear < 1:
        for h in (left, right):
            ring(frame, h[9], clear)
    if eraser:
        cv2.circle(frame, eraser, px(ERASE_RADIUS), (255, 255, 255), px(2), cv2.LINE_AA)
    if pinky_angle is not None:
        draw_menu(frame, selected)
    label, label_color = ("WAND ON", (0, 255, 0)) if wand else ("WAND OFF", (0, 0, 255))
    cv2.putText(frame, label, (px(10), px(30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, label_color, px(2), cv2.LINE_AA)
    draw_help(frame)
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        canvas[:] = 0
        strokes.clear()
        current_stroke = None
    if key == ord("u"):
        if current_stroke:
            strokes.append(current_stroke)
            current_stroke = None
        if strokes:
            strokes.pop()
            redraw_canvas(canvas, strokes)
        prev = None
        tip_history.clear()

save_settings(brush_size, selected_color)
cap.release()
cv2.destroyAllWindows()
