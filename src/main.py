import math
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from hand_tracker import HAND_LINES, HandTracker

WINDOW = "🪄 MahiKamay"
CAMERA_SIZE = (1280, 720)  # the most this camera delivers
COLORS = [  # BGR: red, orange, yellow, green, blue, violet
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 200, 0), (255, 0, 0), (211, 0, 148),
]
# Sizes below are in pixels for a 640-wide frame; px() scales them to the real frame width.
ERASE_RADIUS = 25
MIN_ERASER, MAX_ERASER = 8, 80
TOGETHER = 0.3  # erase only if index and middle fingertips are closer than this fraction of palm length
MIN_BRUSH, MAX_BRUSH = 2, 40
TOGGLE_SECS, CLEAR_SECS = 0.7, 1.0
SMOOTHING_POINTS = 5
BRUSH_CONFIRM_SECS = 0.5
BRUSH_SMOOTHING = 0.2
BRUSH_STABLE_TOLERANCE = 3.0
SHAPE_HOLD_SECS = 1.0
SHAPE_MIN_POINTS = 12
SHAPE_MIN_SIZE = 30
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"

since = {}  # when each gesture started being held


def up(hand, tip):
    return hand[tip][1] < hand[tip - 2][1]  # fingertip above its middle joint


def finger_fully_extended(hand, tip, mcp, extension_ratio=2.2):
    """Require a finger tip to be well beyond its proximal joint."""
    return up(hand, tip) and math.dist(hand[mcp], hand[tip]) > extension_ratio * max(math.dist(hand[mcp], hand[tip - 2]), 1)


def index_fully_extended(hand):
    """Require both index joints to be straight before arming drawing."""
    return finger_fully_extended(hand, 8, 5)


def eraser_pose(hand):
    """Whether the hand is making the close index-and-middle eraser gesture."""
    if not hand or not up(hand, 8) or not up(hand, 12) or up(hand, 16) or up(hand, 20):
        return False
    return math.dist(hand[8], hand[12]) < TOGETHER * math.dist(hand[0], hand[9])


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


def redraw_canvas(canvas, actions):
    """Rebuild the canvas from ordered drawing and erasing actions."""
    canvas[:] = 0
    for action in actions:
        if action[0] == "stroke":
            for start, end, color, thickness in action[1]:
                cv2.line(canvas, start, end, color, thickness)
        else:
            cv2.circle(canvas, action[1], action[2], 0, cv2.FILLED)


def stroke_segments(points, color, thickness):
    """Convert a closed point path into the segment format used by undo."""
    return [
        (tuple(start), tuple(end), color, thickness)
        for start, end in zip(points, points[1:])
    ]


def shape_points(points):
    """Return clean points for a confident triangle, rectangle, or ellipse."""
    if len(points) < SHAPE_MIN_POINTS:
        return None
    contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    area = abs(cv2.contourArea(contour))
    if perimeter < SHAPE_MIN_SIZE or area < SHAPE_MIN_SIZE ** 2:
        return None

    approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True).reshape(-1, 2)
    if len(approximation) == 3 and cv2.isContourConvex(approximation.reshape(-1, 1, 2)):
        return np.vstack((approximation, approximation[0]))

    # Rough rectangles often simplify to 5 or 6 corners. Validate their
    # minimum-area box before ellipse fitting, so they are not mistaken for circles.
    if 4 <= len(approximation) <= 6 and cv2.isContourConvex(approximation.reshape(-1, 1, 2)):
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        box_area = width * height
        box_perimeter = 2 * (width + height)
        if box_area and box_perimeter and area / box_area >= 0.55 and perimeter / box_perimeter >= 0.84:
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
            return np.vstack((box, box[0]))

    if len(points) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (major, minor), angle = ellipse
        ellipse_area = math.pi * major * minor / 4
        if ellipse_area and area / ellipse_area >= 0.65:
            radians = math.radians(angle)
            cos_angle, sin_angle = math.cos(radians), math.sin(radians)
            result = []
            for step in range(65):
                theta = 2 * math.pi * step / 64
                x = (major / 2) * math.cos(theta)
                y = (minor / 2) * math.sin(theta)
                result.append((round(cx + x * cos_angle - y * sin_angle), round(cy + x * sin_angle + y * cos_angle)))
            return np.asarray(result, dtype=np.int32)
    return None


def save_drawing(canvas):
    """Save the ink as a transparent PNG, leaving the empty canvas transparent."""
    drawings_dir = SETTINGS_FILE.parent / "drawings"
    drawings_dir.mkdir(exist_ok=True)
    filename = drawings_dir / f"drawing_{time.strftime('%Y%m%d_%H%M%S')}.png"
    alpha = np.where(canvas.any(axis=2), 255, 0).astype(np.uint8)
    image = np.dstack((canvas, alpha))
    if not cv2.imwrite(str(filename), image):
        raise OSError("OpenCV could not write the drawing")
    return filename


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
    "SAVE: s = transparent PNG",
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
current_points = []
shape_hold_since = None
shape_locked = False
shape_anchor = None
save_message = ""
save_message_until = 0.0
wand = False
last_mode = "off"
latched = False  # stops one fist hold from toggling repeatedly
draw_armed = False
brush_size = settings["brush_size"]  # stored in 640-wide logical pixels
brush = brush_size * scale
brush_candidate = None
brush_candidate_since = None
brush_candidate_confirmed = False
pinch_size = None
eraser_size = float(ERASE_RADIUS)
pinch_kind = None
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
    pinch_pose = bool(left) and math.dist(left[4], left[8]) < 0.35 * max(math.dist(left[0], left[9]), 1)
    pinky_only = bool(left) and not pinch_pose and finger_fully_extended(left, 20, 17, 2.0) and not any(up(left, t) for t in (8, 12, 16))
    if pinky_only:
        tilt = math.degrees(math.atan2(left[17][1] - left[20][1], left[20][0] - left[17][0]))  # 90 = parallel to the left edge, 0 = parallel to the bottom edge
        pinky_angle = tilt if pinky_angle is None else pinky_angle + 0.3 * (tilt - pinky_angle)
        selected = int(np.clip((90 - np.clip(pinky_angle, 0, 90)) / 15, 0, 5))  # same angle as the menu: 90 = red, 0 = violet
        selected_color = selected
        ink = COLORS[selected_color]
    else:
        pinky_angle = None

    eraser_active = wand and eraser_pose(right)

    # Left hand: pinch thumb and index together or apart to preview a brush or eraser size.
    # The size commits only after the pinch position stays steady for a moment.
    if left and not pinky_only and not any(up(left, t) for t in (12, 16, 20)):
        span = max(math.dist(left[0], left[9]), 1)  # hand size, so distance to camera matters less
        pinch = math.dist(left[4], left[8]) / span
        active_kind = "eraser" if eraser_active else "brush"
        size_min, size_max = (MIN_ERASER, MAX_ERASER) if eraser_active else (MIN_BRUSH, MAX_BRUSH)
        target_size = float(np.interp(pinch, (0.15, 1.0), (size_min, size_max)))
        pinch_size = target_size if pinch_size is None else pinch_size + BRUSH_SMOOTHING * (target_size - pinch_size)
        if active_kind != pinch_kind or brush_candidate is None or abs(pinch_size - brush_candidate) > BRUSH_STABLE_TOLERANCE:
            pinch_kind = active_kind
            brush_candidate = pinch_size
            brush_candidate_since = time.time()
            brush_candidate_confirmed = False
        elif not brush_candidate_confirmed and time.time() - brush_candidate_since >= BRUSH_CONFIRM_SECS:
            if eraser_active:
                eraser_size = brush_candidate
            else:
                brush_size = brush_candidate
                brush = brush_size * scale
            brush_candidate_confirmed = True
    else:
        brush_candidate = None
        brush_candidate_since = None
        brush_candidate_confirmed = False
        pinch_size = None
        pinch_kind = None

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
    mode = "eraser" if wand and eraser_pose(right) else "draw" if wand else "off"
    if mode != last_mode:
        if current_stroke:
            strokes.append(("stroke", current_stroke))
        current_stroke = None
        current_points = []
        shape_hold_since = None
        shape_locked = False
        shape_anchor = None
        prev = None
        tip_history.clear()
        last_mode = mode
    eraser_active = mode == "eraser"
    effective_eraser_size = (
        brush_candidate
        if eraser_active and pinch_kind == "eraser" and brush_candidate is not None
        else eraser_size
    )
    clear = hold("clear", wand and both_palms, CLEAR_SECS)

    if clear == 1:
        canvas[:] = 0
        strokes.clear()
        current_stroke = None
        current_points = []
        shape_hold_since = None
        shape_locked = False
        shape_anchor = None
    eraser = None
    if wand and eraser_active:  # point between the two fingertips
        eraser = ((right[8][0] + right[12][0]) // 2, (right[8][1] + right[12][1]) // 2)
        eraser_radius = px(effective_eraser_size)
        cv2.circle(canvas, eraser, eraser_radius, 0, cv2.FILLED)
        strokes.append(("erase", eraser, eraser_radius))
    pt = right[8] if wand and draw_armed and index and not (middle or ring_ or pinky) else None
    cursor = None
    shape_waiting = False
    if shape_locked and pt:
        if shape_anchor and math.dist(pt, shape_anchor) > max(px(24), px(SHAPE_MIN_SIZE)):
            shape_locked = False
            shape_anchor = None
            current_points = []
            current_stroke = None
            prev = None
        else:
            pt = None
            shape_waiting = True
    if pt:
        tip_history.append(pt)
        smooth_pt = tuple(np.mean(tip_history, axis=0).astype(int))
        if prev is None:
            current_stroke = []
            current_points = [smooth_pt]
            shape_hold_since = None
        else:
            current_points.append(smooth_pt)
        if prev:
            thickness = int(brush)
            cv2.line(canvas, prev, smooth_pt, ink, thickness)
            current_stroke.append((prev, smooth_pt, ink, thickness))
        prev = smooth_pt
        cursor = smooth_pt
        if len(current_points) >= SHAPE_MIN_POINTS:
            start_point = current_points[0]
            close_radius = max(px(24), px(SHAPE_MIN_SIZE))
            if math.dist(smooth_pt, start_point) <= close_radius:
                shape_hold_since = shape_hold_since or time.time()
                if time.time() - shape_hold_since >= SHAPE_HOLD_SECS:
                    clean_points = shape_points(current_points)
                    if clean_points is not None:
                        color = current_stroke[0][2] if current_stroke else ink
                        thickness = current_stroke[0][3] if current_stroke else int(brush)
                        current_stroke = stroke_segments(clean_points, color, thickness)
                        strokes.append(("stroke", current_stroke))
                        current_stroke = None
                        redraw_canvas(canvas, strokes)
                        save_message = "Shape polished!"
                        save_message_until = time.time() + 1.5
                        shape_locked = True
                        shape_anchor = start_point
                        shape_hold_since = None
                        current_points = []
                        prev = None
            else:
                shape_hold_since = None
    else:
        if current_stroke:
            strokes.append(("stroke", current_stroke))
            current_stroke = None
        current_points = []
        shape_hold_since = None
        if not shape_waiting:
            shape_locked = False
            shape_anchor = None
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
        preview_brush = effective_eraser_size * scale if eraser_active else (brush_candidate * scale if brush_candidate is not None else brush)
        preview_color = (255, 255, 255) if eraser_active else ink
        cv2.circle(frame, mid, max(int(preview_brush / 2), 1), preview_color, cv2.FILLED)
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
        cv2.circle(frame, eraser, px(effective_eraser_size), (255, 255, 255), px(2), cv2.LINE_AA)
    if pinky_angle is not None:
        draw_menu(frame, selected)
    label, label_color = ("WAND ON", (0, 255, 0)) if wand else ("WAND OFF", (0, 0, 255))
    cv2.putText(frame, label, (px(10), px(26)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, label_color, px(1), cv2.LINE_AA)
    if time.time() < save_message_until:
        cv2.putText(frame, save_message, (px(10), px(58)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 0), px(2), cv2.LINE_AA)
    draw_help(frame)
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        canvas[:] = 0
        strokes.clear()
        current_stroke = None
        current_points = []
        shape_hold_since = None
        shape_locked = False
        shape_anchor = None
        current_points = []
        shape_hold_since = None
        shape_locked = False
        shape_anchor = None
    if key == ord("s"):
        try:
            save_drawing(canvas)
            save_message = "Saved!"
        except OSError:
            save_message = "Save failed"
        save_message_until = time.time() + 1.5
    if key == ord("u"):
        if current_stroke:
            strokes.append(("stroke", current_stroke))
            current_stroke = None
        if strokes:
            strokes.pop()
            redraw_canvas(canvas, strokes)
        prev = None
        tip_history.clear()

save_settings(brush_size, selected_color)
cap.release()
cv2.destroyAllWindows()
