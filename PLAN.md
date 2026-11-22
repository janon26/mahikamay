# MahiKamay — Rough Plan (Draft v0)

> Draw in the air with your finger, see it on your webcam feed.

## The idea in one line
Webcam frame -> find hand landmarks (MediaPipe) -> track the index fingertip -> paint onto a
transparent canvas (NumPy) -> blend canvas over the live frame (OpenCV) -> show it.

## Tech stack
| Piece | Why |
|---|---|
| Python 3.11 / 3.12 | MediaPipe wheels lag behind the newest Python |
| OpenCV (`opencv-python`) | camera capture, drawing lines, display window |
| MediaPipe | 21 hand landmarks per hand, real-time on CPU |
| NumPy | canvas array + masking/blending |

## Gestures
The wand starts off. Nothing draws, erases or clears until it is switched on.

| Hand | Gesture | Action |
|---|---|---|
| Right | Fist held ~0.7s | **Toggle** wand on/off |
| Right | Index finger up only (wand on) | **Draw** |
| Right | Index + middle up, fingertips together (wand on) | **Erase** a small circle at the fingertips |
| Both | Two open palms held ~1s (wand on) | **Clear** the whole canvas |
| Left | Pinch thumb + index in/out, other fingers folded | **Brush size**, with a live preview; other poses keep the last size |

## Milestones
1. **Camera** — open webcam, flip (mirror) it, show FPS. Press `q` to quit.
2. **Hands** — run MediaPipe, draw landmarks on the frame.
3. **Fingertip** — get index tip (landmark 8) in pixel coords, draw a dot.
4. **Draw** — keep previous point, `cv2.line(prev, curr)` on a NumPy canvas, overlay it.
5. **Finger-up detection** — compare tip vs. PIP joint y-values -> draw vs. hover mode.
6. **Toolbar** — color boxes + eraser across the top; hover to select.
7. **Polish** — smoothing (moving average / lerp on the tip), brush size, save (`s` -> PNG).

## Stretch ideas
- Two hands: one draws, the other picks color / brush size
- Pinch (thumb tip + index tip distance) to draw instead of finger-up
- Shape recognition (draw a rough circle -> snap to a perfect circle)
- Undo stack (store strokes, not pixels)
- Record to video / GIF

## Known gotchas
- Mirror the frame (`cv2.flip(frame, 1)`) or drawing feels backwards
- MediaPipe wants RGB, OpenCV gives BGR
- MediaPipe 1.x only has the Tasks API (`mp.solutions` is gone); the hand model downloads on first run
- Jittery fingertip -> needs smoothing
- Lighting matters a lot for detection

## Layout
```
src/
  main.py          # capture -> detect -> draw -> show
  hand_tracker.py  # MediaPipe wrapper: returns hand landmarks
```
