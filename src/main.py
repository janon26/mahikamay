"""MahiKamay: air drawing with your hand. Rough draft.

Run:  python src/main.py      (q = quit, c = clear, s = save)
"""
import time

import cv2

from canvas import Canvas
from hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not open the camera.")

    h, w = frame.shape[:2]
    tracker = HandTracker()
    canvas = Canvas(w, h)
    color = (255, 0, 255)  # BGR. TODO: toolbar to change this
    start = last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror so it feels natural

        tracker.process(frame, (time.time() - start) * 1000)
        lm = tracker.landmarks_px(frame)
        fingers = tracker.fingers_up(lm)

        if lm:
            tip = lm[8]  # index fingertip
            if fingers[1:] == [1, 0, 0, 0]:  # index only -> draw
                canvas.draw_to(tip, color)
            elif all(fingers):  # open palm -> clear
                canvas.clear()
            else:  # anything else -> hover
                canvas.lift()
            cv2.circle(frame, tip, 10, color, cv2.FILLED)
        else:
            canvas.lift()

        out = canvas.overlay(frame)
        # tracker.draw_landmarks(out)  # uncomment to debug landmarks

        now = time.time()
        fps = 1 / max(now - last, 1e-6)
        last = now
        cv2.putText(out, f"FPS {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("MahiKamay", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            canvas.clear()
        if key == ord("s"):
            cv2.imwrite(f"drawing_{int(now)}.png", canvas.img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
