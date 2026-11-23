# mahikamay
Magical hand drawings out of thin air? Real-time computer vision hand tracking? I'm Jon and you're watching MahiKamay!

## Quick start (draft)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python src/main.py               # q = quit, c = clear
```

The wand starts **off**. Hold a **right-hand fist** for ~0.7s to toggle it on or off (a ring fills while you hold).

With the wand on (right hand):
- **Index finger only** draws
- **Index + middle finger up and pressed together** erases a small circle at the fingertips (a spread peace sign does not erase)
- **Both hands open (two palms)** held ~1s clears everything

**Left hand:** pinch thumb and index together or apart, with the other three fingers folded, to change the brush size. Any other left-hand pose keeps the last size. A dot between your fingertips shows the current size.

**Colours (left hand):** raise **only your pinky** to open a quarter-circle colour menu in the bottom-left corner (red, orange, yellow, green, blue, violet). The menu follows your pinky's angle: parallel to the left edge (straight up) is red, and tilting it right toward the bottom edge goes through orange, yellow, green and blue to violet. Lower the pinky to close the menu; the chosen colour stays. Brush size is paused while the menu is open.

Hover over the **?** in the bottom-left corner of the window to see these instructions.

See [PLAN.md](PLAN.md) for the roadmap.
