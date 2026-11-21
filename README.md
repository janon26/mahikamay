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
- **Index + middle finger up** erases a small circle between the two fingertips
- **Open palm** held ~1.5s clears everything

**Left hand:** pinch thumb and index together or apart to change the brush size. A preview dot between your fingertips shows the current size.

See [PLAN.md](PLAN.md) for the roadmap.
