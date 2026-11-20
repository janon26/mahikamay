# mahikamay
Magical hand drawings out of thin air? Real-time computer vision hand tracking? I'm Jon and you're watching MahiKamay!

## Quick start (draft)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python src/main.py               # q = quit, c = clear
```

The wand starts **off**. Hold a **fist** for ~0.7s to toggle it on or off (a ring fills while you hold).

With the wand on:
- **Index finger only** draws
- **Index + middle finger up** erases a small circle between the two fingertips
- **Two open palms** held ~1s clears everything

See [PLAN.md](PLAN.md) for the roadmap.
