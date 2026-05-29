#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Survival Simulation — launcher
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Survival Simulation — GPU Edition"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Python deps ────────────────────────────────────────────
echo "[1/3] Checking Python dependencies..."
pip install numpy pygame moderngl noise numba scipy \
    --break-system-packages --quiet 2>/dev/null || \
pip install numpy pygame moderngl noise numba scipy --quiet

# ── C kernel ───────────────────────────────────────────────
echo "[2/3] Compiling C kernels (fire spread, BFS pathfinding)..."
gcc -O3 -march=native -ffast-math -shared -fPIC \
    -o engine/sim_kernels.so engine/sim_kernels.c -lm \
    && echo "      ✓ C kernels compiled" \
    || echo "      ⚠ C compile failed — Python fallback active"

# ── Run ───────────────────────────────────────────────────
echo "[3/3] Launching simulation..."
echo ""
echo "  Controls:"
echo "    WASD / Arrows  — pan camera"
echo "    Mouse wheel    — zoom in/out"
echo "    + / -          — increase/decrease sim speed"
echo "    Click          — select agent (shows vitals + learning weights)"
echo "    F              — ignite fire at cursor"
echo "    SPACE          — pause"
echo "    R              — reset world"
echo "    ESC            — quit"
echo ""
python3 main.py
