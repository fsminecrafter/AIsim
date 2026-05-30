#!/usr/bin/env bash
# ============================================================
#  Survival Simulation — 3D GPU Edition
#  Run script for Linux / macOS
# ============================================================

set -e

BOLD=$(tput bold 2>/dev/null || echo "")
RESET=$(tput sgr0 2>/dev/null || echo "")
GREEN=$(tput setaf 2 2>/dev/null || echo "")
YELLOW=$(tput setaf 3 2>/dev/null || echo "")
RED=$(tput setaf 1 2>/dev/null || echo "")

echo ""
echo "${BOLD}================================================${RESET}"
echo "${BOLD}  SURVIVAL SIMULATION — 3D GPU Edition${RESET}"
echo "${BOLD}================================================${RESET}"
echo ""

# Python check
if ! command -v python3 &>/dev/null; then
    echo "${RED}[ERROR]${RESET} python3 not found. Install Python 3.9+"
    exit 1
fi

echo "${GREEN}[1/4]${RESET} Python: $(python3 --version)"

echo "${GREEN}[2/4]${RESET} Updating pip..."
python3 -m pip install --upgrade pip -q

echo "${GREEN}[3/4]${RESET} Installing dependencies..."
python3 -m pip install \
    "pygame==2.5.2" \
    moderngl \
    numpy \
    numba \
    scipy \
    noise \
    -q 2>/dev/null || \
python3 -m pip install "pygame==2.5.2" moderngl numpy numba scipy -q

echo "${GREEN}[4/4]${RESET} Compiling C kernels..."
if command -v gcc &>/dev/null; then
    if gcc -O3 -march=native -ffast-math -shared -fPIC \
           -o sim_kernels.so sim_kernels.c 2>/dev/null; then
        echo "  ${GREEN}[OK]${RESET} C kernels compiled."
    else
        echo "  ${YELLOW}[INFO]${RESET} C compilation failed — Python fallback active."
    fi
else
    echo "  ${YELLOW}[INFO]${RESET} gcc not found — Python fallback active."
fi

echo ""
echo "${BOLD}================================================${RESET}"
echo "  Controls:"
echo "    WASD/Arrows     — Pan camera"
echo "    Q / E           — Orbit left / right"
echo "    Z / X           — Tilt camera up / down"
echo "    Middle mouse    — Drag to orbit"
echo "    Scroll          — Zoom"
echo "    Left click      — Select agent"
echo "    F               — Start fire at cursor"
echo "    +/-             — Sim speed"
echo "    Space           — Pause"
echo "    R               — Reset"
echo "    ESC             — Quit"
echo "${BOLD}================================================${RESET}"
echo ""

python3 main.py
