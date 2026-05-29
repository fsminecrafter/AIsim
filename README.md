# Survival Simulation — GPU Edition

A living, breathing survival world with autonomous AI agents that **learn,
evolve, and pass knowledge to their children**.

```
80×80 tile world · up to 120 agents · OpenGL GPU rendering · C + Numba JIT kernels
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   main.py (Python)                  │
│    pygame event loop  ·  sim step  ·  HUD blit      │
└────────────┬──────────────────────────┬─────────────┘
             │                          │
┌────────────▼────────────┐  ┌──────────▼──────────────┐
│   world/world.py        │  │   ai/agent.py           │
│                         │  │                         │
│  Terrain (Perlin noise) │  │  Behaviour tree         │
│  Resource growth        │  │  Learning weights (12)  │
│  Fire spread            │  │  Genetic inheritance    │
│  Season / weather       │  │  Pregnancy / lifecycle  │
│  Structure placement    │  │  Event memory           │
└────────────┬────────────┘  └──────────┬──────────────┘
             │                          │
┌────────────▼──────────────────────────▼─────────────┐
│              GPU / accelerated kernels               │
│                                                      │
│  engine/renderer.py  — OpenGL 3.3 instanced draw    │
│    • world tiles: 1 draw call (6400 instances)      │
│    • agents:       1 draw call (N instances)        │
│    • fire glow:    additive blend pass              │
│    • night/season: fullscreen overlay shader        │
│                                                      │
│  engine/sim_kernels.c  (compiled to .so)            │
│    • fire_spread_step()  — C99, -O3 -march=native   │
│    • bfs_find_nearest()  — C99 BFS pathfinding      │
│    • agent_batch_vitals() — batch stat decay        │
│                                                      │
│  world/world.py — Numba @njit(parallel=True)        │
│    • update_growth_kernel()  — parallel growth      │
│    • update_fire_kernel()    — parallel fire sim    │
│    • astar()                 — A* pathfinding       │
└──────────────────────────────────────────────────────┘
```

---

## AI Learning System

Each agent carries a **weight vector of 12 floats** (one per action type):

```
[flee_fire, drink, seek_safety, find_water, find_food,
 seek_warmth, forage, stockpile, craft_tool, build,
 seek_mate, wander]
```

### Online Reinforcement Learning
Every tick, after acting, the agent measures the **vital delta** (change in
hunger + thirst + warmth + mood). If vitals improved, the action's weight
increases. If they fell, it decreases:

```python
reward = (Δhunger*0.4 + Δthirst*0.4 + Δwarmth*0.1 + Δmood*0.1) / 20.0
weight[action] += LEARNING_RATE * reward
```

This means:
- Agents that drink when thirsty learn to drink more urgently
- Agents that forage before winter learn to stockpile earlier
- Agents that flee fires learn the fear response faster

### Genetic Knowledge Transfer (Inheritance)
When a child is born:

1. **Weights** = average of both parents' weights + Gaussian noise
2. **Memory** = 70% of mother's known resource locations
3. **Generation counter** increments (shown as color tint in HUD)

Generations that survive cold winters have higher `seek_warmth` weights
and will pass those to their children — creating emergent culture.

---

## Systems Overview

| System | Implementation | Acceleration |
|--------|---------------|-------------|
| World rendering | OpenGL 3.3, instanced quads | GPU shaders |
| Fire glow | OpenGL additive blend pass | GPU |
| Night/season tint | Fullscreen GLSL overlay | GPU |
| Fire spread | C99 (`sim_kernels.c`) | CPU, -O3 native |
| BFS pathfinding | C99 | CPU, -O3 native |
| Resource growth | Numba `@njit(parallel=True)` | CPU SIMD |
| A* pathfinding | Numba `@njit` | CPU JIT |
| Vitals decay | Python / C batch kernel | Mixed |
| Agent AI | Python behaviour tree | CPU |

---

## Controls

| Key | Action |
|-----|--------|
| `WASD` / `Arrow keys` | Pan camera |
| Mouse wheel | Zoom |
| `+` / `-` | Speed up / slow down simulation |
| Left click | Select agent (shows vitals, inventory, learning weights) |
| `F` | Ignite fire at cursor |
| `SPACE` | Pause/resume |
| `R` | Reset world |
| `ESC` | Quit |

---

## Vital Decay Rates

| Stat | Base | Rain | Winter | Baby |
|------|------|------|--------|------|
| Hunger | −1/min | ×1.5 | ×1.3 | ×0.7 |
| Thirst | −2/min | ×1.0 | ×1.0 | ×1.0 |
| Warmth | −0.5/min | ×2.0 | ×3.0 | ×1.5 |
| Mood | −0.3/min | ×1.8 | ×0.5 | — |

Near a **fireplace** (3 tiles): warmth +3/min
Inside a **house**: rain penalty removed, cold halved

---

## Building

```bash
# Install Python deps
pip install numpy pygame moderngl noise numba scipy --break-system-packages

# Compile C kernels (optional but recommended for perf)
gcc -O3 -march=native -ffast-math -shared -fPIC \
    -o engine/sim_kernels.so engine/sim_kernels.c -lm

# Validate all systems headlessly
python3 test_headless.py

# Launch
python3 main.py
# or
bash run.sh
```

Requires Python 3.10+, GCC, OpenGL 3.3+ capable display.

---

## File Map

```
survival_sim/
├── main.py                   # Game loop, input, window
├── run.sh                    # Launcher script
├── test_headless.py          # Smoke test (no window)
├── engine/
│   ├── config.py             # All tunable constants
│   ├── renderer.py           # ModernGL OpenGL renderer
│   ├── hud.py                # pygame HUD overlay
│   ├── sim_kernels.c         # C99 hot-path kernels
│   ├── sim_kernels.so        # Compiled shared library
│   └── kernels_py.py         # ctypes bindings + fallback
├── world/
│   └── world.py              # Terrain, fire, growth, Numba JIT
└── ai/
    └── agent.py              # Agent AI, learning, genetics
```
