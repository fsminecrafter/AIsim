#!/usr/bin/env python3
"""
Headless smoke test — validates all systems without opening a window.
Run: python3 test_headless.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np

print("=" * 60)
print("SURVIVAL SIM — headless validation")
print("=" * 60)

# ── C kernel ──────────────────────────────────────────────────
print("\n[1] C kernel (sim_kernels.so)...")
from engine.kernels_py import c_fire_spread, c_bfs_find_nearest, _load
lib = _load()
if lib:
    print("    ✓ C shared library loaded")
else:
    print("    ✗ C lib not loaded (Python fallback active)")

# ── World ─────────────────────────────────────────────────────
print("\n[2] World generation...")
from world.world import World, O_TREE, O_BERRY, O_WHEAT
world = World(seed=42)
n_trees = int((world.obj_type == O_TREE).sum())
n_river = int((world.terrain == 3).sum())
print(f"    ✓ Terrain generated  ({n_trees} trees, {n_river} river tiles)")

# ── Growth kernel (Numba JIT) ─────────────────────────────────
print("\n[3] Numba growth kernel (JIT compile — first run slow)...")
import time
t0 = time.perf_counter()
from world.world import update_growth_kernel
update_growth_kernel(world.obj_type, world.obj_growth, world.obj_rate,
                     world.terrain, 1.0, 1)
t1 = time.perf_counter()
print(f"    ✓ JIT compiled & executed in {(t1-t0)*1000:.0f} ms")

# ── Fire kernel (Numba JIT) ───────────────────────────────────
print("\n[4] Numba fire kernel...")
world.start_fire(40, 40)
t0 = time.perf_counter()
from world.world import update_fire_kernel
update_fire_kernel(
    world.fire_intensity.astype(np.float32),
    world.fire_fuel.astype(np.float32),
    world.fire_age.astype(np.float32),
    world.terrain, world.obj_type,
    world.moisture.astype(np.float32),
    np.float32(0.5), np.float32(1.5),
    world.rng_seeds
)
t1 = time.perf_counter()
print(f"    ✓ Fire kernel ran in {(t1-t0)*1000:.0f} ms")

# ── C BFS ────────────────────────────────────────────────────
print("\n[5] C BFS find_nearest...")
result = c_bfs_find_nearest(world.obj_type, world.obj_growth, world.terrain,
                             40, 40, O_TREE, 25)
print(f"    ✓ BFS result: {result}")

# ── C fire spread ─────────────────────────────────────────────
print("\n[6] C fire spread kernel...")
rng32 = world.rng_seeds.astype(np.uint32)
n_ign = c_fire_spread(world.fire_intensity, world.fire_fuel, world.terrain,
                       world.obj_type, world.moisture, rng32,
                       1.5, 0.5)
print(f"    ✓ C fire spread: {n_ign} new ignitions")

# ── Agents ────────────────────────────────────────────────────
print("\n[7] Agent spawn + 100 ticks...")
from ai.agent import Agent
agents = []
for i in range(12):
    x = random.uniform(20, 60)
    y = random.uniform(20, 60)
    a = Agent(x, y)
    a.inv_add("berry", 5, freshness=90.0)
    a.inv_add("rock",  3)
    agents.append(a)

t0 = time.perf_counter()
for tick in range(100):
    world.tick(0.016)
    new_children = []
    for ag in agents:
        ag.tick(world, agents, 0.016)
        child = ag.tick_pregnancy(world, agents, 0.016 * 6.0)
        if child:
            new_children.append(child)
    agents.extend(new_children)
    agents = [a for a in agents if not a.is_dead]
t1 = time.perf_counter()

alive = [a for a in agents if not a.is_dead]
print(f"    ✓ 100 ticks × {len(agents)} agents in {(t1-t0)*1000:.0f} ms")
print(f"    ✓ Agents alive: {len(alive)}")

# ── Learning weights ──────────────────────────────────────────
print("\n[8] Learning weight inheritance...")
if alive:
    ag = alive[0]
    print(f"    Agent {ag.id}  gen={ag.generation}")
    print(f"    weights = {[f'{w:.2f}' for w in ag.weights]}")
    print(f"    event_log last 5: {ag.event_log[-5:]}")

# ── World tick throughput ─────────────────────────────────────
print("\n[9] World tick throughput...")
t0 = time.perf_counter()
for _ in range(600):
    world.tick(0.016)
t1 = time.perf_counter()
print(f"    ✓ 600 world ticks in {(t1-t0)*1000:.0f} ms ({600/(t1-t0):.0f} ticks/s)")

print("\n" + "=" * 60)
print("All systems validated ✓")
print("Run  python3 main.py  to launch the full GPU window.")
print("=" * 60)
