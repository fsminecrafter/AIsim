"""
World engine — terrain, resources, fire, day/night/season  (v4)

Changes:
  * Smooth heightmap terrain: perlin noise elevation drives a CONTINUOUS
    height value per tile (not just discrete terrain types). The renderer
    reads world.height_map to produce smooth rolling hills with no
    side-gaps.
  * Map size increased to 128×128 (see config.py).
  * S_FARM structure added (placed by agents with hoe + wood).
    Farms auto-generate wheat tiles in a 3×3 area around them.
  * near_river / find_nearest_river use larger radii for the bigger map.
"""

import numpy as np
import math, random
from numba import njit, prange
from engine.config import *

# ─── Terrain constants ───────────────────────────────────────
T_GRASS  = 0
T_DIRT   = 1
T_ROCK   = 2
T_RIVER  = 3

# ─── Object type constants ───────────────────────────────────
O_NONE      = 0
O_TREE      = 1
O_BERRY     = 2
O_WHEAT     = 3
O_ROCK_OBJ  = 4

# ─── Structure constants ─────────────────────────────────────
S_NONE       = 0
S_HOUSE      = 1
S_FIREPLACE  = 2
S_FARM       = 3   # NEW


# ═════════════════════════════════════════════════════════════
#  Numba-JIT kernels
# ═════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def update_growth_kernel(obj_type, obj_growth, obj_rate, terrain, dt, season_idx):
    H, W = obj_type.shape
    SEASON_MULT = np.array([1.2, 2.0, 0.8, 0.0], dtype=np.float32)
    smult = SEASON_MULT[season_idx]
    for y in prange(H):
        for x in prange(W):
            ot = obj_type[y, x]
            if ot == O_NONE:
                continue
            if season_idx == 3 and ot != O_TREE:
                continue
            mult = smult if ot != O_TREE else (smult * 0.5 if smult > 0 else 0.0)
            obj_growth[y, x] = min(1.0, obj_growth[y, x] + obj_rate[y, x] * mult * dt)


@njit(parallel=True, cache=True)
def update_fire_kernel(fire_intensity, fire_fuel, fire_age,
                       terrain, obj_type, moisture,
                       dt, weather_fire_mult, rng_seeds):
    H, W = fire_intensity.shape
    new_ignitions = np.zeros((H, W), dtype=np.float32)
    SPREAD_BASE   = 0.15
    SPREAD_RADIUS = 2
    for y in prange(H):
        for x in prange(W):
            if fire_intensity[y, x] <= 0.0:
                continue
            fire_fuel[y, x] -= dt * 3.0
            fire_age[y, x]  += dt
            if fire_fuel[y, x] <= 0.0:
                fire_intensity[y, x] = 0.0
                fire_fuel[y, x] = 0.0
                continue
            if int(fire_age[y, x]) % 2 == 0:
                seed = rng_seeds[y, x]
                for dy in range(-SPREAD_RADIUS, SPREAD_RADIUS + 1):
                    for dx in range(-SPREAD_RADIUS, SPREAD_RADIUS + 1):
                        ny = y + dy
                        nx = x + dx
                        if ny < 0 or ny >= H or nx < 0 or nx >= W:
                            continue
                        if terrain[ny, nx] == T_RIVER:
                            continue
                        if fire_intensity[ny, nx] > 0:
                            continue
                        ot = obj_type[ny, nx]
                        if ot not in (O_TREE, O_WHEAT):
                            continue
                        moist  = moisture[ny, nx]
                        chance = SPREAD_BASE * (1.0 - moist) * weather_fire_mult
                        seed   = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
                        rv     = (seed >> 16) / 65535.0
                        if rv < chance * dt:
                            new_ignitions[ny, nx] = 1.0
                rng_seeds[y, x] = seed
    for y in prange(H):
        for x in prange(W):
            if new_ignitions[y, x] > 0:
                fire_intensity[y, x] = 30.0
                fire_fuel[y, x]      = 40.0
    return new_ignitions


@njit(cache=True)
def astar(terrain_passable, fire_map, start_y, start_x, goal_y, goal_x):
    H = terrain_passable.shape[0]
    W = terrain_passable.shape[1]
    if start_y == goal_y and start_x == goal_x:
        return [(start_y, start_x)]
    INF = 1e9
    g_cost = np.full((H, W), INF, dtype=np.float64)
    f_cost = np.full((H, W), INF, dtype=np.float64)
    came_from_y = np.full((H, W), -1, dtype=np.int32)
    came_from_x = np.full((H, W), -1, dtype=np.int32)
    in_open = np.zeros((H, W), dtype=np.bool_)
    closed  = np.zeros((H, W), dtype=np.bool_)
    g_cost[start_y, start_x] = 0.0
    h = abs(goal_y-start_y) + abs(goal_x-start_x)
    f_cost[start_y, start_x] = h
    in_open[start_y, start_x] = True
    MAX_ITER = 1200
    for _ in range(MAX_ITER):
        best_f = INF; cy, cx = -1, -1
        for yy in range(H):
            for xx in range(W):
                if in_open[yy, xx] and f_cost[yy, xx] < best_f:
                    best_f = f_cost[yy, xx]; cy, cx = yy, xx
        if cy < 0:
            break
        if cy == goal_y and cx == goal_x:
            path = []
            py, px = cy, cx
            while py >= 0:
                path.append((py, px))
                ny = came_from_y[py, px]
                nx = came_from_x[py, px]
                py, px = ny, nx
            return path[::-1]
        in_open[cy, cx] = False
        closed[cy, cx]  = True
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny = cy+dy; nx = cx+dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W: continue
            if closed[ny, nx]: continue
            if not terrain_passable[ny, nx]: continue
            move_cost = 1.5 if fire_map[ny, nx] > 0 else 1.0
            ng = g_cost[cy, cx] + move_cost
            if ng < g_cost[ny, nx]:
                g_cost[ny, nx] = ng
                h2 = abs(goal_y-ny) + abs(goal_x-nx)
                f_cost[ny, nx] = ng + h2
                came_from_y[ny, nx] = cy
                came_from_x[ny, nx] = cx
                in_open[ny, nx] = True
    return [(-1, -1)]


# ═════════════════════════════════════════════════════════════
#  Smooth heightmap generation helper
# ═════════════════════════════════════════════════════════════

def _generate_smooth_heightmap(rng, H, W, seed_val=42):
    """
    Generate a smooth continuous elevation map using layered Perlin noise.
    Returns float32 array H×W with values roughly in [-1, 1].
    Falls back to smooth sinusoidal noise if the 'noise' package is absent.
    """
    try:
        import noise as _noise
        base = int(rng.integers(0, 255))
        elev = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                # Multiple octaves for natural-looking terrain
                e  = _noise.pnoise2(x/32.0, y/32.0, octaves=6, persistence=0.55,
                                    lacunarity=2.0, base=base)
                elev[y, x] = e
    except Exception:
        # Fallback: sum of sinusoids at different frequencies/phases
        elev = np.zeros((H, W), dtype=np.float32)
        r = np.random.default_rng(seed_val)
        for _ in range(8):
            fx = r.uniform(0.03, 0.15)
            fy = r.uniform(0.03, 0.15)
            px = r.uniform(0, math.tau)
            py = r.uniform(0, math.tau)
            amp = r.uniform(0.1, 0.4)
            for y in range(H):
                for x in range(W):
                    elev[y, x] += amp * math.sin(x * fx + px) * math.sin(y * fy + py)
        # Normalize to [-1, 1]
        lo, hi = elev.min(), elev.max()
        if hi > lo:
            elev = (elev - lo) / (hi - lo) * 2.0 - 1.0

    return elev.astype(np.float32)


# ═════════════════════════════════════════════════════════════
#  World class
# ═════════════════════════════════════════════════════════════

class World:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        W, H = WORLD_W, WORLD_H

        # ── Smooth elevation ──────────────────────────────────
        # Continuous height map used by the renderer for smooth hills.
        # Values: 0.0 = sea level, positive = hills/mountains.
        raw_elev = _generate_smooth_heightmap(rng, H, W, seed_val=seed)

        # Normalize to [0, 1]
        lo, hi = raw_elev.min(), raw_elev.max()
        norm_elev = (raw_elev - lo) / max(hi - lo, 1e-6)

        # Smooth with a box blur to reduce sharp spikes
        from scipy.ndimage import gaussian_filter as _gf
        try:
            norm_elev = _gf(norm_elev, sigma=1.5).astype(np.float32)
        except Exception:
            pass

        # Store as public height_map: float32, shape (H, W)
        # Range: 0 (flat) to ~3.5 world-units of height at peaks
        self.height_map = (norm_elev * 3.5).astype(np.float32)

        # ── Terrain (discrete) ────────────────────────────────
        self.terrain = np.zeros((H, W), dtype=np.int32)
        self._generate_terrain(rng, norm_elev)

        # ── Objects ───────────────────────────────────────────
        self.obj_type        = np.zeros((H, W), dtype=np.int32)
        self.obj_growth      = np.zeros((H, W), dtype=np.float32)
        self.obj_rate        = np.zeros((H, W), dtype=np.float32)
        self.obj_harvest_cd  = np.zeros((H, W), dtype=np.float32)
        self._populate_objects(rng)

        # ── Moisture ──────────────────────────────────────────
        self.moisture = np.zeros((H, W), dtype=np.float32)
        self._calc_moisture()

        # ── Fire ──────────────────────────────────────────────
        self.fire_intensity  = np.zeros((H, W), dtype=np.float32)
        self.fire_fuel       = np.zeros((H, W), dtype=np.float32)
        self.fire_age        = np.zeros((H, W), dtype=np.float32)
        self.rng_seeds       = rng.integers(0, 2**31, (H, W), dtype=np.int64).astype(np.int64)

        # ── Structures ────────────────────────────────────────
        self.structures = np.zeros((H, W), dtype=np.int32)

        # ── Time ──────────────────────────────────────────────
        self.time_of_day  = 0.0
        self.day_number   = 0
        self.season_idx   = 0
        self.total_time   = 0.0

        # ── Weather ───────────────────────────────────────────
        self.weather = "clear"
        self._weather_timer = 0.0
        self._next_weather_change = self._rand_weather_interval()

        # ── Dropped items ─────────────────────────────────────
        self.dropped_items = {}

        self._update_passable()

    # ── Terrain generation ───────────────────────────────────

    def _generate_terrain(self, rng, norm_elev):
        H, W = WORLD_H, WORLD_W
        for y in range(H):
            for x in range(W):
                e = norm_elev[y, x]   # 0..1
                if e > 0.72:
                    self.terrain[y, x] = T_ROCK
                elif e > 0.25:
                    self.terrain[y, x] = T_GRASS
                else:
                    self.terrain[y, x] = T_DIRT

        # Carve rivers
        num_rivers = rng.integers(3, 6)
        for _ in range(num_rivers):
            self._carve_river(rng)

    def _carve_river(self, rng):
        H, W = WORLD_H, WORLD_W
        horizontal = rng.random() > 0.5
        if horizontal:
            y = int(rng.integers(H // 5, 4 * H // 5))
            for x in range(W):
                dy = int(rng.integers(-1, 2))
                y  = max(1, min(H-2, y+dy))
                for wy in range(max(0, y-1), min(H, y+2)):
                    self.terrain[wy, x] = T_RIVER
                    self.height_map[wy, x] = 0.0   # river is flat
        else:
            x = int(rng.integers(W // 5, 4 * W // 5))
            for y in range(H):
                dx = int(rng.integers(-1, 2))
                x  = max(1, min(W-2, x+dx))
                for wx in range(max(0, x-1), min(W, x+2)):
                    self.terrain[y, wx] = T_RIVER
                    self.height_map[y, wx] = 0.0

    def _populate_objects(self, rng):
        H, W = WORLD_H, WORLD_W
        for y in range(H):
            for x in range(W):
                t = self.terrain[y, x]
                if t in (T_RIVER, T_ROCK):
                    continue
                rv = rng.random()
                if t == T_GRASS:
                    if rv < 0.17:
                        self.obj_type[y, x]   = O_TREE
                        self.obj_growth[y, x] = rng.random()
                        self.obj_rate[y, x]   = 1.0 / 60.0
                    elif rv < 0.27:
                        self.obj_type[y, x]   = O_BERRY
                        self.obj_growth[y, x] = rng.random()
                        self.obj_rate[y, x]   = 1.0 / 30.0
                    elif rv < 0.31:
                        self.obj_type[y, x]   = O_ROCK_OBJ
                        self.obj_growth[y, x] = 1.0
                        self.obj_rate[y, x]   = 0.0
                elif t == T_DIRT:
                    if rv < 0.13:
                        self.obj_type[y, x]   = O_WHEAT
                        self.obj_growth[y, x] = rng.random()
                        self.obj_rate[y, x]   = 1.0 / 20.0
                    elif rv < 0.17:
                        self.obj_type[y, x]   = O_ROCK_OBJ
                        self.obj_growth[y, x] = 1.0
                        self.obj_rate[y, x]   = 0.0

    def _calc_moisture(self):
        river_mask = (self.terrain == T_RIVER).astype(np.float32)
        try:
            from scipy.ndimage import gaussian_filter as gf
            self.moisture = gf(river_mask, sigma=5.0).astype(np.float32)
            self.moisture = np.clip(self.moisture * 6.0, 0.0, 1.0)
        except ImportError:
            self.moisture = river_mask * 0.8

    def _update_passable(self):
        self.passable = (self.terrain != T_ROCK).astype(np.bool_)

    # ── Weather ──────────────────────────────────────────────

    def _rand_weather_interval(self):
        return random.uniform(DAY_DURATION * 3, DAY_DURATION * 8)

    WEATHER_CYCLE = {
        "spring": ["clear", "cloudy", "rain", "clear"],
        "summer": ["clear", "clear", "cloudy", "rain"],
        "autumn": ["cloudy", "rain", "storm", "clear"],
        "winter": ["clear", "cloudy", "blizzard", "blizzard"],
    }
    WEATHER_FIRE_MULT   = {"clear": 1.5, "cloudy": 1.0, "rain": 0.1, "storm": 0.0, "blizzard": 0.0}
    WEATHER_MOOD_MULT   = {"clear": 1.0, "cloudy": 1.1, "rain": 1.8, "storm": 2.5, "blizzard": 3.0}
    WEATHER_HUNGER_MULT = {"clear": 1.0, "cloudy": 1.0, "rain": 1.5, "storm": 2.0, "blizzard": 2.0}
    WEATHER_WARMTH_MULT = {"clear": 1.0, "cloudy": 1.1, "rain": 2.0, "storm": 3.0, "blizzard": 5.0}
    SEASON_COLD_MULT    = {"spring": 0.5, "summer": 0.2, "autumn": 0.8, "winter": 3.0}

    @property
    def season_name(self):
        return SEASONS[self.season_idx]

    @property
    def is_day(self):
        return self.time_of_day < 0.5

    # ── Tick ─────────────────────────────────────────────────

    def tick(self, real_dt):
        sim_dt = real_dt * SIM_SPEED
        self.total_time  += sim_dt
        self.time_of_day  = (self.time_of_day + sim_dt / DAY_DURATION) % 1.0
        if int(self.total_time / DAY_DURATION) > self.day_number:
            self.day_number += 1
            if self.day_number % SEASON_DAYS == 0:
                self.season_idx = (self.season_idx + 1) % 4

        self._weather_timer += sim_dt
        if self._weather_timer >= self._next_weather_change:
            pool = self.WEATHER_CYCLE[self.season_name]
            self.weather = random.choice(pool)
            self._weather_timer = 0.0
            self._next_weather_change = self._rand_weather_interval()
            if self.weather in ("rain", "storm", "blizzard"):
                self.moisture = np.clip(self.moisture + 0.3, 0.0, 1.0)
            else:
                self.moisture = np.clip(self.moisture - 0.05, 0.0, 1.0)

        update_growth_kernel(self.obj_type, self.obj_growth, self.obj_rate,
                             self.terrain, sim_dt, self.season_idx)

        mask = self.obj_harvest_cd > 0
        self.obj_harvest_cd[mask] -= sim_dt

        # Farm produces wheat in surrounding tiles
        self._tick_farms(sim_dt)

        if self.season_idx != 3:
            fire_mult = np.float32(self.WEATHER_FIRE_MULT.get(self.weather, 1.0))
            update_fire_kernel(
                self.fire_intensity.astype(np.float32),
                self.fire_fuel.astype(np.float32),
                self.fire_age.astype(np.float32),
                self.terrain, self.obj_type,
                self.moisture.astype(np.float32),
                np.float32(sim_dt), fire_mult,
                self.rng_seeds,
            )
            burned = (self.fire_intensity > 0) & (self.obj_type == O_TREE)
            self.obj_type[burned]   = O_NONE
            self.obj_growth[burned] = 0.0

        if self.season_idx == 1 and self.weather == "clear":
            if random.random() < 0.0001 * sim_dt:
                trees = np.argwhere(self.obj_type == O_TREE)
                if len(trees) > 0:
                    ty, tx = trees[random.randint(0, len(trees)-1)]
                    self.start_fire(ty, tx)

    def _tick_farms(self, sim_dt):
        """
        Farms generate wheat in a 3×3 area around them.
        Wheat grows fast on farm tiles (season permitting).
        """
        farm_tiles = np.argwhere(self.structures == S_FARM)
        if len(farm_tiles) == 0:
            return
        H, W = WORLD_H, WORLD_W
        for (fy, fx) in farm_tiles:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = fy+dy, fx+dx
                    if not (0 <= ny < H and 0 <= nx < W):
                        continue
                    if self.structures[ny, nx] != S_NONE:
                        continue
                    if self.terrain[ny, nx] not in (T_GRASS, T_DIRT):
                        continue
                    if self.obj_type[ny, nx] == O_NONE:
                        # Seed new wheat
                        self.obj_type[ny, nx]   = O_WHEAT
                        self.obj_growth[ny, nx] = 0.0
                        self.obj_rate[ny, nx]   = 1.0 / 8.0   # fast on farm
                    elif self.obj_type[ny, nx] == O_WHEAT:
                        # Boost growth rate on farm tiles
                        self.obj_rate[ny, nx] = 1.0 / 8.0

    def start_fire(self, y, x):
        if self.terrain[y, x] != T_RIVER:
            self.fire_intensity[y, x] = 50.0
            self.fire_fuel[y, x]      = 60.0
            self.fire_age[y, x]       = 0.0

    def extinguish_fire(self, y, x):
        self.fire_intensity[y, x] = 0.0
        self.fire_fuel[y, x]      = 0.0

    def place_structure(self, y, x, stype):
        # Structure REPLACES whatever tile was there
        self.structures[y, x] = stype
        # Clear any object on this tile
        self.obj_type[y, x]   = O_NONE
        self.obj_growth[y, x] = 0.0
        self._update_passable()

    def remove_object(self, y, x):
        self.obj_type[y, x]   = O_NONE
        self.obj_growth[y, x] = 0.0

    def harvest(self, y, x):
        ot = self.obj_type[y, x]
        if ot == O_NONE:   return None
        g  = self.obj_growth[y, x]
        if g < 1.0:        return None
        cd = self.obj_harvest_cd[y, x]
        if cd > 0:         return None

        if ot == O_BERRY:
            self.obj_growth[y, x]    = 0.0
            self.obj_harvest_cd[y, x] = 20.0
            return {"type": "berry", "quantity": 3, "freshness": 100.0}
        elif ot == O_TREE:
            self.obj_growth[y, x]    = 0.4
            self.obj_harvest_cd[y, x] = 120.0
            return {"type": "wood", "quantity": 2, "freshness": None}
        elif ot == O_WHEAT:
            self.obj_growth[y, x] = 0.0
            return {"type": "wheat", "quantity": 2, "freshness": 100.0}
        elif ot == O_ROCK_OBJ:
            return {"type": "rock", "quantity": 1, "freshness": None}
        return None

    def drop_items(self, y, x, items):
        key = (int(y), int(x))
        if key not in self.dropped_items:
            self.dropped_items[key] = []
        self.dropped_items[key].extend(items)

    def pick_dropped(self, y, x):
        key = (int(y), int(x))
        return self.dropped_items.pop(key, [])

    # ── Spatial queries ───────────────────────────────────────

    def near_river(self, y, x, radius=1):
        H, W = WORLD_H, WORLD_W
        iy, ix = int(y), int(x)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = iy+dy, ix+dx
                if 0 <= ny < H and 0 <= nx < W:
                    if self.terrain[ny, nx] == T_RIVER:
                        return True
        return False

    def find_nearest_water_tile(self, y, x, radius=12):
        H, W = WORLD_H, WORLD_W
        iy, ix = int(y), int(x)
        best_d = 1e9; best = None
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = iy+dy, ix+dx
                if 0 <= ny < H and 0 <= nx < W:
                    if self.terrain[ny, nx] == T_RIVER:
                        d = abs(dy) + abs(dx)
                        if d < best_d:
                            best_d = d; best = (ny, nx)
        return best

    def near_fire(self, y, x, radius=3):
        H, W = WORLD_H, WORLD_W
        iy, ix = int(y), int(x)
        y0, y1 = max(0, iy-radius), min(H, iy+radius+1)
        x0, x1 = max(0, ix-radius), min(W, ix+radius+1)
        return np.any(self.fire_intensity[y0:y1, x0:x1] > 0)

    def near_structure(self, y, x, stype, radius=3):
        H, W = WORLD_H, WORLD_W
        iy, ix = int(y), int(x)
        y0, y1 = max(0, iy-radius), min(H, iy+radius+1)
        x0, x1 = max(0, ix-radius), min(W, ix+radius+1)
        return np.any(self.structures[y0:y1, x0:x1] == stype)

    def inside_house(self, y, x):
        iy, ix = int(y), int(x)
        if 0 <= iy < WORLD_H and 0 <= ix < WORLD_W:
            return self.structures[iy, ix] == S_HOUSE
        return False

    def find_nearest(self, y, x, obj_type_val, radius=30):
        H, W = WORLD_H, WORLD_W
        iy, ix = int(y), int(x)
        best_d = 1e9; best = None
        y0, y1 = max(0, iy-radius), min(H, iy+radius+1)
        x0, x1 = max(0, ix-radius), min(W, ix+radius+1)
        for ty in range(y0, y1):
            for tx in range(x0, x1):
                if self.obj_type[ty, tx] == obj_type_val and self.obj_growth[ty, tx] >= 1.0:
                    d = abs(ty-iy) + abs(tx-ix)
                    if d < best_d:
                        best_d = d; best = (ty, tx)
        return best

    def find_nearest_river(self, y, x, radius=55):
        H, W = WORLD_H, WORLD_W
        iy, ix = int(y), int(x)
        best_d = 1e9; best = None
        y0, y1 = max(0, iy-radius), min(H, iy+radius+1)
        x0, x1 = max(0, ix-radius), min(W, ix+radius+1)
        for ty in range(y0, y1):
            for tx in range(x0, x1):
                if self.terrain[ty, tx] == T_RIVER:
                    d = abs(ty-iy) + abs(tx-ix)
                    if d < best_d:
                        best_d = d; best = (ty, tx)
        return best

    def find_empty_ground(self, cy, cx, radius=6):
        H, W = WORLD_H, WORLD_W
        for r in range(1, radius+1):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    ty, tx = int(cy)+dy, int(cx)+dx
                    if 0 <= ty < H and 0 <= tx < W:
                        if (self.terrain[ty, tx] in (T_GRASS, T_DIRT)
                                and self.structures[ty, tx] == S_NONE
                                and self.obj_type[ty, tx] == O_NONE):
                            return (ty, tx)
        return None
