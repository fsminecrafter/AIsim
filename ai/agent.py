"""
Agent AI -- behavior tree, learning weights, genetic inheritance.

Recovered from the remaining bytecode and surrounding project files after the
source file was accidentally replaced by a unified diff.
"""

import math
import random

import engine.config as cfg
from ai.actions import (
    ACTION_NAMES,
    NUM_ACTIONS,
    A_FLEE_FIRE,
    A_DRINK,
    A_SEEK_SAFETY,
    A_FIND_WATER,
    A_FIND_FOOD,
    A_SEEK_WARMTH,
    A_FORAGE,
    A_STOCKPILE,
    A_CRAFT_TOOL,
    A_BUILD,
    A_SEEK_MATE,
    A_WANDER,
)
from engine.config import *
from world.world import *


ITEM_FOOD_VALUE = {
    "berry": 15,
    "wheat": 8,
    "bread": 40,
    "cooked_meat": 50,
}

CRAFT_RECIPES = {
    "sharpened_stone": {"rock": 2},
    "axe": {"sharpened_stone": 1, "wood": 1},
    "hoe": {"sharpened_stone": 1, "wood": 1},
    "shovel": {"sharpened_stone": 1, "wood": 1},
    "wooden_block": {"wood": 2},
    "fireplace": {"rock": 4, "wood": 2},
    "bread": {"wheat": 2},
}

CRAFT_TIME = {
    "sharpened_stone": 3.0,
    "axe": 5.0,
    "hoe": 5.0,
    "shovel": 5.0,
    "wooden_block": 4.0,
    "fireplace": 10.0,
    "bread": 10.0,
}

TOOL_DURABILITY = 100
_agent_id_counter = [0]


def new_id():
    _agent_id_counter[0] += 1
    return f"A{_agent_id_counter[0]:04d}"


def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


class Agent:
    def __init__(
        self,
        x,
        y,
        gender=None,
        parent_weights=None,
        world=None,
        hidden_size=None,
    ):
        self.id = new_id()
        self.gender = gender or random.choice(["male", "female"])
        self.age = 0.0
        self.winters = 0
        self.stage = "adult"

        self.hunger = random.uniform(60, 90)
        self.thirst = random.uniform(60, 90)
        self.warmth = random.uniform(70, 90)
        self.mood = random.uniform(60, 90)
        self.health = 100.0

        self.x = float(x)
        self.y = float(y)
        self.path = []
        self.target = None

        self.current_action = A_WANDER
        self.action_timer = 0.0
        self.action_target = None
        self.crafting_item = None
        self.craft_remaining = 0.0
        self.is_dead = False

        self.inventory = {}
        self.inv_limit = 10

        self.memory = {
            O_TREE: [],
            O_BERRY: [],
            O_WHEAT: [],
            "river": [],
            "home": [],
        }
        self.home_y = None
        self.home_x = None

        self.partner_id = None
        self.children_ids = []
        self.preg_timer = None

        self.alert_fire = False
        self.alert_timer = 0.0

        if parent_weights is not None:
            self.weights = [
                max(0.3, min(3.0, parent_weights[i] + random.gauss(0, MUTATION_RATE)))
                for i in range(NUM_ACTIONS)
            ]
        else:
            self.weights = [1.0] * NUM_ACTIONS

        # ── Exploration trait ─────────────────────────────────
        # 0.0 = strong homebody (pulled hard toward familiar tiles)
        # 1.0 = restless wanderer (home pull completely ignored)
        # Survival actions (fire/thirst/hunger/cold) bypass this entirely.
        self.exploration = 0.3
        self.visit_heat  = {}   # (ty, tx) -> visit count
        self._heat_timer = 0.0  # seconds between heat recordings

        self.event_log = []
        self.prev_vitals = (self.hunger, self.thirst, self.warmth, self.mood)

        self.food_eaten = 0
        self.children_born = 0
        self.fires_escaped = 0
        self.generation = 0

    def inv_count(self, itype):
        return self.inventory.get(itype, {}).get("quantity", 0)

    def inv_total_slots(self):
        return sum(item["quantity"] for item in self.inventory.values())

    def inv_add(self, itype, qty=1, freshness=None, durability=None):
        if qty <= 0:
            return False
        item = self.inventory.setdefault(
            itype,
            {
                "quantity": 0,
                "freshness": freshness,
                "durability": durability if durability is not None else TOOL_DURABILITY,
            },
        )
        item["quantity"] += qty
        if freshness is not None:
            current = item.get("freshness")
            item["freshness"] = freshness if current is None else max(current, freshness)
        if durability is not None:
            item["durability"] = durability
        return True

    def inv_remove(self, itype, qty=1):
        if self.inv_count(itype) < qty:
            return False
        self.inventory[itype]["quantity"] -= qty
        if self.inventory[itype]["quantity"] <= 0:
            del self.inventory[itype]
        return True

    def has_recipe(self, recipe_name):
        needs = CRAFT_RECIPES.get(recipe_name)
        if not needs:
            return False
        return all(self.inv_count(mat) >= qty for mat, qty in needs.items())

    def consume_recipe(self, recipe_name):
        needs = CRAFT_RECIPES.get(recipe_name, {})
        for mat, qty in needs.items():
            self.inv_remove(mat, qty)

    def total_food(self):
        total = 0
        for ftype, value in ITEM_FOOD_VALUE.items():
            total += self.inv_count(ftype) * value
        return total

    def has_tool(self, tool):
        return self.inv_count(tool) > 0

    def record_event(self, action_id, reward):
        """Adjust weight for action based on reward."""
        if action_id is None:
            return
        self.weights[action_id] = max(
            0.3,
            min(3.0, self.weights[action_id] + LEARNING_RATE * reward),
        )
        self.event_log.append((action_id, reward))
        if len(self.event_log) > 20:
            self.event_log.pop(0)

    def _calc_reward(self):
        h, t, w, m = self.hunger, self.thirst, self.warmth, self.mood
        ph, pt, pw, pm = self.prev_vitals
        reward = (h - ph) * 0.4 + (t - pt) * 0.4 + (w - pw) * 0.1 + (m - pm) * 0.1
        if min(h, t, w, m) < 20.0:
            reward -= 0.2
        self.prev_vitals = (h, t, w, m)
        return reward

    def decide(self, world, agents):
        if self.stage == "baby":
            return self._baby_decide(world, agents)

        w = list(self.weights)
        iy, ix = int(self.y), int(self.x)
        if 0 <= iy < WORLD_H and 0 <= ix < WORLD_W:
            if world.fire_intensity[iy, ix] > 0:
                return A_FLEE_FIRE
        if self.alert_fire:
            return A_FLEE_FIRE

        if self.thirst < 15:
            return A_DRINK
        if self.health < 25:
            return A_SEEK_SAFETY
        if self.thirst < 30:
            return A_FIND_WATER
        if self.hunger < 50:
            return A_FIND_FOOD
        if self.warmth < 50:
            return A_SEEK_WARMTH

        season = world.season_name
        if season in ("winter", "autumn") and self.total_food() < 80:
            w[A_STOCKPILE] += 1.0
        if not self.has_tool("axe") and (
            self.inv_count("rock") >= 2 or self.has_recipe("axe")
        ):
            w[A_CRAFT_TOOL] += 0.8
        if not world.near_structure(self.y, self.x, S_FIREPLACE, 6):
            if self.has_recipe("fireplace"):
                w[A_BUILD] += 1.2
            elif self.has_recipe("wooden_block"):
                w[A_CRAFT_TOOL] += 0.4
        if self.home_y is None and self.inv_count("wooden_block") >= 2:
            w[A_BUILD] += 0.5
        if self._can_reproduce(world, agents):
            w[A_SEEK_MATE] += 1.5

        return max(range(NUM_ACTIONS), key=lambda i: w[i])

    def _baby_decide(self, world, agents):
        iy, ix = int(self.y), int(self.x)
        if 0 <= iy < WORLD_H and 0 <= ix < WORLD_W and world.fire_intensity[iy, ix] > 0:
            return A_FLEE_FIRE
        if self.thirst < 30:
            return A_FIND_WATER
        if self.hunger < 40:
            return A_FIND_FOOD
        if self.warmth < 40:
            return A_SEEK_WARMTH
        return A_WANDER

    def _can_reproduce(self, world, agents):
        return (
            world.season_name == "summer"
            and self.stage == "adult"
            and self.preg_timer is None
            and self.gender == "female"
            and self.mood > 60
            and self.hunger > 50
            and self.total_food() >= 5
            and len(agents) < MAX_AGENTS
        )

    def tick(self, world, agents, real_dt):
        if self.is_dead:
            return

        sim_dt = real_dt * getattr(cfg, "SIM_SPEED", SIM_SPEED)
        self.age += sim_dt
        prev_winters = self.winters
        self.winters = int(self.age / max(1.0, DAY_DURATION * SEASON_DAYS * 4))
        if self.winters > prev_winters:
            self._on_winter_tick(world)

        for itype, idata in list(self.inventory.items()):
            if idata.get("freshness") is not None:
                idata["freshness"] = max(0.0, idata["freshness"] - sim_dt * 0.05)
                if idata["freshness"] <= 0:
                    self.inv_remove(itype, idata["quantity"])

        weather = world.weather
        season = world.season_name
        wm = world.WEATHER_MOOD_MULT.get(weather, 1.0)
        whm = world.WEATHER_HUNGER_MULT.get(weather, 1.0)
        www = world.WEATHER_WARMTH_MULT.get(weather, 1.0)
        scm = world.SEASON_COLD_MULT.get(season, 1.0)
        baby_mod = 0.7 if self.stage == "baby" else 1.0
        elder_sp = 1.3 if self.stage == "elder" else 1.0
        near_fp = world.near_structure(self.y, self.x, S_FIREPLACE, 3)
        roof = world.inside_house(self.y, self.x)

        self.hunger = max(0, self.hunger - HUNGER_DECAY * whm * sim_dt * baby_mod)
        self.thirst = max(0, self.thirst - THIRST_DECAY * sim_dt)
        self.mood = max(0, self.mood - MOOD_DECAY * wm * sim_dt)

        warmth_loss = WARMTH_DECAY * www * scm * sim_dt * elder_sp
        if near_fp:
            warmth_loss *= 0.3
            self.warmth = min(100, self.warmth + 3.0 * sim_dt)
        if roof:
            warmth_loss *= 0.5
            self.mood = min(100, self.mood + 0.5 * sim_dt)
        self.warmth = max(0, self.warmth - warmth_loss)

        if min(self.hunger, self.thirst, self.warmth, self.mood) <= 0:
            self.health -= 8 * sim_dt
        elif min(self.hunger, self.thirst, self.warmth) < 15:
            self.health -= 2 * sim_dt
        else:
            self.health = min(100, self.health + 0.4 * sim_dt)

        if self.health <= 0:
            self.is_dead = True
            drops = [
                {"type": itype, **idata}
                for itype, idata in self.inventory.items()
                if idata.get("quantity", 0) > 0
            ]
            if drops:
                world.drop_items(self.y, self.x, drops)
            return

        if world.near_fire(self.y, self.x, 5):
            self.alert_fire = True
            self.alert_timer = 10.0
        elif self.alert_timer > 0:
            self.alert_timer -= sim_dt
        else:
            self.alert_fire = False

        # ── Record visit heatmap (every ~2 sim-seconds) ───────
        self._heat_timer += sim_dt
        if self._heat_timer >= 2.0:
            self._heat_timer = 0.0
            key = (int(self.y), int(self.x))
            self.visit_heat[key] = self.visit_heat.get(key, 0) + 1
            # Cap heat dict size so it doesn't grow unbounded
            if len(self.visit_heat) > 400:
                # Drop the coldest (least-visited) tiles
                sorted_keys = sorted(self.visit_heat, key=lambda k: self.visit_heat[k])
                for k in sorted_keys[:100]:
                    del self.visit_heat[k]

        action = self.decide(world, agents)
        self.current_action = action
        speed = 2.0 if world.is_day else 1.2
        prev_v = self.prev_vitals
        self._execute(action, world, agents, sim_dt, speed)
        reward = self._calc_reward()
        if abs(reward) > 0.01 or prev_v != self.prev_vitals:
            self.record_event(action, reward)

    def _on_winter_tick(self, world):
        if self.winters >= 15:
            self.is_dead = True
        elif self.winters >= 13:
            self.stage = "elder"
        elif self.winters >= 3:
            self.stage = "adult"
        else:
            self.stage = "baby"

    def _execute(self, action, world, agents, sim_dt, speed):
        if action == A_FLEE_FIRE:
            self._flee_fire(world, sim_dt, speed, all_agents=agents)
        elif action in (A_DRINK, A_FIND_WATER):
            self._seek_water(world, sim_dt, speed)
        elif action in (A_FIND_FOOD, A_FORAGE):
            self._seek_food(world, sim_dt, speed)
        elif action == A_STOCKPILE:
            self._seek_food(world, sim_dt, speed, stockpile=True)
        elif action == A_SEEK_WARMTH:
            self._seek_warmth(world, sim_dt, speed)
        elif action == A_CRAFT_TOOL:
            self._craft(world, sim_dt)
        elif action == A_BUILD:
            self._build(world, sim_dt)
        elif action == A_SEEK_MATE:
            self._seek_mate(world, agents, sim_dt, speed)
        elif action == A_SEEK_SAFETY:
            self._seek_safety(world, sim_dt, speed)
        else:
            self._wander(world, sim_dt, speed)
        self._loot_ground(world)

    def _move_toward(self, ty, tx, world, sim_dt, speed):
        dy = ty - self.y
        dx = tx - self.x
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.5:
            return True
        step = min(dist, max(0.0, speed * sim_dt))
        if dist > 0:
            self.y += dy / dist * step
            self.x += dx / dist * step
        self.y = max(0, min(WORLD_H - 1, self.y))
        self.x = max(0, min(WORLD_W - 1, self.x))
        return False

    def _flee_fire(self, world, sim_dt, speed, all_agents=None):
        iy, ix = int(self.y), int(self.x)
        best_y, best_x = iy, ix
        if 0 <= iy < WORLD_H and 0 <= ix < WORLD_W:
            fires = []
            for yy in range(max(0, iy - 5), min(WORLD_H, iy + 6)):
                for xx in range(max(0, ix - 5), min(WORLD_W, ix + 6)):
                    if world.fire_intensity[yy, xx] > 0:
                        fires.append((yy, xx))
            if fires:
                fy, fx = min(fires, key=lambda p: abs(p[0] - iy) + abs(p[1] - ix))
                best_y = iy + (iy - fy) * 2
                best_x = ix + (ix - fx) * 2
            else:
                best_y = iy + random.choice([-1, 1]) * 5
                best_x = ix + random.choice([-1, 1]) * 5
        best_y = max(0, min(WORLD_H - 1, best_y))
        best_x = max(0, min(WORLD_W - 1, best_x))
        arrived = self._move_toward(best_y, best_x, world, sim_dt, speed * 2.0)
        if arrived:
            self.fires_escaped += 1
            self.alert_fire = False
            self.alert_timer = 0.0

    def _seek_water(self, world, sim_dt, speed):
        if world.near_river(self.y, self.x, 1):
            self.thirst = min(100, self.thirst + 40 * sim_dt)
            self.record_event(A_DRINK, 1.0)
            return
        river = world.find_nearest_river(self.y, self.x, 40)
        if river:
            self.memory.setdefault("river", []).append(river)
            if len(self.memory["river"]) > 5:
                self.memory["river"].pop(0)
            self._move_toward(river[0], river[1], world, sim_dt, speed)
        else:
            self._wander(world, sim_dt, speed)

    def _seek_food(self, world, sim_dt, speed, stockpile=False):
        if not stockpile:
            for ftype in ("bread", "berry", "wheat"):
                if self.inv_count(ftype) > 0 and self.hunger < 90:
                    value = ITEM_FOOD_VALUE.get(ftype, 10)
                    self.inv_remove(ftype, 1)
                    self.hunger = min(100, self.hunger + value)
                    self.food_eaten += 1
                    self.record_event(A_FIND_FOOD, 1.0)
                    return

        targets = []
        berry_pos = world.find_nearest(self.y, self.x, O_BERRY, 25)
        wheat_pos = world.find_nearest(self.y, self.x, O_WHEAT, 25)
        rock_pos = world.find_nearest(self.y, self.x, O_ROCK_OBJ, 12)
        if berry_pos:
            targets.append((berry_pos, O_BERRY))
        if wheat_pos and (stockpile or self.hunger < 80):
            targets.append((wheat_pos, O_WHEAT))
        if rock_pos and self.inv_count("rock") < 4:
            targets.append((rock_pos, O_ROCK_OBJ))
        if stockpile:
            tree = world.find_nearest(self.y, self.x, O_TREE, 25)
            if tree:
                targets.append((tree, O_TREE))

        if not targets:
            self._wander(world, sim_dt, speed)
            return

        target, obj_t = min(
            targets,
            key=lambda t: abs(t[0][0] - self.y) + abs(t[0][1] - self.x),
        )
        arrived = self._move_toward(target[0], target[1], world, sim_dt, speed)
        if arrived:
            item = world.harvest(target[0], target[1])
            if item is None and obj_t == O_ROCK_OBJ:
                world.remove_object(target[0], target[1])
                item = {"type": "rock", "quantity": 1, "freshness": None}
            if item:
                self.inv_add(
                    item.get("type", "berry"),
                    item.get("quantity", 1),
                    item.get("freshness"),
                )
                self.memory.setdefault(obj_t, []).append(target)
                if len(self.memory[obj_t]) > 5:
                    self.memory[obj_t].pop(0)
                self.record_event(A_FORAGE, 0.5)

    def _seek_warmth(self, world, sim_dt, speed):
        if world.near_structure(self.y, self.x, S_FIREPLACE, 3):
            self.warmth = min(100, self.warmth + 8 * sim_dt)
            return
        best = None
        best_d = 1000000000.0
        for ty in range(WORLD_H):
            for tx in range(WORLD_W):
                if world.structures[ty, tx] == S_FIREPLACE:
                    d = abs(ty - self.y) + abs(tx - self.x)
                    if d < best_d:
                        best_d = d
                        best = (ty, tx)
        if best:
            self._move_toward(best[0], best[1], world, sim_dt, speed)
        elif self.has_recipe("fireplace"):
            self._build(world, sim_dt)
        else:
            self._seek_food(world, sim_dt, speed, stockpile=True)

    def _pickup_nearby_items(self, world, radius=2):
        """Scavenge any dropped items within radius tiles before crafting."""
        iy, ix = int(self.y), int(self.x)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < WORLD_H and 0 <= nx < WORLD_W:
                    items = world.pick_dropped(ny, nx)
                    for item in items:
                        if self.inv_total_slots() < self.inv_limit:
                            self.inv_add(
                                item.get("type", "rock"),
                                item.get("quantity", 1),
                                item.get("freshness"),
                            )
                        else:
                            world.drop_items(ny, nx, [item])

    def _craft(self, world, sim_dt):
        """Craft items. Consumes materials and waits out the craft timer."""
        # ── still busy crafting ───────────────────────────────
        if self.craft_remaining > 0:
            self.craft_remaining -= sim_dt
            if self.craft_remaining <= 0:
                self.craft_remaining = 0.0
                item = self.crafting_item
                self.crafting_item = None
                self.inv_add(item, 1)
                self.record_event(A_CRAFT_TOOL, 1.0)
            return

        # ── already started a craft this frame (guard) ────────
        if self.crafting_item is not None:
            return

        # Scoop up any items nearby before trying to craft
        self._pickup_nearby_items(world, radius=2)

        # Priority-ordered recipes. Skip tools already owned (no hoarding).
        for recipe in ("bread", "fireplace", "axe", "hoe", "shovel",
                       "sharpened_stone", "wooden_block"):
            # bread only near a fireplace
            if recipe == "bread" and not world.near_structure(
                    self.y, self.x, S_FIREPLACE, 3):
                continue
            # Don't craft another axe/hoe/shovel if one is already in inventory
            if recipe in ("axe", "hoe", "shovel") and self.has_tool(recipe):
                continue
            # Don't craft sharpened_stone if we already have one and no axe recipe yet
            if recipe == "sharpened_stone" and self.inv_count("sharpened_stone") >= 2:
                continue
            if self.has_recipe(recipe):
                self.consume_recipe(recipe)
                self.crafting_item = recipe
                self.craft_remaining = CRAFT_TIME.get(recipe, 3.0)
                return

        # Nothing to craft — go gather materials instead
        self._seek_food(world, sim_dt, 1.0, stockpile=True)

    def _build(self, world, sim_dt):
        """Build structures. Planks can be placed freely at any empty spot near agent."""
        if self.has_recipe("fireplace") and not world.near_structure(self.y, self.x, S_FIREPLACE, 6):
            spot = world.find_empty_ground(self.y, self.x, 4)
            if spot:
                arrived = self._move_toward(spot[0], spot[1], world, sim_dt, 1.0)
                if arrived:
                    self.consume_recipe("fireplace")
                    world.place_structure(spot[0], spot[1], S_FIREPLACE)
                    self.record_event(A_BUILD, 1.0)
            return

        if self.inv_count("wooden_block") >= 2:
            # Agent chooses placement freely: near home if known, else current pos
            if self.home_y is not None:
                # extend existing home — place adjacent to home tile
                spot = world.find_empty_ground(self.home_y, self.home_x, 3)
            else:
                # brand new home — place right where agent is standing
                spot = world.find_empty_ground(self.y, self.x, 3)
            if spot:
                # Move toward chosen spot
                arrived = self._move_toward(spot[0], spot[1], world, sim_dt, 1.0)
                if arrived:
                    self.inv_remove("wooden_block", 2)
                    world.place_structure(spot[0], spot[1], S_HOUSE)
                    self.home_y, self.home_x = spot[0], spot[1]
                    self.memory.setdefault("home", []).append(spot)
                    self.record_event(A_BUILD, 1.5)
            return

        # Craft a wooden block from wood in hand or nearby
        self._pickup_nearby_items(world, radius=2)
        if self.has_recipe("wooden_block"):
            self.consume_recipe("wooden_block")
            self.inv_add("wooden_block", 1)
            self.record_event(A_BUILD, 0.3)
        else:
            self._wander(world, sim_dt, 1.0)

    def _seek_mate(self, world, agents, sim_dt, speed):
        best = None
        best_d = 1000000000.0
        for ag in agents:
            if ag is self or ag.is_dead:
                continue
            if ag.gender == self.gender or ag.stage != "adult":
                continue
            d = abs(ag.x - self.x) + abs(ag.y - self.y)
            if d < best_d:
                best_d = d
                best = ag
        if best is None:
            self._wander(world, sim_dt, speed)
            return
        arrived = self._move_toward(best.y, best.x, world, sim_dt, speed)
        if arrived and self.gender == "female" and self.preg_timer is None:
            self.partner_id = best.id
            best.partner_id = self.id
            self.preg_timer = 30.0
            self.children_born += 1
            self.record_event(A_SEEK_MATE, 1.0)

    def _seek_safety(self, world, sim_dt, speed):
        if self.home_y is not None:
            self._move_toward(self.home_y, self.home_x, world, sim_dt, speed * 1.2)
        else:
            self._wander(world, sim_dt, speed)

    def _wander(self, world, sim_dt, speed):
        """
        Wander behaviour shaped by exploration trait (0–1).

        Low exploration  → agent is pulled toward its hottest (most-visited)
                           tiles and picks short-range wander targets nearby.
        High exploration → large random targets, home-pull ignored entirely.

        Survival actions completely bypass this method, so thirst/hunger/fire
        never interact with the exploration trait.
        """
        at_target = (
            self.target is None
            or (abs(self.x - self.target[1]) < 1 and abs(self.y - self.target[0]) < 1)
        )

        if at_target:
            home_y = self.home_y if self.home_y is not None else WORLD_H / 2
            home_x = self.home_x if self.home_x is not None else WORLD_W / 2

            # Base wander radius scales with exploration (4 to 35 tiles)
            r = 4.0 + self.exploration * 31.0

            # Night-time restriction applies uniformly (safety)
            if not world.is_day:
                r *= 0.5

            if self.exploration < 0.5 and self.visit_heat:
                # --- Homebody: bias toward a warm (familiar) tile ----------
                # Find the hottest tile within 2×r of home
                best_key   = None
                best_score = -1
                cx, cy = home_x, home_y
                for (ky, kx), heat in self.visit_heat.items():
                    dist_from_home = abs(ky - home_y) + abs(kx - home_x)
                    if dist_from_home > r * 2:
                        continue
                    # Score: heat × (1 - how far it is / max range)
                    score = heat * (1.0 - dist_from_home / max(r * 2, 1))
                    if score > best_score:
                        best_score = score
                        best_key   = (ky, kx)

                if best_key and best_score > 0:
                    # Add small jitter so they don't freeze on one spot
                    jitter = 3.0 * (1.0 - self.exploration)
                    ty = _clamp(best_key[0] + random.uniform(-jitter, jitter),
                                0, WORLD_H - 1)
                    tx = _clamp(best_key[1] + random.uniform(-jitter, jitter),
                                0, WORLD_W - 1)
                else:
                    # No heat data yet — stay close to home
                    ty = _clamp(home_y + random.uniform(-r * 0.4, r * 0.4), 0, WORLD_H - 1)
                    tx = _clamp(home_x + random.uniform(-r * 0.4, r * 0.4), 0, WORLD_W - 1)
            else:
                # --- Explorer: random target within full radius --------------
                ty = _clamp(home_y + random.uniform(-r, r), 0, WORLD_H - 1)
                tx = _clamp(home_x + random.uniform(-r, r), 0, WORLD_W - 1)

            self.target = (ty, tx)

        self._move_toward(self.target[0], self.target[1], world, sim_dt, speed * 0.6)

    def _loot_ground(self, world):
        items = world.pick_dropped(int(self.y), int(self.x))
        for item_dict in items:
            if self.inv_total_slots() >= self.inv_limit:
                continue
            self.inv_add(
                item_dict.get("type", "rock"),
                item_dict.get("quantity", 1),
                item_dict.get("freshness"),
            )

    def tick_pregnancy(self, world, agents, sim_dt):
        if self.preg_timer is None:
            return None
        self.preg_timer -= sim_dt
        if self.preg_timer > 0:
            return None

        self.preg_timer = None
        child = Agent(
            self.x + random.uniform(-1, 1),
            self.y + random.uniform(-1, 1),
            world=world,
        )
        child.stage = "baby"
        child.winters = 0
        child.age = 0.0
        child.generation = self.generation + 1

        child.memory = {}
        for k, v in self.memory.items():
            child.memory[k] = v[: int(len(v) * KNOWLEDGE_INHERIT)]

        # Pass on a fraction of the parent's heatmap (familiar ground)
        if self.visit_heat:
            sorted_heat = sorted(self.visit_heat.items(), key=lambda kv: -kv[1])
            inherited   = sorted_heat[:int(len(sorted_heat) * KNOWLEDGE_INHERIT)]
            child.visit_heat = dict(inherited)

        partner = next((a for a in agents if a.id == self.partner_id), None)
        if partner:
            child.weights = [
                max(
                    0.3,
                    min(
                        3.0,
                        self.weights[i] * 0.5
                        + partner.weights[i] * 0.5
                        + random.gauss(0, MUTATION_RATE),
                    ),
                )
                for i in range(NUM_ACTIONS)
            ]
            # Exploration trait: average parents + mutation, clamped 0–1
            child.exploration = max(0.0, min(1.0,
                self.exploration * 0.5
                + partner.exploration * 0.5
                + random.gauss(0, 0.05)
            ))
        else:
            child.weights = [
                max(0.3, min(3.0, w + random.gauss(0, MUTATION_RATE)))
                for w in self.weights
            ]
            child.exploration = max(0.0, min(1.0,
                self.exploration + random.gauss(0, 0.05)
            ))

        if self.inv_count("berry") > 1:
            self.inv_remove("berry", 1)
            child.inv_add("berry", 1, freshness=100.0)
        self.children_ids.append(child.id)
        return child