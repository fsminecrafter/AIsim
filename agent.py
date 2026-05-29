"""
Agent AI — behavior tree, learning weights, genetic inheritance.

Each agent has:
  - A neural "preference vector" (12 floats) that biases its decision weights
  - An event memory (last 20 events) that updates weights online
  - On birth, children inherit parent weights + mutation

Learning: Reinforcement-style. When an action leads to a positive vital change,
that action's weight increases. Weights are passed to children (weighted average
of both parents + noise).
"""

import math, random, uuid
from world.world import *
from engine.config import *

# ─── Action IDs ───────────────────────────────────────────────
A_FLEE_FIRE    = 0
A_DRINK        = 1
A_SEEK_SAFETY  = 2
A_FIND_WATER   = 3
A_FIND_FOOD    = 4
A_SEEK_WARMTH  = 5
A_FORAGE       = 6
A_STOCKPILE    = 7
A_CRAFT_TOOL   = 8
A_BUILD        = 9
A_SEEK_MATE    = 10
A_WANDER       = 11
NUM_ACTIONS    = 12

ACTION_NAMES = [
    "🔥Flee", "💧Drink", "🛡Safety", "🔍Water", "🍎Food",
    "🌡Warmth", "🌾Forage", "📦Stock", "🔨Craft", "🏠Build",
    "❤Mate", "👣Wander"
]

ITEM_FOOD_VALUE = {
    "berry": 15, "wheat": 8, "bread": 40, "cooked_meat": 50
}

CRAFT_RECIPES = {
    "sharpened_stone": {"rock": 2},
    "axe":   {"sharpened_stone": 1, "wood": 1},
    "hoe":   {"sharpened_stone": 1, "wood": 1},
    "shovel":{"sharpened_stone": 1, "wood": 1},
    "wooden_block": {"wood": 2},
    "fireplace":    {"rock": 4, "wood": 2},
    "bread":        {"wheat": 2},  # requires fireplace nearby
}

CRAFT_TIME = {
    "sharpened_stone": 3.0,
    "axe": 5.0, "hoe": 5.0, "shovel": 5.0,
    "wooden_block": 4.0,
    "fireplace": 10.0,
    "bread": 10.0,
}

TOOL_DURABILITY = 100

_agent_id_counter = [0]

def new_id():
    _agent_id_counter[0] += 1
    return f"A{_agent_id_counter[0]:04d}"


class Agent:
    def __init__(self, x, y, gender=None, parent_weights=None, world=None):
        self.id       = new_id()
        self.gender   = gender or random.choice(["male", "female"])
        self.age      = 0.0          # real-sim seconds (NOT winters yet)
        self.winters  = 0
        self.stage    = "adult"      # baby / adult / elder

        # Vitals
        self.hunger  = random.uniform(60, 90)
        self.thirst  = random.uniform(60, 90)
        self.warmth  = random.uniform(70, 90)
        self.mood    = random.uniform(60, 90)
        self.health  = 100.0

        # Position
        self.x = float(x)
        self.y = float(y)
        self.path     = []
        self.target   = None

        # Action state
        self.current_action  = A_WANDER
        self.action_timer    = 0.0
        self.action_target   = None    # (y,x) tile
        self.crafting_item   = None
        self.craft_remaining = 0.0
        self.is_dead         = False

        # Inventory {type: {"quantity": int, "freshness": float or None, "durability": int}}
        self.inventory  = {}
        self.inv_limit  = 10

        # Memory: known resource locations {O_TREE: [(y,x),...], ...}
        self.memory     = {O_TREE: [], O_BERRY: [], O_WHEAT: [], "river": [], "home": []}
        self.home_y     = None
        self.home_x     = None

        # Social
        self.partner_id   = None
        self.children_ids = []
        self.preg_timer   = None   # seconds countdown

        # Alert state (fleeing from fire, shared danger)
        self.alert_fire    = False
        self.alert_timer   = 0.0

        # ── Learning weights ──────────────────────────────────
        # Each of NUM_ACTIONS has a learned "urgency bias" (0.5–2.0)
        # Higher = agent will trigger this action at higher thresholds
        if parent_weights is not None:
            noise = [random.gauss(0, MUTATION_RATE) for _ in range(NUM_ACTIONS)]
            self.weights = [
                max(0.3, min(3.0, parent_weights[i] + noise[i]))
                for i in range(NUM_ACTIONS)
            ]
        else:
            self.weights = [1.0] * NUM_ACTIONS

        # Recent event log for online learning
        self.event_log   = []
        self.prev_vitals = (self.hunger, self.thirst, self.warmth, self.mood)

        # Stats for UI
        self.food_eaten      = 0
        self.children_born   = 0
        self.fires_escaped   = 0
        self.generation      = 0

    # ═══════════════════════════════════════════════════════
    #  Inventory helpers
    # ═══════════════════════════════════════════════════════

    def inv_count(self, itype):
        if itype in self.inventory:
            return self.inventory[itype]["quantity"]
        return 0

    def inv_total_slots(self):
        return sum(1 for v in self.inventory.values() if v["quantity"] > 0)

    def inv_add(self, itype, qty=1, freshness=None, durability=None):
        if itype in self.inventory:
            self.inventory[itype]["quantity"] += qty
        else:
            self.inventory[itype] = {
                "quantity": qty,
                "freshness": freshness,
                "durability": durability or TOOL_DURABILITY
            }

    def inv_remove(self, itype, qty=1):
        if itype not in self.inventory: return False
        if self.inventory[itype]["quantity"] < qty: return False
        self.inventory[itype]["quantity"] -= qty
        if self.inventory[itype]["quantity"] <= 0:
            del self.inventory[itype]
        return True

    def has_recipe(self, recipe_name):
        needs = CRAFT_RECIPES.get(recipe_name, {})
        for mat, qty in needs.items():
            if self.inv_count(mat) < qty: return False
        return True

    def consume_recipe(self, recipe_name):
        needs = CRAFT_RECIPES.get(recipe_name, {})
        for mat, qty in needs.items():
            self.inv_remove(mat, qty)

    def total_food(self):
        total = 0
        for ft in ITEM_FOOD_VALUE:
            total += self.inv_count(ft)
        return total

    def has_tool(self, tool):
        return self.inv_count(tool) > 0

    # ═══════════════════════════════════════════════════════
    #  Learning
    # ═══════════════════════════════════════════════════════

    def record_event(self, action_id, reward):
        """Adjust weight for action based on reward (+/-)."""
        self.weights[action_id] = max(0.3, min(3.0,
            self.weights[action_id] + LEARNING_RATE * reward
        ))
        self.event_log.append((action_id, reward))
        if len(self.event_log) > 20:
            self.event_log.pop(0)

    def _calc_reward(self):
        h, t, w, m = self.hunger, self.thirst, self.warmth, self.mood
        ph, pt, pw, pm = self.prev_vitals
        dh = h - ph
        dt = t - pt
        dw = w - pw
        dm = m - pm
        reward = dh*0.4 + dt*0.4 + dw*0.1 + dm*0.1
        return reward / 20.0  # normalise

    # ═══════════════════════════════════════════════════════
    #  Behaviour tree  (priority-ordered)
    # ═══════════════════════════════════════════════════════

    def decide(self, world, agents):
        w = self.weights

        # Babies have a simplified tree
        if self.stage == "baby":
            return self._baby_decide(world, agents)

        # ── CRITICAL ──────────────────────────────────────
        if world.fire_intensity[int(self.y), int(self.x)] > 0:
            return A_FLEE_FIRE
        if self.alert_fire:
            return A_FLEE_FIRE
        if self.thirst < 5  * w[A_DRINK]:      return A_DRINK
        if self.health < 15 * w[A_SEEK_SAFETY]: return A_SEEK_SAFETY

        # ── URGENT ────────────────────────────────────────
        if self.thirst < 25 * w[A_FIND_WATER]: return A_FIND_WATER
        if self.hunger < 20 * w[A_FIND_FOOD]:  return A_FIND_FOOD

        # ── IMPORTANT ─────────────────────────────────────
        if self.warmth < 30 * w[A_SEEK_WARMTH]: return A_SEEK_WARMTH
        if self.hunger < 50 * w[A_FORAGE]:      return A_FORAGE

        # ── LONG TERM ─────────────────────────────────────
        season = world.season_name
        if season == "winter" and self.total_food() < 20 * w[A_STOCKPILE]:
            return A_STOCKPILE
        if season == "autumn" and self.total_food() < 30 * w[A_STOCKPILE]:
            return A_STOCKPILE

        # craft if needed
        if not self.has_tool("axe") and (self.inv_count("rock") >= 2 or self.inv_count("sharpened_stone") >= 1):
            return A_CRAFT_TOOL
        if not self.has_tool("hoe") and self.has_recipe("hoe"):
            return A_CRAFT_TOOL
        if self.inv_count("rock") < 2 and self.inv_count("sharpened_stone") == 0:
            return A_FORAGE  # go get rocks

        # build fireplace if cold and no nearby
        if (world.season_idx in (2, 3) and not world.near_structure(self.y, self.x, S_FIREPLACE, 5)
                and self.has_recipe("fireplace")):
            return A_BUILD

        # build house if homeless and has blocks
        if self.home_y is None and self.inv_count("wooden_block") >= 4:
            return A_BUILD

        # cook bread if wheat + near fireplace
        if (self.inv_count("wheat") >= 2 and
                world.near_structure(self.y, self.x, S_FIREPLACE, 3)):
            return A_CRAFT_TOOL  # reuse slot — cooking

        # reproduce
        if self._can_reproduce(world, agents):
            return A_SEEK_MATE

        # autumn: forage to build stockpile
        if season == "autumn":
            return A_FORAGE

        return A_WANDER

    def _baby_decide(self, world, agents):
        if world.fire_intensity[int(self.y), int(self.x)] > 0: return A_FLEE_FIRE
        if self.thirst < 30: return A_FIND_WATER
        if self.hunger < 30: return A_FIND_FOOD
        if self.warmth < 40: return A_SEEK_WARMTH
        return A_WANDER

    def _can_reproduce(self, world, agents):
        if world.season_name != "summer": return False
        if self.stage != "adult": return False
        if self.preg_timer is not None: return False
        if self.gender == "female":
            if self.mood < 60: return False
            if self.hunger < 50: return False
            if self.total_food() < 5: return False
        return True

    # ═══════════════════════════════════════════════════════
    #  Tick  (called every frame, sim_dt scaled)
    # ═══════════════════════════════════════════════════════

    def tick(self, world, agents, real_dt):
        if self.is_dead: return
        sim_dt = real_dt * SIM_SPEED

        # ── age / stage ───────────────────────────────────
        self.age += sim_dt
        prev_winters = self.winters
        self.winters = int(self.age / (DAY_DURATION * SEASON_DAYS * 4))
        if self.winters > prev_winters:
            self._on_winter_tick(world)

        # ── spoilage ─────────────────────────────────────
        for itype, idata in list(self.inventory.items()):
            if idata["freshness"] is not None:
                idata["freshness"] -= sim_dt * (100.0 / (DAY_DURATION * 5))  # 5 day base
                if idata["freshness"] <= 0:
                    del self.inventory[itype]

        # ── vitals decay ─────────────────────────────────
        weather = world.weather
        season  = world.season_name
        wm  = world.WEATHER_MOOD_MULT.get(weather, 1.0)
        whm = world.WEATHER_HUNGER_MULT.get(weather, 1.0)
        www = world.WEATHER_WARMTH_MULT.get(weather, 1.0)
        scm = world.SEASON_COLD_MULT.get(season, 1.0)

        baby_mod = 0.7 if self.stage == "baby" else 1.0
        elder_sp = 0.7 if self.stage == "elder" else 1.0

        self.hunger = max(0, self.hunger - HUNGER_DECAY * whm * sim_dt * baby_mod * 60)
        self.thirst = max(0, self.thirst - THIRST_DECAY * sim_dt * 60)
        self.mood   = max(0, self.mood   - MOOD_DECAY * wm * sim_dt * 60)

        # warmth
        near_fp = world.near_structure(self.y, self.x, S_FIREPLACE, 3)
        roof    = world.inside_house(self.y, self.x)
        warmth_loss = WARMTH_DECAY * www * scm * sim_dt * 60
        if near_fp:  warmth_loss -= 3.0 * sim_dt
        if roof:     warmth_loss *= 0.5
        self.warmth = max(0, min(100, self.warmth - warmth_loss))

        # damage from zero stats
        if self.warmth <= 0:  self.health -= 8  * sim_dt
        if self.hunger <= 0:  self.health -= 5  * sim_dt
        if self.thirst <= 0:  self.health -= 15 * sim_dt
        if self.stage == "elder": self.health -= 0.5 * sim_dt
        if self.health <= 0:
            self.is_dead = True
            world.drop_items(int(self.y), int(self.x), list(self.inventory.values()))
            self.inventory = {}
            return

        # fire damage
        if world.fire_intensity[int(self.y), int(self.x)] > 0:
            self.health -= 20 * sim_dt
            self.alert_fire = True
            self.alert_timer = 10.0

        # alert decay
        if self.alert_timer > 0:
            self.alert_timer -= sim_dt
        else:
            self.alert_fire = False

        # ── mood boosts ───────────────────────────────────
        if self.hunger > 80: self.mood = min(100, self.mood + 0.1 * sim_dt)
        if near_fp:          self.mood = min(100, self.mood + 0.2 * sim_dt)
        if roof:             self.mood = min(100, self.mood + 0.1 * sim_dt)

        # ── action execution ──────────────────────────────
        prev_v = self.prev_vitals
        action = self.decide(world, agents)

        speed = 5.0 * elder_sp * (0.7 if not world.is_day else 1.0)
        self._execute(action, world, agents, sim_dt, speed)

        # ── online learning ───────────────────────────────
        reward = self._calc_reward()
        if abs(reward) > 0.005:
            self.record_event(action, reward)
        self.prev_vitals = (self.hunger, self.thirst, self.warmth, self.mood)
        self.current_action = action

    def _on_winter_tick(self, world):
        if self.winters >= 15:
            self.is_dead = True
            return
        if self.winters >= 13:
            self.stage = "elder"
        elif self.winters >= 3:
            self.stage = "adult"
        else:
            self.stage = "baby"

    # ═══════════════════════════════════════════════════════
    #  Execute action
    # ═══════════════════════════════════════════════════════

    def _execute(self, action, world, agents, sim_dt, speed):
        if action == A_FLEE_FIRE:
            self._flee_fire(world, sim_dt, speed, all_agents=agents)
        elif action in (A_DRINK, A_FIND_WATER):
            self._seek_water(world, sim_dt, speed)
        elif action in (A_FIND_FOOD, A_FORAGE, A_STOCKPILE):
            self._seek_food(world, sim_dt, speed, stockpile=(action == A_STOCKPILE))
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

        # pick up dropped items while passing
        self._loot_ground(world)

    # ── movement helper ───────────────────────────────────
    def _move_toward(self, ty, tx, world, sim_dt, speed):
        dy = ty - self.y
        dx = tx - self.x
        dist = math.sqrt(dy*dy + dx*dx)
        if dist < 0.5:
            return True  # arrived
        step = speed * sim_dt
        self.y += (dy / dist) * min(step, dist)
        self.x += (dx / dist) * min(step, dist)
        # clamp
        self.y = max(0, min(WORLD_H - 1, self.y))
        self.x = max(0, min(WORLD_W - 1, self.x))
        return False

    def _flee_fire(self, world, sim_dt, speed, all_agents=None):
        iy, ix = int(self.y), int(self.x)
        # move away from nearest fire
        if world.fire_intensity[iy, ix] > 0:
            # find direction away
            best_y = self.y - 5 * random.choice([-1, 1])
            best_x = self.x - 5 * random.choice([-1, 1])
            best_y = max(0, min(WORLD_H-1, best_y))
            best_x = max(0, min(WORLD_W-1, best_x))
            self._move_toward(best_y, best_x, world, sim_dt, speed * 2.0)
            self.fires_escaped += 1
        # alert nearby agents
        if all_agents:
            for other in all_agents:
                if other is not self and abs(other.x - self.x) + abs(other.y - self.y) < 8:
                    other.alert_fire = True
                    other.alert_timer = max(other.alert_timer, 8.0)

    def _seek_water(self, world, sim_dt, speed):
        if world.near_river(self.y, self.x, 1):
            self.thirst = min(100, self.thirst + 40)
            self.record_event(A_DRINK, 1.0)
        else:
            # use memory or find
            river = world.find_nearest_river(self.y, self.x)
            if river:
                arrived = self._move_toward(river[0], river[1], world, sim_dt, speed)
                if arrived:
                    self.memory["river"].append(river)
                    if len(self.memory["river"]) > 5:
                        self.memory["river"].pop(0)

    def _seek_food(self, world, sim_dt, speed, stockpile=False):
        # eat from inventory first
        eaten = False
        for ftype in ("bread", "berry", "wheat"):
            if self.stage == "baby" and ftype == "wheat":
                continue
            if self.inv_count(ftype) > 0 and self.hunger < 90:
                val = ITEM_FOOD_VALUE.get(ftype, 0)
                self.hunger = min(100, self.hunger + val)
                self.inv_remove(ftype)
                self.food_eaten += 1
                self.record_event(A_FIND_FOOD, 1.0)
                eaten = True
                break

        if not eaten or stockpile:
            # forage for berries first, then wheat
            berry_pos = world.find_nearest(self.y, self.x, O_BERRY)
            wheat_pos = world.find_nearest(self.y, self.x, O_WHEAT)
            rock_pos  = world.find_nearest(self.y, self.x, O_ROCK_OBJ)
            target = berry_pos or wheat_pos

            # also pick up rocks while foraging
            if rock_pos and self.inv_count("rock") < 4:
                if abs(rock_pos[0]-self.y) + abs(rock_pos[1]-self.x) < 3:
                    world.remove_object(rock_pos[0], rock_pos[1])
                    self.inv_add("rock", 1)

            if target:
                arrived = self._move_toward(target[0], target[1], world, sim_dt, speed)
                if arrived:
                    item = world.harvest(target[0], target[1])
                    if item:
                        self.inv_add(item["type"], item["quantity"], item.get("freshness"))
                        # update memory
                        obj_t = O_BERRY if item["type"] == "berry" else O_WHEAT
                        if target not in self.memory.get(obj_t, []):
                            self.memory.setdefault(obj_t, []).append(target)
                        self.record_event(A_FORAGE, 0.5)
            else:
                # go cut trees for wood (so we can craft)
                tree = world.find_nearest(self.y, self.x, O_TREE)
                if tree:
                    arrived = self._move_toward(tree[0], tree[1], world, sim_dt, speed)
                    if arrived:
                        item = world.harvest(tree[0], tree[1])
                        if item:
                            self.inv_add(item["type"], item["quantity"])
                else:
                    self._wander(world, sim_dt, speed)

    def _seek_warmth(self, world, sim_dt, speed):
        # try to get near a fireplace
        if world.near_structure(self.y, self.x, S_FIREPLACE, 3):
            return  # already warm
        # find nearest fireplace
        H, W = WORLD_H, WORLD_W
        best_d = 1e9
        best = None
        for ty in range(H):
            for tx in range(W):
                if world.structures[ty, tx] == S_FIREPLACE:
                    d = abs(ty-self.y) + abs(tx-self.x)
                    if d < best_d:
                        best_d = d
                        best = (ty, tx)
        if best:
            self._move_toward(best[0]+1, best[1], world, sim_dt, speed)
        elif self.has_recipe("fireplace"):
            self._build(world, sim_dt)
        else:
            # go get materials
            self._seek_food(world, sim_dt, speed)

    def _craft(self, world, sim_dt):
        if self.craft_remaining > 0:
            self.craft_remaining -= sim_dt
            if self.craft_remaining <= 0:
                # complete craft
                item = self.crafting_item
                if item:
                    self.inv_add(item)
                    self.crafting_item = None
                    self.record_event(A_CRAFT_TOOL, 1.0)
            return

        # decide what to craft
        # priority: sharpened_stone → axe/hoe → fireplace → wooden_block → bread
        for recipe in ["sharpened_stone", "axe", "hoe", "fireplace", "wooden_block", "bread"]:
            if recipe == "axe" and self.has_tool("axe"): continue
            if recipe == "hoe" and self.has_tool("hoe"): continue
            if recipe == "bread" and not world.near_structure(self.y, self.x, S_FIREPLACE, 3): continue
            if recipe == "bread" and self.inv_count("wheat") < 2: continue
            if self.has_recipe(recipe):
                self.consume_recipe(recipe)
                self.crafting_item   = recipe
                self.craft_remaining = CRAFT_TIME[recipe]
                break

    def _build(self, world, sim_dt):
        # Build fireplace first if needed, then house
        if self.has_recipe("fireplace") and not world.near_structure(self.y, self.x, S_FIREPLACE, 6):
            spot = world.find_empty_ground(self.y, self.x, 4)
            if spot:
                # safety distance from wood structures
                too_close = world.near_structure(spot[0], spot[1], S_HOUSE, 2)
                if not too_close:
                    arrived = self._move_toward(spot[0], spot[1], world, sim_dt, 5.0)
                    if arrived:
                        self.consume_recipe("fireplace")
                        world.place_structure(spot[0], spot[1], S_FIREPLACE)
                        self.record_event(A_BUILD, 1.0)
        elif self.inv_count("wooden_block") >= 4 and self.home_y is None:
            spot = world.find_empty_ground(self.y, self.x, 6)
            if spot:
                arrived = self._move_toward(spot[0], spot[1], world, sim_dt, 5.0)
                if arrived:
                    # place house
                    for dy2 in range(-1, 2):
                        for dx2 in range(-1, 2):
                            ty2, tx2 = spot[0]+dy2, spot[1]+dx2
                            if 0 <= ty2 < WORLD_H and 0 <= tx2 < WORLD_W:
                                if world.structures[ty2, tx2] == S_NONE:
                                    world.place_structure(ty2, tx2, S_HOUSE)
                    self.inv_remove("wooden_block", min(4, self.inv_count("wooden_block")))
                    self.home_y = spot[0]
                    self.home_x = spot[1]
                    self.record_event(A_BUILD, 1.5)
        elif self.inv_count("wood") >= 2:
            # craft wooden blocks
            if self.has_recipe("wooden_block"):
                self.consume_recipe("wooden_block")
                self.inv_add("wooden_block")
        else:
            self._wander(world, sim_dt, 5.0)

    def _seek_mate(self, world, agents, sim_dt, speed):
        # find eligible opposite-gender adult
        best = None
        best_d = 1e9
        for ag in agents:
            if ag is self or ag.is_dead: continue
            if ag.gender == self.gender: continue
            if ag.stage != "adult": continue
            if ag.partner_id is not None: continue
            d = abs(ag.x - self.x) + abs(ag.y - self.y)
            if d < best_d:
                best_d = d
                best = ag

        if best is None:
            self._wander(world, sim_dt, speed)
            return

        arrived = self._move_toward(best.y, best.x, world, sim_dt, speed)
        if arrived and best_d < 4:
            # reproduce
            if self.gender == "female" and self.preg_timer is None:
                self.preg_timer   = 30.0  # 30 sim-seconds pregnancy
                self.partner_id   = best.id
                best.partner_id   = self.id
                self.children_born += 1
                self.record_event(A_SEEK_MATE, 1.0)

    def _seek_safety(self, world, sim_dt, speed):
        # go home or find shelter
        if self.home_y is not None:
            self._move_toward(self.home_y, self.home_x, world, sim_dt, speed * 1.2)
        else:
            self._wander(world, sim_dt, speed)

    def _wander(self, world, sim_dt, speed):
        # if no target or reached target, pick new one
        if self.target is None or (abs(self.x - self.target[1]) < 1 and abs(self.y - self.target[0]) < 1):
            home_y = self.home_y or WORLD_H / 2
            home_x = self.home_x or WORLD_W / 2
            r = 15 if not world.is_day else 25
            ty = max(0, min(WORLD_H-1, home_y + random.uniform(-r, r)))
            tx = max(0, min(WORLD_W-1, home_x + random.uniform(-r, r)))
            self.target = (ty, tx)
        self._move_toward(self.target[0], self.target[1], world, sim_dt, speed * 0.6)

    def _loot_ground(self, world):
        items = world.pick_dropped(int(self.y), int(self.x))
        for item_dict in items:
            if self.inv_total_slots() < self.inv_limit:
                self.inv_add(item_dict.get("type", "rock"),
                              item_dict.get("quantity", 1),
                              item_dict.get("freshness"))

    # ── pregnancy ─────────────────────────────────────────
    def tick_pregnancy(self, world, agents, sim_dt):
        if self.preg_timer is None: return None
        self.preg_timer -= sim_dt
        if self.preg_timer <= 0:
            self.preg_timer = None
            # give birth
            child = Agent(self.x + random.uniform(-1, 1),
                          self.y + random.uniform(-1, 1),
                          world=world)
            child.stage = "baby"
            child.winters = 0
            child.age = 0.0
            child.generation = self.generation + 1
            # inherit knowledge
            child.memory = {}
            for k, v in self.memory.items():
                child.memory[k] = v[:int(len(v) * KNOWLEDGE_INHERIT)]
            # inherit + blend weights from parents
            partner = next((a for a in agents if a.id == self.partner_id), None)
            if partner:
                child.weights = [
                    max(0.3, min(3.0,
                        (self.weights[i] * 0.5 + partner.weights[i] * 0.5)
                        + random.gauss(0, MUTATION_RATE)
                    ))
                    for i in range(NUM_ACTIONS)
                ]
            else:
                child.weights = [
                    max(0.3, min(3.0, w + random.gauss(0, MUTATION_RATE)))
                    for w in self.weights
                ]
            # start with some food from mother
            if self.inv_count("berry") > 1:
                self.inv_remove("berry", 1)
                child.inv_add("berry", 1, freshness=100.0)
            self.children_ids.append(child.id)
            return child
        return None
