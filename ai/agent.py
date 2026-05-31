"""
Agent AI — behaviour tree, learning weights, genetic inheritance.

v4 fixes:
  * FIX A — Stage regression (day 18 everyone becomes baby):
             _on_winter_tick used age-derived winters which caused adult→baby
             regression. Now stage only ever advances (baby→adult→elder),
             never goes backward, and winters accumulates monotonically.
  * FIX B — Recreation gate: lowered thresholds + recreation stays active as
             long as vitals hold. No longer requires simultaneous peak in
             ALL vitals.
  * FIX C — Buildings: simplified build system — houses cost wood directly
             (no wooden_block intermediary), farms require hoe + wood,
             build utility thresholds lowered so agents actually build.
  * FIX D — Exploration: randomized at spawn, not fixed 0.30.
  * FIX E — Crafting: updated recipes for new farm / house system.
"""

import math
import random

import engine.config as cfg
from ai.actions import (
    ACTION_NAMES, NUM_ACTIONS,
    A_FLEE_FIRE, A_DRINK, A_SEEK_SAFETY, A_FIND_WATER, A_FIND_FOOD,
    A_SEEK_WARMTH, A_FORAGE, A_STOCKPILE, A_CRAFT_TOOL, A_BUILD,
    A_SEEK_MATE, A_WANDER, A_SLEEP, A_RECREATE,
)
from engine.config import *
from world.world import *


ITEM_FOOD_VALUE = {
    "berry":       15,
    "wheat":        8,
    "bread":       40,
    "cooked_meat": 50,
}

# Updated craft recipes — no more wooden_block; houses built directly with wood
CRAFT_RECIPES = {
    "sharpened_stone": {"rock": 2},
    "axe":             {"sharpened_stone": 1, "wood": 1},
    "hoe":             {"sharpened_stone": 1, "wood": 1},
    "shovel":          {"sharpened_stone": 1, "wood": 1},
    "fireplace":       {"rock": 4, "wood": 2},
    "bread":           {"wheat": 2},
    # House is now "build" directly — see _build()
    # Farm needs hoe + wood in _build()
}

CRAFT_TIME = {
    "sharpened_stone": 3.0,
    "axe":   5.0,
    "hoe":   5.0,
    "shovel": 5.0,
    "fireplace": 10.0,
    "bread": 10.0,
}

TOOL_DURABILITY = 100
_agent_id_counter = [0]

SLEEP_NIGHT_MOOD_THRESH = 40.0
SLEEP_MIN_DURATION    = 15.0
SLEEP_MAX_DURATION    = 45.0
SLEEP_WARMTH_RESTORE  = 2.0
SLEEP_MOOD_RESTORE    = 1.5
SLEEP_HEALTH_RESTORE  = 0.3

RECREATION_FOOD_GATE  = 15      # FIX B — lowered
RECREATION_VITAL_GATE = 55.0   # FIX B — lowered

HEART_EFFECT_DURATION = 3.0


def new_id():
    _agent_id_counter[0] += 1
    return f"A{_agent_id_counter[0]:04d}"


def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


class Agent:
    def __init__(self, x, y, gender=None, parent_weights=None, world=None, hidden_size=None):
        self.id     = new_id()
        self.gender = gender or random.choice(["male", "female"])
        self.age    = 0.0
        self.winters = 0
        self.stage  = "adult"

        self.hunger = random.uniform(60, 90)
        self.thirst = random.uniform(60, 90)
        self.warmth = random.uniform(70, 90)
        self.mood   = random.uniform(60, 90)
        self.health = 100.0

        self.x = float(x)
        self.y = float(y)
        self.path   = []
        self.target = None

        self.current_action = A_WANDER
        self.action_timer   = 0.0
        self.action_target  = None
        self.crafting_item    = None
        self.craft_remaining  = 0.0
        self.is_dead = False

        self.inventory = {}
        self.inv_limit = 14

        self.memory = {
            O_TREE:   [],
            O_BERRY:  [],
            O_WHEAT:  [],
            "river":  [],
            "home":   [],
        }
        self.home_y = None
        self.home_x = None

        self.partner_id     = None
        self.children_ids   = []
        self.parent_ids     = []
        self.preg_timer     = None
        self.postpartum_timer = 0.0

        self.drink_cooldown = 0.0
        self.grace_timer    = 0.0

        self.sleeping       = False
        self.sleep_timer    = 0.0
        self.wake_reason    = None
        self._sleep_cooldown = 0.0

        self.heart_effect_timer = 0.0

        self.friends       = {}
        self.nemeses       = {}
        self._fight_timers = {}

        self.alert_fire  = False
        self.alert_timer = 0.0

        if parent_weights is not None:
            self.weights = [
                max(0.3, min(3.0, parent_weights[i] + random.gauss(0, MUTATION_RATE)))
                for i in range(NUM_ACTIONS)
            ]
        else:
            self.weights = [1.0] * NUM_ACTIONS

        # FIX D — randomize exploration at spawn
        self.exploration  = random.uniform(0.15, 0.65)
        self.visit_heat   = {}
        self._heat_timer  = 0.0

        self.event_log  = []
        self.prev_vitals = (self.hunger, self.thirst, self.warmth, self.mood)

        self.food_eaten    = 0
        self.children_born = 0
        self.fires_escaped = 0
        self.generation    = 0

    # ── Inventory helpers ─────────────────────────────────────

    def inv_count(self, itype):
        return self.inventory.get(itype, {}).get("quantity", 0)

    def inv_total_slots(self):
        return sum(item["quantity"] for item in self.inventory.values())

    def inv_add(self, itype, qty=1, freshness=None, durability=None):
        if qty <= 0:
            return False
        item = self.inventory.setdefault(
            itype,
            {"quantity": 0, "freshness": freshness,
             "durability": durability if durability is not None else TOOL_DURABILITY},
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
        if action_id is None or action_id >= len(self.weights):
            return
        self.weights[action_id] = max(
            0.3, min(3.0, self.weights[action_id] + LEARNING_RATE * reward),
        )
        self.event_log.append((action_id, reward))
        if len(self.event_log) > 20:
            self.event_log.pop(0)

    def _calc_reward(self):
        h, t, w, m = self.hunger, self.thirst, self.warmth, self.mood
        ph, pt, pw, pm = self.prev_vitals
        reward = (h-ph)*0.4 + (t-pt)*0.4 + (w-pw)*0.1 + (m-pm)*0.1
        if min(h, t, w, m) < 20.0:
            reward -= 0.2
        self.prev_vitals = (h, t, w, m)
        return reward

    # ── Relationship helpers ──────────────────────────────────

    def is_family(self, other_id):
        return other_id in self.parent_ids or other_id in self.children_ids

    def is_nemesis(self, other_id):
        return self.nemeses.get(other_id, False)

    def friendship_score(self, other_id):
        return self.friends.get(other_id, 0.0)

    def gain_friendship(self, other_id, amount):
        if self.is_family(other_id):
            return
        if other_id not in self.friends and len(self.friends) >= MAX_FRIENDS:
            if amount >= FRIEND_NEAR_GAIN * 10:
                weakest = min(self.friends, key=lambda k: self.friends[k])
                if self.friends[weakest] < self.friends.get(other_id, 0) + amount:
                    del self.friends[weakest]
                else:
                    return
            else:
                return
        current = self.friends.get(other_id, 0.0)
        self.friends[other_id] = min(100.0, current + amount)

    def decay_friendships(self):
        for fid in list(self.friends.keys()):
            if random.random() < FRIEND_FORGET_CHANCE:
                self.friends[fid] -= FRIEND_FORGET_AMOUNT
                if self.friends[fid] <= 0:
                    del self.friends[fid]

    def make_nemesis(self, other):
        self.nemeses[other.id] = True
        other.nemeses[self.id] = True
        if other.id in self.friends:
            self.friends[other.id] = max(0.0, self.friends[other.id] - 10.0)
        if self.id in other.friends:
            other.friends[self.id] = max(0.0, other.friends[self.id] - 10.0)

    def _update_social(self, agents, sim_dt):
        self.decay_friendships()
        for other in agents:
            if other is self or other.is_dead:
                continue
            dist = abs(other.x - self.x) + abs(other.y - self.y)
            if dist > 8.0:
                continue
            oid = other.id
            if not self.is_nemesis(oid):
                self.gain_friendship(oid, FRIEND_NEAR_GAIN * sim_dt)
                other.gain_friendship(self.id, FRIEND_NEAR_GAIN * sim_dt)
            if (self.gender == "male" and other.gender == "male"
                    and self.stage == "adult" and other.stage == "adult"
                    and dist < 2.0 and not self.is_family(oid)
                    and random.random() < NEMESIS_CHANCE * sim_dt):
                self.make_nemesis(other)
            if self.is_nemesis(oid) and dist < 2.0:
                cd = self._fight_timers.get(oid, 0.0)
                if cd <= 0 and random.random() < NEMESIS_FIGHT_CHANCE * sim_dt:
                    self._do_fight(other, agents)
                    self._fight_timers[oid] = 5.0
            if oid in self._fight_timers:
                self._fight_timers[oid] = max(0.0, self._fight_timers[oid] - sim_dt)
            if (not self.is_nemesis(oid)
                    and (self.is_family(oid) or self.friendship_score(oid) > FRIEND_THRESHOLD)
                    and other.hunger < 30 and self.total_food() > REPRO_MIN_FOOD
                    and dist < 3.0 and random.random() < 0.002 * sim_dt):
                self._share_food(other)
            if (not self.is_nemesis(oid)
                    and self.friendship_score(oid) > FRIEND_THRESHOLD
                    and 4.0 < dist < 20.0
                    and self.current_action == A_WANDER
                    and random.random() < 0.05 * sim_dt):
                self.target = (other.y + random.uniform(-2, 2),
                               other.x + random.uniform(-2, 2))

    def _do_fight(self, other, agents):
        self.health  = max(1.0, self.health  - FIGHT_DAMAGE)
        other.health = max(1.0, other.health - FIGHT_DAMAGE)
        self.mood    = max(0.0, self.mood    - FIGHT_MOOD_LOSS)
        other.mood   = max(0.0, other.mood   - FIGHT_MOOD_LOSS)
        for ag in agents:
            if ag is self or ag is other or ag.is_dead:
                continue
            ds = abs(ag.x - self.x) + abs(ag.y - self.y)
            do = abs(ag.x - other.x) + abs(ag.y - other.y)
            if ds > 6 and do > 6:
                continue
            if random.random() >= FIGHT_JOIN_CHANCE:
                continue
            if ag.friendship_score(self.id) > FRIEND_THRESHOLD or self.is_family(ag.id):
                other.health = max(1.0, other.health - FIGHT_DAMAGE * 0.5)
            elif ag.friendship_score(other.id) > FRIEND_THRESHOLD or other.is_family(ag.id):
                self.health = max(1.0, self.health - FIGHT_DAMAGE * 0.5)

    def _share_food(self, other):
        for ftype in ("berry", "bread", "wheat"):
            if self.inv_count(ftype) > 1:
                self.inv_remove(ftype, 1)
                other.inv_add(ftype, 1, freshness=80.0)
                other.hunger = min(100, other.hunger + ITEM_FOOD_VALUE.get(ftype, 10))
                self.gain_friendship(other.id, FRIEND_SHARE_GAIN)
                other.gain_friendship(self.id, FRIEND_SHARE_GAIN)
                break

    # ── Decision making ───────────────────────────────────────

    def _should_sleep(self, world):
        if self.stage == "baby":   return False
        if self.sleeping:          return False
        if self._sleep_cooldown > 0: return False
        if self.alert_fire:        return False
        if min(self.hunger, self.thirst) < 20: return False
        if not world.is_day and self.mood < SLEEP_NIGHT_MOOD_THRESH:
            return True
        if not world.is_day and self.mood < 70 and random.random() < 0.005:
            return True
        return False

    def _should_recreate(self):
        """
        FIX B — Relaxed recreation gate. Agents recreate when they have
        enough food AND no vital is critically low. Doesn't require ALL
        vitals above a high bar simultaneously.
        """
        if self.stage == "baby":
            return False
        if self.total_food() < RECREATION_FOOD_GATE:
            return False
        # At least one critical vital must not be at crisis level
        if min(self.hunger, self.thirst) < 35:
            return False
        # Don't recreate if warmth is dangerously low
        if self.warmth < 30:
            return False
        # FIX B: Use softer threshold — majority of vitals above gate, not all
        vitals = [self.hunger, self.thirst, self.warmth]
        above = sum(1 for v in vitals if v >= RECREATION_VITAL_GATE)
        return above >= 2   # 2 out of 3 vitals must be OK

    def decide(self, world, agents):
        if self.stage == "baby":
            return self._baby_decide(world, agents)

        if self.sleeping:
            urgent = (self.alert_fire or self.hunger < 20 or self.thirst < 15
                      or self.warmth < 25 or self.health < 20)
            if urgent or self.sleep_timer <= 0:
                self.sleeping    = False
                self.wake_reason = "urgent" if urgent else "rested"
                self._sleep_cooldown = 10.0
            else:
                return A_SLEEP

        # Emergency: fire
        w = list(self.weights)
        iy, ix = int(self.y), int(self.x)
        if 0 <= iy < WORLD_H and 0 <= ix < WORLD_W:
            if world.fire_intensity[iy, ix] > 0:
                return A_FLEE_FIRE
        if self.alert_fire:
            return A_FLEE_FIRE

        # Thirst
        if self.drink_cooldown <= 0:
            if self.thirst < 15:  return A_DRINK
            if self.thirst < DRINK_THRESHOLD: return A_FIND_WATER

        if self.health < 25:    return A_SEEK_SAFETY
        if self.hunger < 20:    return A_FIND_FOOD
        if self.warmth < 40:    return A_SEEK_WARMTH

        # FIX B: Recreation — keep the gate active if vitals hold
        if self._should_recreate():
            w[A_FORAGE]    = 0.0
            w[A_STOCKPILE] = 0.0
            w[A_RECREATE]  = 2.5
            w[A_SEEK_MATE] = 1.5 if self._can_reproduce(world, agents) else 0.0
            w[A_BUILD]    += self._build_utility(world, agents)
            w[A_CRAFT_TOOL] += 0.5 if not self.has_tool("axe") else 0.0
            w[A_WANDER]   += 1.0
            return max(range(NUM_ACTIONS), key=lambda i: w[i])

        if self.hunger < 50:
            return A_FIND_FOOD

        season = world.season_name
        if season in ("winter", "autumn") and self.total_food() < 80:
            w[A_STOCKPILE] += 1.0
        if not self.has_tool("axe") and (
            self.inv_count("rock") >= 2 or self.has_recipe("axe")
        ):
            w[A_CRAFT_TOOL] += 0.8

        w[A_BUILD] += self._build_utility(world, agents)

        if not world.near_structure(self.y, self.x, S_FIREPLACE, 6):
            if self.has_recipe("fireplace"):
                w[A_BUILD] += 1.5   # stronger push to build fireplace
            elif self.inv_count("rock") >= 4 and self.inv_count("wood") >= 2:
                w[A_CRAFT_TOOL] += 0.6

        if self._can_reproduce(world, agents):
            w[A_SEEK_MATE] += 1.5

        if self._should_sleep(world):
            self.sleeping    = True
            self.sleep_timer = random.uniform(SLEEP_MIN_DURATION, SLEEP_MAX_DURATION)
            self.wake_reason = None
            return A_SLEEP

        return max(range(NUM_ACTIONS), key=lambda i: w[i])

    def _build_utility(self, world, agents):
        """
        FIX C — Simplified, lower-threshold build utility.
        No longer requires ALL vitals at BUILD_PREFER_THRESH simultaneously.
        """
        bonus = 0.0

        # Relaxed: at least hunger and thirst OK, warmth flexible
        if self.hunger < BUILD_PREFER_THRESH or self.thirst < BUILD_PREFER_THRESH:
            return 0.0

        if self.total_food() > BUILD_FOOD_SURPLUS:
            bonus += 0.8

        for other in agents:
            if other is self or other.is_dead:
                continue
            if self.is_family(other.id) or self.friendship_score(other.id) > FRIEND_THRESHOLD:
                dist = abs(other.x - self.x) + abs(other.y - self.y)
                if dist < 10:
                    bonus += 0.3
                    break

        # FIX C: Heavily encourage building a house (no wooden_block step)
        if self.home_y is None and self.inv_count("wood") >= HOUSE_WOOD_COST:
            bonus += 2.0
        elif self.home_y is None:
            bonus += 0.3   # still wander toward wood

        # Encourage farm building if has hoe + wood
        if (self.has_tool("hoe")
                and self.inv_count("wood") >= FARM_WOOD_COST
                and not world.near_structure(self.y, self.x, S_FARM, 8)):
            bonus += 1.5

        season = world.season_name
        if season in ("autumn", "winter"):
            bonus += 0.6

        if not self.has_tool("axe"):
            bonus -= 0.2

        return bonus

    def _baby_decide(self, world, agents):
        iy, ix = int(self.y), int(self.x)
        if 0 <= iy < WORLD_H and 0 <= ix < WORLD_W and world.fire_intensity[iy, ix] > 0:
            return A_FLEE_FIRE
        if self.drink_cooldown <= 0 and self.thirst < 50:
            return A_FIND_WATER
        if self.hunger < 40:
            return A_FIND_FOOD
        if self.warmth < 40:
            return A_SEEK_WARMTH
        return A_WANDER

    def _can_reproduce(self, world, agents):
        if self.stage != "adult":          return False
        if self.preg_timer is not None:    return False
        if self.postpartum_timer > 0:      return False
        if self.gender == "male":          return False
        if self.hunger < REPRO_MIN_HUNGER: return False
        if self.mood < REPRO_MIN_MOOD:     return False
        if self.total_food() < REPRO_MIN_FOOD: return False
        if REPRO_NEED_SHELTER and self.home_y is None: return False
        pop = len([a for a in agents if not a.is_dead])
        return pop < MAX_AGENTS

    # ── Tick ─────────────────────────────────────────────────

    def tick(self, world, agents, real_dt):
        if self.is_dead:
            return

        sim_dt = real_dt * getattr(cfg, "SIM_SPEED", SIM_SPEED)
        self.age += sim_dt

        # FIX A — winters is a simple counter, never goes backward.
        # Stage only advances: baby→adult→elder. Never regresses.
        new_winters = int(self.age / max(1.0, DAY_DURATION * SEASON_DAYS * 4))
        if new_winters > self.winters:
            self.winters = new_winters
            self._on_winter_tick()

        if self.grace_timer > 0:
            self.grace_timer = max(0.0, self.grace_timer - sim_dt)
        if self.postpartum_timer > 0:
            self.postpartum_timer = max(0.0, self.postpartum_timer - sim_dt)
        if self.drink_cooldown > 0:
            self.drink_cooldown = max(0.0, self.drink_cooldown - sim_dt)
        if self._sleep_cooldown > 0:
            self._sleep_cooldown = max(0.0, self._sleep_cooldown - sim_dt)
        if self.heart_effect_timer > 0:
            self.heart_effect_timer = max(0.0, self.heart_effect_timer - real_dt)

        for itype, idata in list(self.inventory.items()):
            if idata.get("freshness") is not None:
                idata["freshness"] = max(0.0, idata["freshness"] - sim_dt * 0.05)
                if idata["freshness"] <= 0:
                    self.inv_remove(itype, idata["quantity"])

        weather = world.weather
        season  = world.season_name
        wm  = world.WEATHER_MOOD_MULT.get(weather, 1.0)
        whm = world.WEATHER_HUNGER_MULT.get(weather, 1.0)
        www = world.WEATHER_WARMTH_MULT.get(weather, 1.0)
        scm = world.SEASON_COLD_MULT.get(season, 1.0)

        in_grace = self.grace_timer > 0
        if self.stage == "baby":
            hunger_mod = BABY_HUNGER_DECAY * (0.4 if in_grace else 1.0)
            thirst_mod = BABY_THIRST_DECAY * (0.4 if in_grace else 1.0)
        else:
            hunger_mod = 1.0
            thirst_mod = 1.0

        sleep_mod  = 0.1 if self.sleeping else 1.0
        elder_sp   = 1.3 if self.stage == "elder" else 1.0
        near_fp    = world.near_structure(self.y, self.x, S_FIREPLACE, 3)
        roof       = world.inside_house(self.y, self.x)

        self.hunger = max(0, self.hunger - HUNGER_DECAY * whm * sim_dt * hunger_mod * sleep_mod)
        self.thirst = max(0, self.thirst - THIRST_DECAY * sim_dt * thirst_mod * sleep_mod)
        self.mood   = max(0, self.mood   - MOOD_DECAY   * wm  * sim_dt * sleep_mod)

        mood_friends = sum(
            1 for a in agents
            if not a.is_dead and a is not self
            and abs(a.x - self.x) + abs(a.y - self.y) < 6
            and (self.is_family(a.id) or self.friendship_score(a.id) > FRIEND_THRESHOLD)
        )
        if mood_friends > 0:
            self.mood = min(100, self.mood + 0.3 * mood_friends * sim_dt)

        warmth_loss = WARMTH_DECAY * www * scm * sim_dt * elder_sp * sleep_mod
        if near_fp:
            warmth_loss *= 0.3
            self.warmth = min(100, self.warmth + 3.0 * sim_dt)
        if roof:
            warmth_loss *= 0.5
            self.mood = min(100, self.mood + 0.5 * sim_dt)
        if in_grace:
            warmth_loss *= 0.5
        self.warmth = max(0, self.warmth - warmth_loss)

        if self.sleeping:
            self.warmth = min(100, self.warmth + SLEEP_WARMTH_RESTORE * sim_dt)
            self.mood   = min(100, self.mood   + SLEEP_MOOD_RESTORE   * sim_dt)
            self.health = min(100, self.health + SLEEP_HEALTH_RESTORE * sim_dt)
            self.sleep_timer = max(0.0, self.sleep_timer - sim_dt)

        if min(self.hunger, self.thirst, self.warmth, self.mood) <= 0:
            if in_grace:
                self.health -= 1.0 * sim_dt
            else:
                self.health -= 8 * sim_dt
        elif min(self.hunger, self.thirst, self.warmth) < 15:
            self.health -= 2 * sim_dt
        else:
            self.health = min(100, self.health + 0.4 * sim_dt)

        if self.health <= 0:
            self.is_dead = True
            drops = [{"type": itype, **idata}
                     for itype, idata in self.inventory.items()
                     if idata.get("quantity", 0) > 0]
            if drops:
                world.drop_items(self.y, self.x, drops)
            return

        if world.near_fire(self.y, self.x, 5):
            self.alert_fire  = True
            self.alert_timer = 10.0
            if self.sleeping:
                self.sleeping = False
                self.wake_reason = "fire"
                self._sleep_cooldown = 30.0
        elif self.alert_timer > 0:
            self.alert_timer -= sim_dt
        else:
            self.alert_fire = False

        self._heat_timer += sim_dt
        if self._heat_timer >= 2.0:
            self._heat_timer = 0.0
            key = (int(self.y), int(self.x))
            self.visit_heat[key] = self.visit_heat.get(key, 0) + 1
            if len(self.visit_heat) > 600:
                sorted_keys = sorted(self.visit_heat, key=lambda k: self.visit_heat[k])
                for k in sorted_keys[:150]:
                    del self.visit_heat[k]

        self._update_social(agents, sim_dt)

        action = self.decide(world, agents)
        self.current_action = action
        speed = 2.0 if world.is_day else 1.2
        prev_v = self.prev_vitals
        self._execute(action, world, agents, sim_dt, speed)
        reward = self._calc_reward()
        if abs(reward) > 0.01 or prev_v != self.prev_vitals:
            self.record_event(action, reward)

    def _on_winter_tick(self):
        """
        FIX A — Stage only advances (baby→adult→elder), never regresses.
        The original code had: if winters >= 3: adult, else baby
        which caused adults to revert to baby if winters happened to land
        back below 3 (e.g. after reset / float rounding).
        """
        if self.is_dead:
            return
        if self.winters >= 15:
            self.is_dead = True
            return
        # Stage advances based on cumulative winters, never goes backward
        if self.winters >= 13 and self.stage != "elder":
            self.stage = "elder"
        elif self.winters >= 3 and self.stage == "baby":
            self.stage = "adult"
        # If already adult/elder and winters < 13, do nothing — no regression

    def _execute(self, action, world, agents, sim_dt, speed):
        if action == A_FLEE_FIRE:
            self._flee_fire(world, sim_dt, speed, all_agents=agents)
        elif action in (A_DRINK, A_FIND_WATER):
            self._seek_water(world, sim_dt, speed, agents)
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
        elif action == A_SLEEP:
            pass
        elif action == A_RECREATE:
            self._recreate(world, agents, sim_dt, speed)
        else:
            self._wander(world, sim_dt, speed)
        if action != A_SLEEP:
            self._loot_ground(world)

    def _move_toward(self, ty, tx, world, sim_dt, speed):
        dy = ty - self.y
        dx = tx - self.x
        dist = math.sqrt(dx*dx + dy*dy)
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
            for yy in range(max(0, iy-5), min(WORLD_H, iy+6)):
                for xx in range(max(0, ix-5), min(WORLD_W, ix+6)):
                    if world.fire_intensity[yy, xx] > 0:
                        fires.append((yy, xx))
            if fires:
                fy, fx = min(fires, key=lambda p: abs(p[0]-iy)+abs(p[1]-ix))
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
            self.alert_fire  = False
            self.alert_timer = 0.0

    def _seek_water(self, world, sim_dt, speed, agents=None):
        if self.thirst >= DRINK_SATISFIED:
            self.drink_cooldown = DRINK_COOLDOWN
            return
        drink_radius = BABY_DRINK_RADIUS if self.stage == "baby" else 1
        if world.near_river(self.y, self.x, drink_radius):
            gained = 40 * sim_dt
            self.thirst = min(100, self.thirst + gained)
            self.record_event(A_DRINK, 1.0)
            if self.thirst >= DRINK_SATISFIED:
                self.drink_cooldown = DRINK_COOLDOWN
            return
        river = world.find_nearest_river(self.y, self.x, 60)
        if river:
            self.memory.setdefault("river", []).append(river)
            if len(self.memory["river"]) > 5:
                self.memory["river"].pop(0)
            self._move_toward(river[0], river[1], world, sim_dt, speed)
        else:
            if self.stage == "baby" and agents:
                parent = next(
                    (a for a in agents if a.id in self.parent_ids
                     and not a.is_dead and a.memory.get("river")),
                    None,
                )
                if parent:
                    self._move_toward(parent.y, parent.x, world, sim_dt, speed)
                    return
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
        berry_pos = world.find_nearest(self.y, self.x, O_BERRY, 30)
        wheat_pos = world.find_nearest(self.y, self.x, O_WHEAT, 30)
        rock_pos  = world.find_nearest(self.y, self.x, O_ROCK_OBJ, 15)
        if berry_pos:
            targets.append((berry_pos, O_BERRY))
        if wheat_pos and (stockpile or self.hunger < 80):
            targets.append((wheat_pos, O_WHEAT))
        if rock_pos and self.inv_count("rock") < 4:
            targets.append((rock_pos, O_ROCK_OBJ))
        if stockpile:
            tree = world.find_nearest(self.y, self.x, O_TREE, 30)
            if tree:
                targets.append((tree, O_TREE))

        if not targets:
            self._wander(world, sim_dt, speed)
            return

        target, obj_t = min(targets, key=lambda t: abs(t[0][0]-self.y)+abs(t[0][1]-self.x))
        arrived = self._move_toward(target[0], target[1], world, sim_dt, speed)
        if arrived:
            item = world.harvest(target[0], target[1])
            if item is None and obj_t == O_ROCK_OBJ:
                world.remove_object(target[0], target[1])
                item = {"type": "rock", "quantity": 1, "freshness": None}
            if item:
                self.inv_add(item.get("type","berry"), item.get("quantity",1), item.get("freshness"))
                self.memory.setdefault(obj_t, []).append(target)
                if len(self.memory[obj_t]) > 5:
                    self.memory[obj_t].pop(0)
                self.record_event(A_FORAGE, 0.5)

    def _seek_warmth(self, world, sim_dt, speed):
        if world.near_structure(self.y, self.x, S_FIREPLACE, 3):
            self.warmth = min(100, self.warmth + 8 * sim_dt)
            return
        best = None; best_d = 1e9
        for ty in range(WORLD_H):
            for tx in range(WORLD_W):
                if world.structures[ty, tx] == S_FIREPLACE:
                    d = abs(ty-self.y) + abs(tx-self.x)
                    if d < best_d:
                        best_d = d; best = (ty, tx)
        if best:
            self._move_toward(best[0], best[1], world, sim_dt, speed)
        elif self.has_recipe("fireplace"):
            self._build(world, sim_dt)
        else:
            self._seek_food(world, sim_dt, speed, stockpile=True)

    def _pickup_nearby_items(self, world, radius=2):
        iy, ix = int(self.y), int(self.x)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = iy+dy, ix+dx
                if 0 <= ny < WORLD_H and 0 <= nx < WORLD_W:
                    items = world.pick_dropped(ny, nx)
                    for item in items:
                        if self.inv_total_slots() < self.inv_limit:
                            self.inv_add(item.get("type","rock"), item.get("quantity",1), item.get("freshness"))
                        else:
                            world.drop_items(ny, nx, [item])

    def _craft(self, world, sim_dt):
        if self.craft_remaining > 0:
            self.craft_remaining -= sim_dt
            if self.craft_remaining <= 0:
                self.craft_remaining = 0.0
                item = self.crafting_item
                self.crafting_item = None
                self.inv_add(item, 1)
                self.record_event(A_CRAFT_TOOL, 1.0)
            return

        if self.crafting_item is not None:
            return

        self._pickup_nearby_items(world, radius=2)

        for recipe in ("bread", "fireplace", "axe", "hoe", "shovel", "sharpened_stone"):
            if recipe == "bread" and not world.near_structure(self.y, self.x, S_FIREPLACE, 3):
                continue
            if recipe in ("axe", "hoe", "shovel") and self.has_tool(recipe):
                continue
            if recipe == "sharpened_stone" and self.inv_count("sharpened_stone") >= 2:
                continue
            if self.has_recipe(recipe):
                self.consume_recipe(recipe)
                self.crafting_item   = recipe
                self.craft_remaining = CRAFT_TIME.get(recipe, 3.0)
                return

        self._seek_food(world, sim_dt, 1.0, stockpile=True)

    def _build(self, world, sim_dt):
        """
        FIX C — Simplified building:
          1. Fireplace: rock×4 + wood×2, placed on nearest empty ground.
          2. House: wood×HOUSE_WOOD_COST, replaces a ground tile with S_HOUSE.
          3. Farm: hoe×1 + wood×FARM_WOOD_COST, places S_FARM tile.
        No more wooden_block intermediate.
        """
        # Priority 1: fireplace if none nearby and has materials
        if (not world.near_structure(self.y, self.x, S_FIREPLACE, 8)
                and self.has_recipe("fireplace")):
            spot = world.find_empty_ground(self.y, self.x, 5)
            if spot:
                arrived = self._move_toward(spot[0], spot[1], world, sim_dt, 1.0)
                if arrived:
                    self.consume_recipe("fireplace")
                    world.place_structure(spot[0], spot[1], S_FIREPLACE)
                    self.record_event(A_BUILD, 1.5)
            return

        # Priority 2: house — costs HOUSE_WOOD_COST wood directly
        if self.home_y is None and self.inv_count("wood") >= HOUSE_WOOD_COST:
            spot = world.find_empty_ground(self.y, self.x, 4)
            if spot:
                arrived = self._move_toward(spot[0], spot[1], world, sim_dt, 1.0)
                if arrived:
                    self.inv_remove("wood", HOUSE_WOOD_COST)
                    world.place_structure(spot[0], spot[1], S_HOUSE)
                    self.home_y, self.home_x = spot[0], spot[1]
                    self.memory.setdefault("home", []).append(spot)
                    self.record_event(A_BUILD, 2.0)
            return

        # Priority 3: farm — costs hoe + FARM_WOOD_COST wood
        if (self.has_tool("hoe")
                and self.inv_count("wood") >= FARM_WOOD_COST
                and not world.near_structure(self.y, self.x, S_FARM, 8)):
            if self.home_y is not None:
                spot = world.find_empty_ground(self.home_y, self.home_x, 5)
            else:
                spot = world.find_empty_ground(self.y, self.x, 5)
            if spot:
                arrived = self._move_toward(spot[0], spot[1], world, sim_dt, 1.0)
                if arrived:
                    self.inv_remove("wood", FARM_WOOD_COST)
                    # Hoe is consumed (used up) to build the farm
                    self.inv_remove("hoe", 1)
                    world.place_structure(spot[0], spot[1], S_FARM)
                    self.record_event(A_BUILD, 2.5)
            return

        # Gather wood if needed for any of the above
        tree = world.find_nearest(self.y, self.x, O_TREE, 30)
        if tree:
            arrived = self._move_toward(tree[0], tree[1], world, sim_dt, 1.0)
            if arrived:
                item = world.harvest(tree[0], tree[1])
                if item:
                    self.inv_add(item.get("type","wood"), item.get("quantity",1))
        else:
            self._wander(world, sim_dt, 1.0)

    def _seek_mate(self, world, agents, sim_dt, speed):
        if self.gender != "female":
            best = self._find_best_mate_candidate(agents)
            if best is None:
                self._wander(world, sim_dt, speed)
                return
            self._move_toward(best.y, best.x, world, sim_dt, speed)
            return

        best = self._find_best_mate_candidate(agents)
        if best is None:
            self._wander(world, sim_dt, speed)
            return

        arrived = self._move_toward(best.y, best.x, world, sim_dt, speed)
        if not arrived:
            return

        pop = len([a for a in agents if not a.is_dead])
        pop_factor = max(0.0, 1.0 - (pop / max(1, MAX_AGENTS)) ** POP_PRESSURE_EXP)
        conception_chance = BIRTH_SUCCESS_RATE * pop_factor

        if random.random() < conception_chance * sim_dt * 0.1:
            self.partner_id = best.id
            best.partner_id = self.id
            self.preg_timer = 30.0
            self.children_born += 1
            self.gain_friendship(best.id, FRIEND_DANGER_GAIN)
            best.gain_friendship(self.id, FRIEND_DANGER_GAIN)
            self.record_event(A_SEEK_MATE, 1.0)
            self.heart_effect_timer = HEART_EFFECT_DURATION
            best.heart_effect_timer = HEART_EFFECT_DURATION

    def _find_best_mate_candidate(self, agents):
        best = None; best_score = -1.0
        for ag in agents:
            if ag is self or ag.is_dead: continue
            if ag.gender == self.gender:  continue
            if ag.stage != "adult":       continue
            if self.is_family(ag.id):     continue
            if self.is_nemesis(ag.id):    continue
            if ag.postpartum_timer > 0:   continue
            if ag.preg_timer is not None: continue
            score = self.friendship_score(ag.id)
            if self.gender == "female" and score < FRIEND_MATE_THRESHOLD:
                continue
            if score > best_score:
                best_score = score; best = ag
        return best

    def _seek_safety(self, world, sim_dt, speed):
        if self.home_y is not None:
            self._move_toward(self.home_y, self.home_x, world, sim_dt, speed * 1.2)
        else:
            self._wander(world, sim_dt, speed)

    def _recreate(self, world, agents, sim_dt, speed):
        target_ag = None
        for ag in agents:
            if ag is self or ag.is_dead:
                continue
            if self.friendship_score(ag.id) > FRIEND_THRESHOLD or self.is_family(ag.id):
                dist = abs(ag.x - self.x) + abs(ag.y - self.y)
                if 3 < dist < 18:
                    target_ag = ag
                    break
        if target_ag is not None:
            self._move_toward(
                target_ag.y + random.uniform(-1.5, 1.5),
                target_ag.x + random.uniform(-1.5, 1.5),
                world, sim_dt, speed * 0.5,
            )
        else:
            self._wander(world, sim_dt, speed * 0.4)
        self.mood = min(100, self.mood + 0.5 * sim_dt)

    def _wander(self, world, sim_dt, speed):
        at_target = (
            self.target is None
            or (abs(self.x - self.target[1]) < 1 and abs(self.y - self.target[0]) < 1)
        )
        if at_target:
            home_y = self.home_y if self.home_y is not None else WORLD_H / 2
            home_x = self.home_x if self.home_x is not None else WORLD_W / 2
            r = 4.0 + self.exploration * 40.0
            if not world.is_day:
                r *= 0.5
            if self.exploration < 0.5 and self.visit_heat:
                best_key = None; best_score = -1
                for (ky, kx), heat in self.visit_heat.items():
                    dist_from_home = abs(ky-home_y) + abs(kx-home_x)
                    if dist_from_home > r * 2: continue
                    score = heat * (1.0 - dist_from_home / max(r*2, 1))
                    if score > best_score:
                        best_score = score; best_key = (ky, kx)
                if best_key and best_score > 0:
                    jitter = 3.0 * (1.0 - self.exploration)
                    ty = _clamp(best_key[0] + random.uniform(-jitter, jitter), 0, WORLD_H-1)
                    tx = _clamp(best_key[1] + random.uniform(-jitter, jitter), 0, WORLD_W-1)
                else:
                    ty = _clamp(home_y + random.uniform(-r*0.4, r*0.4), 0, WORLD_H-1)
                    tx = _clamp(home_x + random.uniform(-r*0.4, r*0.4), 0, WORLD_W-1)
            else:
                ty = _clamp(home_y + random.uniform(-r, r), 0, WORLD_H-1)
                tx = _clamp(home_x + random.uniform(-r, r), 0, WORLD_W-1)
            self.target = (ty, tx)
        self._move_toward(self.target[0], self.target[1], world, sim_dt, speed * 0.6)

    def _loot_ground(self, world):
        items = world.pick_dropped(int(self.y), int(self.x))
        for item_dict in items:
            if self.inv_total_slots() >= self.inv_limit:
                continue
            self.inv_add(item_dict.get("type","rock"), item_dict.get("quantity",1), item_dict.get("freshness"))

    # ── Pregnancy / birth ─────────────────────────────────────

    def tick_pregnancy(self, world, agents, sim_dt):
        if self.preg_timer is None:
            return None
        self.preg_timer -= sim_dt
        if self.preg_timer > 0:
            return None

        self.preg_timer = None
        mother_stage_before = self.stage

        self.postpartum_timer = PREGNANCY_COOLDOWN
        if self.health < POSTPARTUM_HEALTH:
            self.health = POSTPARTUM_HEALTH

        pop = len([a for a in agents if not a.is_dead])
        pop_factor = max(0.0, 1.0 - (pop / max(1, MAX_AGENTS)) ** POP_PRESSURE_EXP)
        if random.random() > BIRTH_SUCCESS_RATE * max(0.3, pop_factor):
            assert self.stage == mother_stage_before
            return None

        child = Agent(
            self.x + random.uniform(-1, 1),
            self.y + random.uniform(-1, 1),
            world=world,
        )
        child.stage      = "baby"
        child.winters    = 0
        child.age        = 0.0
        child.health     = BABY_HEALTH_START
        child.hunger     = 80.0
        child.thirst     = 80.0
        child.warmth     = 70.0
        child.mood       = 80.0
        child.generation = self.generation + 1
        child.grace_timer = BABY_GRACE_PERIOD
        child.is_dead    = False

        child.parent_ids = [self.id]
        if self.partner_id:
            child.parent_ids.append(self.partner_id)
        self.children_ids.append(child.id)

        child.memory = {}
        for k, v in self.memory.items():
            child.memory[k] = v[:int(len(v) * KNOWLEDGE_INHERIT)]

        if self.visit_heat:
            sorted_heat = sorted(self.visit_heat.items(), key=lambda kv: -kv[1])
            inherited   = sorted_heat[:int(len(sorted_heat) * KNOWLEDGE_INHERIT)]
            child.visit_heat = dict(inherited)

        partner = next((a for a in agents if a.id == self.partner_id), None)
        if partner:
            child.weights = [
                max(0.3, min(3.0, self.weights[i]*0.5 + partner.weights[i]*0.5 + random.gauss(0, MUTATION_RATE)))
                for i in range(NUM_ACTIONS)
            ]
            child.exploration = max(0.1, min(0.9,
                self.exploration*0.5 + partner.exploration*0.5 + random.gauss(0, 0.08)
            ))
        else:
            child.weights = [
                max(0.3, min(3.0, w + random.gauss(0, MUTATION_RATE)))
                for w in self.weights
            ]
            child.exploration = max(0.1, min(0.9,
                self.exploration + random.gauss(0, 0.08)
            ))

        if self.inv_count("berry") > 1:
            self.inv_remove("berry", 1)
            child.inv_add("berry", 1, freshness=100.0)

        assert self.stage == mother_stage_before, (
            f"BUG: mother {self.id} stage changed during birth! "
            f"Before={mother_stage_before}, After={self.stage}"
        )
        assert not self.is_dead or self.health <= 0

        return child
