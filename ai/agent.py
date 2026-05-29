diff --git a/ai/agent.py b/ai/agent.py
index 95fb650ff80616a526c876ba8fe0a62f5ead9248..e3b3912b21585a4041531cd78e41f53ba3919424 100644
--- a/ai/agent.py
+++ b/ai/agent.py
@@ -1,62 +1,58 @@
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
 
-# ─── Action IDs ───────────────────────────────────────────────
-A_FLEE_FIRE    = 0
-A_DRINK        = 1
-A_SEEK_SAFETY  = 2
-A_FIND_WATER   = 3
-A_FIND_FOOD    = 4
-A_SEEK_WARMTH  = 5
-A_FORAGE       = 6
-A_STOCKPILE    = 7
-A_CRAFT_TOOL   = 8
-A_BUILD        = 9
-A_SEEK_MATE    = 10
-A_WANDER       = 11
-NUM_ACTIONS    = 12
-
-ACTION_NAMES = [
-    "🔥Flee", "💧Drink", "🛡Safety", "🔍Water", "🍎Food",
-    "🌡Warmth", "🌾Forage", "📦Stock", "🔨Craft", "🏠Build",
-    "❤Mate", "👣Wander"
-]
+from ai.actions import (
+    ACTION_NAMES,
+    NUM_ACTIONS,
+    A_FLEE_FIRE,
+    A_DRINK,
+    A_SEEK_SAFETY,
+    A_FIND_WATER,
+    A_FIND_FOOD,
+    A_SEEK_WARMTH,
+    A_FORAGE,
+    A_STOCKPILE,
+    A_CRAFT_TOOL,
+    A_BUILD,
+    A_SEEK_MATE,
+    A_WANDER,
+)
 
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
 
