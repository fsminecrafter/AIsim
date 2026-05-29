"""
Shared action identifiers for the agent decision system and HUD.

Keep these constants in a lightweight module so display code can render action
labels without importing the full agent implementation and its simulation-time
dependencies.
"""

# ─── Action IDs ───────────────────────────────────────────────
A_FLEE_FIRE = 0
A_DRINK = 1
A_SEEK_SAFETY = 2
A_FIND_WATER = 3
A_FIND_FOOD = 4
A_SEEK_WARMTH = 5
A_FORAGE = 6
A_STOCKPILE = 7
A_CRAFT_TOOL = 8
A_BUILD = 9
A_SEEK_MATE = 10
A_WANDER = 11

ACTION_NAMES = [
    "🔥Flee", "💧Drink", "🛡Safety", "🔍Water", "🍎Food",
    "🌡Warmth", "🌾Forage", "📦Stock", "🔨Craft", "🏠Build",
    "❤Mate", "👣Wander"
]
NUM_ACTIONS = len(ACTION_NAMES)

__all__ = [
    "A_FLEE_FIRE", "A_DRINK", "A_SEEK_SAFETY", "A_FIND_WATER",
    "A_FIND_FOOD", "A_SEEK_WARMTH", "A_FORAGE", "A_STOCKPILE",
    "A_CRAFT_TOOL", "A_BUILD", "A_SEEK_MATE", "A_WANDER",
    "ACTION_NAMES", "NUM_ACTIONS",
]
