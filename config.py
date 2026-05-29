# ─────────────────────────────────────────────────────────────
#  Survival Simulation — Configuration
# ─────────────────────────────────────────────────────────────

# World
WORLD_W       = 80          # grid tiles wide
WORLD_H       = 80          # grid tiles tall
TILE_SIZE     = 10.0        # world-units per tile

# Display
SCREEN_W      = 1280
SCREEN_H      = 800
TARGET_FPS    = 60

# Simulation speed (1.0 = real-time, higher = faster)
SIM_SPEED     = 6.0

# Agents
INITIAL_AGENTS     = 12
MAX_AGENTS         = 120
AGENT_RADIUS       = 0.35   # visual size in tiles

# Vitals (per real second at SIM_SPEED=1, scaled later)
HUNGER_DECAY   = 1.0 / 60.0
THIRST_DECAY   = 2.0 / 60.0
WARMTH_DECAY   = 0.5 / 60.0
MOOD_DECAY     = 0.3 / 60.0

# Time
DAY_DURATION       = 60.0   # real seconds per in-game day (at SIM_SPEED=1)
SEASON_DAYS        = 20     # in-game days per season
SEASONS            = ["spring", "summer", "autumn", "winter"]

# Fire
FIRE_SPREAD_INTERVAL = 2.0
FIRE_BASE_SPREAD     = 0.15

# Learning / genetics
LEARNING_RATE        = 0.1
MUTATION_RATE        = 0.05
KNOWLEDGE_INHERIT    = 0.7   # fraction of parent knowledge passed to child

# Colors (RGBA float)
COL_GRASS      = (0.22, 0.45, 0.15, 1.0)
COL_DIRT       = (0.55, 0.38, 0.22, 1.0)
COL_ROCK       = (0.50, 0.50, 0.52, 1.0)
COL_RIVER      = (0.15, 0.40, 0.75, 1.0)
COL_TREE       = (0.10, 0.35, 0.10, 1.0)
COL_BERRY      = (0.60, 0.10, 0.10, 1.0)
COL_WHEAT      = (0.85, 0.75, 0.20, 1.0)
COL_FIRE       = (0.95, 0.40, 0.05, 1.0)
COL_HOUSE      = (0.62, 0.48, 0.32, 1.0)
COL_FIREPLACE  = (0.80, 0.30, 0.00, 1.0)
