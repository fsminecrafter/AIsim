# ─────────────────────────────────────────────────────────────
#  Survival Simulation — Configuration
# ─────────────────────────────────────────────────────────────

# World
WORLD_W       = 80
WORLD_H       = 80
TILE_SIZE     = 10.0

# Display
SCREEN_W      = 1280
SCREEN_H      = 800
TARGET_FPS    = 60

# Simulation speed
SIM_SPEED     = 6.0

# Agents
INITIAL_AGENTS     = 12
MAX_AGENTS         = 120
AGENT_RADIUS       = 0.35

# Vitals (per real second at SIM_SPEED=1)
HUNGER_DECAY   = 1.0 / 60.0
THIRST_DECAY   = 2.0 / 60.0
WARMTH_DECAY   = 0.5 / 60.0
MOOD_DECAY     = 0.3 / 60.0

# Time
DAY_DURATION       = 60.0
SEASON_DAYS        = 20
SEASONS            = ["spring", "summer", "autumn", "winter"]

# Seasons enabled (can be overridden by menu)
SEASONS_ENABLED    = True
STARTING_SEASON    = 0   # 0=spring 1=summer 2=autumn 3=winter

# Fire
FIRE_SPREAD_INTERVAL = 2.0
FIRE_BASE_SPREAD     = 0.15

# Learning / genetics
LEARNING_RATE        = 0.1
MUTATION_RATE        = 0.05
KNOWLEDGE_INHERIT    = 0.7

# Neural network size (number of hidden units in agent weight vector)
# 12 = original, can be expanded
NEURAL_SIZE          = 12

# Terrain seed
TERRAIN_SEED         = 42

# GPU / rendering
GPU_AVAILABLE        = False  # detected at startup

# Colors (RGBA float)
COL_GRASS      = (0.22, 0.45, 0.15, 1.0)
COL_DIRT       = (0.55, 0.38, 0.22, 1.0)
COL_ROCK       = (0.50, 0.50, 0.52, 1.0)
COL_RIVER      = (0.15, 0.40, 0.75, 1.0)
COL_SAND       = (0.85, 0.78, 0.50, 1.0)
COL_MOUNTAIN   = (0.55, 0.52, 0.50, 1.0)
COL_SNOW       = (0.90, 0.92, 0.95, 1.0)
COL_TREE       = (0.10, 0.35, 0.10, 1.0)
COL_BERRY      = (0.60, 0.10, 0.10, 1.0)
COL_WHEAT      = (0.85, 0.75, 0.20, 1.0)
COL_FIRE       = (0.95, 0.40, 0.05, 1.0)
COL_HOUSE      = (0.62, 0.48, 0.32, 1.0)
COL_FIREPLACE  = (0.80, 0.30, 0.00, 1.0)