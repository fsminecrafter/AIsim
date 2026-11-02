# ─────────────────────────────────────────────────────────────
#  Survival Simulation — Configuration  (v4)
# ─────────────────────────────────────────────────────────────

# World  (increased from 80×80)
WORLD_W       = 128
WORLD_H       = 128
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

# Seasons enabled
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
NEURAL_SIZE          = 14

# Terrain seed
TERRAIN_SEED         = 42

# GPU / rendering
GPU_AVAILABLE        = False  # detected at startup

# ── Social relationship system ────────────────────────────────

# Friendship
MAX_FRIENDS          = 6
FRIEND_NEAR_GAIN     = 0.04
FRIEND_SHARE_GAIN    = 2.0
FRIEND_DANGER_GAIN   = 5.0
FRIEND_FORGET_CHANCE = 0.001
FRIEND_FORGET_AMOUNT = 1.0
FRIEND_THRESHOLD     = 10.0
FRIEND_MATE_THRESHOLD = 25.0
FAMILY_FRIEND_BONUS  = 999.0

# Nemesis
NEMESIS_CHANCE       = 0.08
NEMESIS_FIGHT_CHANCE = 0.15
FIGHT_JOIN_CHANCE    = 0.25
FIGHT_DAMAGE         = 5.0
FIGHT_MOOD_LOSS      = 3.0

# Reproduction
PREGNANCY_COOLDOWN   = 180.0
POSTPARTUM_HEALTH    = 60.0
BIRTH_SUCCESS_RATE   = 0.30
POP_PRESSURE_EXP     = 2.0
REPRO_MIN_FOOD       = 30
REPRO_MIN_HUNGER     = 55
REPRO_MIN_MOOD       = 55
REPRO_NEED_SHELTER   = False

# Baby
BABY_GRACE_PERIOD    = 60.0
BABY_HEALTH_START    = 90.0
BABY_HUNGER_DECAY    = 0.55
BABY_THIRST_DECAY    = 0.80
BABY_DRINK_RADIUS    = 3

# Drinking / thirst satisfaction
DRINK_COOLDOWN       = 20.0
DRINK_THRESHOLD      = 45.0
DRINK_SATISFIED      = 85.0

# Building utility thresholds
BUILD_FOOD_SURPLUS   = 40       # lowered so agents build sooner
BUILD_PREFER_THRESH  = 55       # lowered so building is easier to trigger

# ── Sleep (FIX 3) ─────────────────────────────────────────────
SLEEP_NIGHT_MOOD_THRESH = 40.0
SLEEP_MIN_DURATION      = 15.0
SLEEP_MAX_DURATION      = 45.0
SLEEP_WARMTH_RESTORE    = 2.0
SLEEP_MOOD_RESTORE      = 1.5
SLEEP_HEALTH_RESTORE    = 0.3

# ── Recreation gate ───────────────────────────────────────────
# Lowered thresholds so recreation keeps being triggered, not just early-game
RECREATION_FOOD_GATE    = 15     # reduced from 20
RECREATION_VITAL_GATE   = 55.0   # reduced from 60

# ── Heart effect ──────────────────────────────────────────────
HEART_EFFECT_DURATION   = 3.0

# ── Building structures ───────────────────────────────────────
# Houses now replace a tile (no wooden_block needed, use wood directly)
HOUSE_WOOD_COST         = 6      # wood needed to build a house
FARM_HOE_COST           = 1      # 1 hoe
FARM_WOOD_COST          = 6      # 6 wood

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
