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
# 14 = new default (includes A_SLEEP + A_RECREATE)
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
BUILD_FOOD_SURPLUS   = 60
BUILD_PREFER_THRESH  = 70

# ── Sleep (FIX 3) ─────────────────────────────────────────────
SLEEP_NIGHT_MOOD_THRESH = 40.0   # mood below this at night → try to sleep
SLEEP_MIN_DURATION      = 15.0   # minimum sim-seconds per nap
SLEEP_MAX_DURATION      = 45.0   # maximum sim-seconds per nap
SLEEP_WARMTH_RESTORE    = 2.0    # warmth/s while sleeping
SLEEP_MOOD_RESTORE      = 1.5    # mood/s while sleeping
SLEEP_HEALTH_RESTORE    = 0.3    # health/s while sleeping

# ── Recreation gate (FIX 2) ───────────────────────────────────
RECREATION_FOOD_GATE    = 20     # food-value units above which recreation kicks in
RECREATION_VITAL_GATE   = 60.0   # all core vitals must exceed this to recreate

# ── Heart effect (FIX 5) ──────────────────────────────────────
HEART_EFFECT_DURATION   = 3.0    # seconds heart floats above colonist after conception

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
