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

# ── Social relationship system ────────────────────────────────

# Friendship
MAX_FRIENDS          = 6        # cap on non-family friends
FRIEND_NEAR_GAIN     = 0.04     # score gained per sim-second while adjacent
FRIEND_SHARE_GAIN    = 2.0      # score gained when an agent shares an item
FRIEND_DANGER_GAIN   = 5.0      # score gained when surviving danger together
FRIEND_FORGET_CHANCE = 0.001    # chance per tick a non-refreshed friendship decays
FRIEND_FORGET_AMOUNT = 1.0      # how much the score drops on a forget tick
FRIEND_THRESHOLD     = 10.0     # score needed to be "friends"
FRIEND_MATE_THRESHOLD = 25.0    # minimum friendship score to attempt reproduction
FAMILY_FRIEND_BONUS  = 999.0    # internal constant: family ties never decay

# Nemesis
NEMESIS_CHANCE       = 0.08     # chance per collision/fight tick two males become nemeses
NEMESIS_FIGHT_CHANCE = 0.15     # chance per tick nemeses start a fight when close
FIGHT_JOIN_CHANCE    = 0.25     # chance a nearby friend joins a fight
FIGHT_DAMAGE         = 5.0      # health damage per fight tick
FIGHT_MOOD_LOSS      = 3.0      # mood hit on fight winner too

# Reproduction
PREGNANCY_COOLDOWN   = 180.0    # sim-seconds before a mother can conceive again
POSTPARTUM_HEALTH    = 60.0     # mother health floor right after birth (prevents death)
BIRTH_SUCCESS_RATE   = 0.30     # base chance a qualifying pair actually conceives
# Soft population pressure: birth chance scales by (1 - pop/MAX_AGENTS)^POP_PRESSURE_EXP
POP_PRESSURE_EXP     = 2.0
REPRO_MIN_FOOD       = 30       # minimum total food value in inventory to conceive
REPRO_MIN_HUNGER     = 55       # minimum hunger stat to conceive
REPRO_MIN_MOOD       = 55       # minimum mood stat to conceive
REPRO_NEED_SHELTER   = False    # set True to require a home_y to conceive

# Baby
BABY_GRACE_PERIOD    = 60.0     # sim-seconds after birth where baby gets extra protection
BABY_HEALTH_START    = 90.0     # newborns start with high health
BABY_HUNGER_DECAY    = 0.55     # hunger decay multiplier for babies (lower than adults)
BABY_THIRST_DECAY    = 0.80     # thirst decay multiplier for babies
BABY_DRINK_RADIUS    = 3        # babies can drink from water tiles up to this many tiles away

# Drinking / thirst satisfaction
DRINK_COOLDOWN       = 20.0     # sim-seconds agent cannot re-select drink after drinking full
DRINK_THRESHOLD      = 45.0     # thirst must be BELOW this to trigger seek-water priority
DRINK_SATISFIED      = 85.0     # thirst level considered "satisfied" (resets cooldown timer)

# Building utility thresholds
BUILD_FOOD_SURPLUS   = 60       # total food value above which building becomes attractive
BUILD_PREFER_THRESH  = 70       # hunger+thirst+warmth all above this → prefer build/improve

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
