
 """
 HUD — draws stats overlay using pygame surfaces, composited over the GL frame.
 Includes: season/weather bar, agent count, selected agent vitals panel,
 population chart, learning weights radar, event log.
 """
 
 import pygame
 import math
 from engine.config import *
-from ai.agent import ACTION_NAMES, NUM_ACTIONS
+from ai.actions import ACTION_NAMES, NUM_ACTIONS
 
 FONT_MONO = None
 FONT_TITLE = None
 FONT_SM = None
 
 _pygame_init = False
 
 def ensure_fonts():
     global FONT_MONO, FONT_TITLE, FONT_SM, _pygame_init
     if not _pygame_init:
         pygame.font.init()
         _pygame_init = True
     if FONT_MONO is None:
         FONT_MONO  = pygame.font.SysFont("monospace",   13, bold=False)
         FONT_TITLE = pygame.font.SysFont("monospace",   15, bold=True)
         FONT_SM    = pygame.font.SysFont("monospace",   11)
 
 
 SEASON_COLORS = {
     "spring": (80, 200, 120),
     "summer": (240, 200, 60),
     "autumn": (210, 120, 40),
     "winter": (160, 190, 230),
 }
 WEATHER_ICONS = {
