"""
HUD — draws stats overlay using pygame surfaces, composited over the GL frame.
Includes: season/weather bar, agent count, selected agent vitals panel,
population chart, learning weights radar, event log.
"""

import pygame
import math
from engine.config import *
from ai.actions import ACTION_NAMES, NUM_ACTIONS

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
    "clear": "☀", "cloudy": "☁", "rain": "🌧", "storm": "⛈", "blizzard": "❄"
}

BAR_COLORS = {
    "hunger": (230, 160, 60),
    "thirst": (80, 160, 240),
    "warmth": (240, 100, 60),
    "mood":   (180, 120, 240),
    "health": (80, 220, 80),
}


def _bar(surf, x, y, w, h, val, col, label, font):
    pygame.draw.rect(surf, (30, 30, 30), (x, y, w, h))
    fill = int(w * max(0, min(100, val)) / 100)
    pygame.draw.rect(surf, col, (x, y, fill, h))
    pygame.draw.rect(surf, (80, 80, 80), (x, y, w, h), 1)
    txt = font.render(f"{label} {int(val):3d}", True, (220, 220, 220))
    surf.blit(txt, (x + 2, y))


def draw_hud(screen_surf, world, agents, selected_agent, pop_history, sim_speed_ref):
    ensure_fonts()
    W, H = screen_surf.get_size()

    # ── top bar ──────────────────────────────────────────────
    bar_h = 28
    pygame.draw.rect(screen_surf, (20, 20, 25, 200), (0, 0, W, bar_h))

    season = world.season_name
    scol   = SEASON_COLORS.get(season, (200, 200, 200))
    day_pct = (world.day_number % SEASON_DAYS) / SEASON_DAYS

    # season progress pill
    pill_w = 180
    pygame.draw.rect(screen_surf, (40, 40, 45), (8, 4, pill_w, 20), border_radius=4)
    pygame.draw.rect(screen_surf, scol, (8, 4, int(pill_w * day_pct), 20), border_radius=4)
    stxt = FONT_TITLE.render(f"{season.upper()} · Day {world.day_number % SEASON_DAYS + 1}", True, (240,240,240))
    screen_surf.blit(stxt, (12, 6))

    # weather
    wicon = WEATHER_ICONS.get(world.weather, "?")
    wtxt  = FONT_TITLE.render(f"{world.weather.upper()}", True, (200, 220, 240))
    screen_surf.blit(wtxt, (200, 6))

    # time of day
    tod  = world.time_of_day
    hour = int(tod * 24)
    ampm = "AM" if hour < 12 else "PM"
    h12  = hour % 12 or 12
    ttxt = FONT_TITLE.render(f"{h12:02d}:00 {ampm}", True, (240, 200, 100) if tod < 0.5 else (120, 130, 180))
    screen_surf.blit(ttxt, (380, 6))

    # agent count
    live = [a for a in agents if not a.is_dead]
    atxt = FONT_TITLE.render(f"👥 {len(live)}/{MAX_AGENTS}", True, (180, 240, 180))
    screen_surf.blit(atxt, (500, 6))

    # sim speed
    sptxt = FONT_TITLE.render(f"⚡{sim_speed_ref[0]:.0f}×", True, (240, 180, 80))
    screen_surf.blit(sptxt, (620, 6))

    # ── population chart (bottom-left) ───────────────────────
    ch_w, ch_h = 180, 60
    ch_x, ch_y = 10, H - ch_h - 10
    pygame.draw.rect(screen_surf, (15, 15, 20, 200), (ch_x, ch_y, ch_w, ch_h))
    pygame.draw.rect(screen_surf, (60, 60, 70), (ch_x, ch_y, ch_w, ch_h), 1)
    if pop_history:
        max_p = max(max_pop := max(pop_history), 1)
        pts = []
        for i, p in enumerate(pop_history[-ch_w:]):
            px = ch_x + int(i * ch_w / max(len(pop_history[-ch_w:]), 1))
            py = ch_y + ch_h - int(p / max_p * (ch_h - 4)) - 2
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(screen_surf, (80, 220, 80), False, pts, 2)
    plbl = FONT_SM.render("Population", True, (160, 200, 160))
    screen_surf.blit(plbl, (ch_x + 4, ch_y + 2))

    # ── selected agent panel (right side) ─────────────────────
    if selected_agent and not selected_agent.is_dead:
        ag   = selected_agent
        px   = W - 220
        py   = 40
        pw, ph = 210, 300
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((15, 15, 20, 210))
        pygame.draw.rect(panel, (80, 80, 100), (0, 0, pw, ph), 1)

        # name/stage
        gen_col = (min(255, 100 + ag.generation * 30), 200, 200)
        ntxt = FONT_TITLE.render(f"{ag.id}  {ag.stage.upper()}", True, gen_col)
        panel.blit(ntxt, (6, 5))
        gtxt = FONT_SM.render(f"{ag.gender} · Gen {ag.generation} · Winter {ag.winters}", True, (180,180,200))
        panel.blit(gtxt, (6, 20))

        # vitals bars
        by = 38
        for stat, col in BAR_COLORS.items():
            val = getattr(ag, stat, 100)
            _bar(panel, 6, by, pw-12, 14, val, col, stat[:3].upper(), FONT_SM)
            by += 17

        # current action
        aname = ACTION_NAMES[ag.current_action] if ag.current_action is not None else "?"
        atxt2 = FONT_SM.render(f"Action: {aname}", True, (220, 220, 100))
        panel.blit(atxt2, (6, by + 2))
        by += 16

        # inventory
        inv_txt = FONT_SM.render("Inventory:", True, (180, 180, 180))
        panel.blit(inv_txt, (6, by))
        by += 13
        for itype, idata in list(ag.inventory.items())[:5]:
            qty = idata["quantity"]
            itxt = FONT_SM.render(f"  {itype[:12]:<12} ×{qty}", True, (200, 200, 160))
            panel.blit(itxt, (6, by))
            by += 12

        # learning weights radar (mini)
        if by + 80 < ph:
            by += 4
            _draw_radar(panel, 6, by, pw-12, 70, ag.weights)
            by += 74

        screen_surf.blit(panel, (px, py))

    # ── mini event log (bottom-right) ─────────────────────────
    log_x = W - 260
    log_y = H - 120
    if selected_agent and selected_agent.event_log:
        for i, (aid, reward) in enumerate(selected_agent.event_log[-8:]):
            col = (80, 220, 80) if reward > 0 else (220, 80, 80)
            etxt = FONT_SM.render(f"{ACTION_NAMES[aid]} {reward:+.2f}", True, col)
            screen_surf.blit(etxt, (log_x, log_y + i * 13))


def _draw_radar(surf, x, y, w, h, weights):
    """Mini radar chart for the NUM_ACTIONS weights."""
    cx, cy = x + w//2, y + h//2
    r  = min(w, h) // 2 - 4
    n  = NUM_ACTIONS
    pts_outer = []
    pts_val   = []
    for i in range(n):
        angle = 2 * math.pi * i / n - math.pi / 2
        ox = cx + int(r * math.cos(angle))
        oy = cy + int(r * math.sin(angle))
        pts_outer.append((ox, oy))
        v = weights[i] / 3.0
        vx = cx + int(r * v * math.cos(angle))
        vy = cy + int(r * v * math.sin(angle))
        pts_val.append((vx, vy))
    if len(pts_outer) >= 3:
        pygame.draw.polygon(surf, (50, 50, 70), pts_outer)
        pygame.draw.polygon(surf, (80, 80, 100), pts_outer, 1)
        pygame.draw.polygon(surf, (80, 180, 120, 160), pts_val)
        pygame.draw.polygon(surf, (100, 220, 140), pts_val, 1)
