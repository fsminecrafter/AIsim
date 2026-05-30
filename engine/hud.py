"""
HUD — draws stats overlay using pygame surfaces, composited over the GL frame.
"""

import pygame
import math
from engine.config import *
from ai.actions import ACTION_NAMES, NUM_ACTIONS

FONT_MONO  = None
FONT_TITLE = None
FONT_SM    = None
FONT_XS    = None
_pygame_init = False


def ensure_fonts():
    global FONT_MONO, FONT_TITLE, FONT_SM, FONT_XS, _pygame_init
    if not _pygame_init:
        pygame.font.init()
        _pygame_init = True
    if FONT_MONO is None:
        FONT_MONO  = pygame.font.SysFont("monospace", 13, bold=False)
        FONT_TITLE = pygame.font.SysFont("monospace", 14, bold=True)
        FONT_SM    = pygame.font.SysFont("monospace", 11)
        FONT_XS    = pygame.font.SysFont("monospace", 10)


# ── constants ─────────────────────────────────────────────────

SEASON_COLORS = {
    "spring": (80,  200, 120),
    "summer": (240, 200,  60),
    "autumn": (210, 120,  40),
    "winter": (160, 190, 230),
}

WEATHER_ICON = {
    "clear": "☀", "cloudy": "☁", "rain": "🌧", "storm": "⛈", "blizzard": "❄"
}

VITAL_COLS = {
    "hunger": (230, 155,  55),
    "thirst": ( 70, 155, 235),
    "warmth": (235,  90,  55),
    "mood":   (170, 110, 235),
    "health": ( 70, 210,  70),
}

ACTION_COLS = {
    0:  (255,  80,  30),
    1:  ( 80, 160, 255),
    2:  (160, 200, 255),
    3:  ( 80, 180, 255),
    4:  (100, 230,  80),
    5:  (255, 160,  50),
    6:  (180, 230,  80),
    7:  (200, 200,  80),
    8:  (200, 140,  60),
    9:  (220, 185,  90),
    10: (255,  80, 180),
    11: (140, 140, 155),
}

STAGE_COLS = {
    "baby":  (240, 210,  70),
    "adult": (200, 215, 235),
    "elder": (175, 170, 200),
}

GENDER_ICON = {"male": "♂", "female": "♀"}
GENDER_COL  = {"male": (110, 170, 240), "female": (240, 120, 185)}


# ── primitive helpers ─────────────────────────────────────────

def _panel(surf, x, y, w, h, alpha=200, border=True):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((10, 12, 18, alpha))
    if border:
        pygame.draw.rect(s, (60, 70, 95), (0, 0, w, h), 1)
    surf.blit(s, (x, y))


def _bar(surf, x, y, w, h, val, col, label, font, show_num=True):
    pygame.draw.rect(surf, (25, 27, 35), (x, y, w, h))
    fill = int(w * max(0, min(100, val)) / 100)
    pygame.draw.rect(surf, col, (x, y, fill, h))
    if fill > 2:
        light = tuple(min(255, c + 60) for c in col)
        pygame.draw.line(surf, light, (x, y), (x + fill - 1, y))
    pygame.draw.rect(surf, (65, 70, 95), (x, y, w, h), 1)
    if show_num:
        txt = font.render(f"{label} {int(val):3d}", True, (215, 218, 228))
        surf.blit(txt, (x + 3, y))


def _label(surf, x, y, text, font, col=(200, 205, 220)):
    t = font.render(text, True, col)
    surf.blit(t, (x, y))
    return t.get_width()


# ── colonist leaderboard ──────────────────────────────────────

def draw_colonist_panel(screen_surf, mx, my, agents, selected_agent,
                        agent_panel_active=False):
    """
    Clickable top-10 colonist leaderboard ranked by composite vitals score.
    Returns the Agent that was clicked (to select it), or None.
    """
    ensure_fonts()
    W = screen_surf.get_width()
    pw, row_h = 210, 17
    right_offset = 8 + (215 + 6 if agent_panel_active else 0)
    px = W - pw - right_offset
    py = 40
    header_h = 20

    live = [a for a in agents if not a.is_dead]

    def _score(a):
        return (getattr(a, "health", 100)
                + getattr(a, "hunger", 100) * 0.3
                + getattr(a, "thirst", 100) * 0.3
                + getattr(a, "warmth", 100) * 0.2
                + getattr(a, "mood",   100) * 0.2
                + getattr(a, "winters", 0)  * 15)

    top10   = sorted(live, key=_score, reverse=True)[:10]
    total_h = header_h + len(top10) * row_h + 6

    _panel(screen_surf, px, py, pw, total_h, alpha=215)
    _label(screen_surf, px + 8, py + 4,
           f"COLONISTS  {len(live)} alive", FONT_XS, (90, 100, 130))

    clicked = None
    for i, ag in enumerate(top10):
        ry          = py + header_h + i * row_h
        row_rect_x  = px + 4
        row_rect_w  = pw - 8

        is_sel  = (ag is selected_agent)
        hovered = (row_rect_x <= mx <= row_rect_x + row_rect_w and
                   ry <= my <= ry + row_h - 1)

        if is_sel:
            s = pygame.Surface((row_rect_w, row_h - 1), pygame.SRCALPHA)
            s.fill((70, 110, 80, 160))
            screen_surf.blit(s, (row_rect_x, ry))
        elif hovered:
            s = pygame.Surface((row_rect_w, row_h - 1), pygame.SRCALPHA)
            s.fill((60, 70, 100, 120))
            screen_surf.blit(s, (row_rect_x, ry))

        _label(screen_surf, px + 6,  ry + 2, f"{i+1:2d}.", FONT_XS, (80, 88, 115))

        gcol  = GENDER_COL.get(ag.gender, (160, 160, 160))
        gicon = GENDER_ICON.get(ag.gender, "?")
        _label(screen_surf, px + 26, ry + 2, gicon, FONT_XS, gcol)

        ncol = (255, 255, 200) if is_sel else ((255, 255, 255) if hovered else (215, 218, 228))
        _label(screen_surf, px + 38, ry + 2, str(ag.id)[:10], FONT_XS, ncol)

        scol   = STAGE_COLS.get(ag.stage, (180, 180, 180))
        sbadge = FONT_XS.render(ag.stage[:3].upper(), True, scol)
        screen_surf.blit(sbadge, (px + pw - 68, ry + 2))

        wcol = (180, 200, 240) if ag.winters > 0 else (60, 68, 90)
        _label(screen_surf, px + pw - 34, ry + 2, f"❄{ag.winters}", FONT_XS, wcol)

        hp      = getattr(ag, "health", 100)
        bar_w   = int(row_rect_w * hp / 100)
        bar_col = (60, 200, 80) if hp > 60 else (220, 170, 40) if hp > 30 else (220, 60, 60)
        pygame.draw.rect(screen_surf, bar_col,
                         (row_rect_x, ry + row_h - 3, bar_w, 2))

        if hovered:
            clicked = ag

    return clicked


# ── main HUD draw ─────────────────────────────────────────────

def draw_hud(screen_surf, world, agents, selected_agent, pop_history,
             sim_speed_ref, fps=0, cam_yaw=0.0):
    ensure_fonts()
    W, H = screen_surf.get_size()

    live = [a for a in agents if not a.is_dead]

    # ══ TOP BAR ══════════════════════════════════════════════
    bar_h = 32
    _panel(screen_surf, 0, 0, W, bar_h, alpha=210, border=False)
    pygame.draw.line(screen_surf, (50, 60, 85), (0, bar_h), (W, bar_h), 1)

    x = 10

    season  = world.season_name
    scol    = SEASON_COLORS.get(season, (200, 200, 200))
    day_pct = (world.day_number % SEASON_DAYS) / max(1, SEASON_DAYS)
    pill_w  = 140
    pygame.draw.rect(screen_surf, (35, 38, 50), (x, 5, pill_w, 22), border_radius=4)
    pygame.draw.rect(screen_surf, scol,         (x, 5, int(pill_w * day_pct), 22), border_radius=4)
    pygame.draw.rect(screen_surf, scol,         (x, 5, pill_w, 22), 1, border_radius=4)
    _label(screen_surf, x + 5, 9,
           f"{season.upper()[:3]}  Day {world.day_number % SEASON_DAYS + 1:02d}",
           FONT_TITLE, (240, 242, 250))
    x += pill_w + 10

    tod  = world.time_of_day
    hour = int(tod * 24)
    ampm = "AM" if hour < 12 else "PM"
    h12  = hour % 12 or 12
    tcol = (240, 195, 90) if world.is_day else (130, 140, 195)
    icon = "☀" if world.is_day else "☾"
    _label(screen_surf, x, 9, f"{icon} {h12:02d}{ampm}", FONT_TITLE, tcol)
    x += 90

    wcol = {"clear": (240,220,80), "rain": (100,160,255),
            "storm": (180,100,255), "blizzard": (200,220,255), "cloudy": (180,190,200)}
    wc = wcol.get(world.weather, (200, 200, 200))
    _label(screen_surf, x, 9, world.weather.upper(), FONT_TITLE, wc)
    x += 90

    _label(screen_surf, x, 9, f"👥 {len(live)}", FONT_TITLE, (120, 225, 130))
    x += 70

    _label(screen_surf, x, 9, f"⚡{sim_speed_ref[0]:.0f}×", FONT_TITLE, (235, 175, 75))

    fps_col = (80,225,80) if fps >= 50 else (235,195,60) if fps >= 30 else (225,80,60)
    _label(screen_surf, W - 75,  9, f"FPS {fps:3.0f}",       FONT_TITLE, fps_col)
    _label(screen_surf, W - 160, 9, f"Day {world.day_number}", FONT_SM,   (140, 145, 165))

    # ══ POPULATION CHART (bottom-left) ═══════════════════════
    ch_w, ch_h = 190, 68
    ch_x, ch_y = 10, H - ch_h - 10
    _panel(screen_surf, ch_x - 4, ch_y - 18, ch_w + 8, ch_h + 22)
    _label(screen_surf, ch_x, ch_y - 16, "POPULATION", FONT_XS, (100, 110, 135))
    _label(screen_surf, ch_x + ch_w - 40, ch_y - 16,
           f"max {MAX_AGENTS}", FONT_XS, (80, 88, 110))

    if pop_history:
        max_p = max(max(pop_history), 1)
        pts   = []
        hist  = pop_history[-ch_w:]
        for i, p in enumerate(hist):
            px2 = ch_x + int(i * (ch_w - 1) / max(len(hist) - 1, 1))
            py2 = ch_y + ch_h - 2 - int(p / max_p * (ch_h - 4))
            pts.append((px2, py2))
        if len(pts) >= 2:
            fill_pts  = [(ch_x, ch_y + ch_h - 2)] + pts + [(pts[-1][0], ch_y + ch_h - 2)]
            fill_surf = pygame.Surface((ch_w, ch_h), pygame.SRCALPHA)
            adj = [(px2 - ch_x, py2 - ch_y) for (px2, py2) in fill_pts]
            pygame.draw.polygon(fill_surf, (70, 200, 90, 40), adj)
            screen_surf.blit(fill_surf, (ch_x, ch_y))
            pygame.draw.lines(screen_surf, (80, 220, 90), False, pts, 2)
        _label(screen_surf, ch_x + 2, ch_y + ch_h - 14,
               str(pop_history[-1]), FONT_SM, (120, 230, 120))

    # ══ COMPASS ══════════════════════════════════════════════
    _draw_compass(screen_surf, 10 + ch_w + 14, H - 42, cam_yaw)

    # ══ CONTROLS HINT ════════════════════════════════════════
    hint = "WASD/Arrows:pan  Scroll:zoom  Q/E:orbit  R:reset  Space:pause  F:fire"
    ht   = FONT_XS.render(hint, True, (70, 78, 105))
    screen_surf.blit(ht, ((W - ht.get_width()) // 2, H - 14))

    # ══ SELECTED AGENT PANEL (right side) ════════════════════
    if selected_agent and not selected_agent.is_dead:
        _draw_agent_panel(screen_surf, selected_agent, W, H)

    # ══ MINI EVENT LOG ════════════════════════════════════════
    if selected_agent and selected_agent.event_log:
        log_x = W - 235
        log_y = H - 130
        _panel(screen_surf, log_x - 4, log_y - 4,
               230, 14 + len(selected_agent.event_log[-7:]) * 14)
        for i, (aid, reward) in enumerate(selected_agent.event_log[-7:]):
            acol = ACTION_COLS.get(aid, (180, 180, 180))
            rcol = (80, 225, 80) if reward > 0 else (225, 80, 80)
            _label(screen_surf, log_x, log_y + i * 14,
                   f"{'▲' if reward>0 else '▼'} {ACTION_NAMES[aid]:<12}", FONT_XS, acol)
            _label(screen_surf, log_x + 155, log_y + i * 14,
                   f"{reward:+.2f}", FONT_XS, rcol)


# ── agent detail panel ────────────────────────────────────────

def _draw_agent_panel(surf, ag, W, H):
    ensure_fonts()
    pw, ph = 215, 330
    px = W - pw - 8
    py = 40
    _panel(surf, px, py, pw, ph, alpha=215)

    gen_col = (min(255, 90 + ag.generation * 25), 200, 210)
    _label(surf, px + 8, py + 6, f"{ag.id}", FONT_TITLE, gen_col)
    stage_col = {"baby": (240,210,70), "adult": (200,215,235), "elder": (175,170,200)}
    _label(surf, px + 65, py + 6, f"[{ag.stage.upper()}]", FONT_TITLE,
           stage_col.get(ag.stage, (200, 200, 200)))
    g_icon = "♂" if ag.gender == "male" else "♀"
    g_col  = (110, 170, 240) if ag.gender == "male" else (240, 120, 185)
    _label(surf, px + 155, py + 6, g_icon, FONT_TITLE, g_col)
    # Camera-follow indicator (camera always follows selected colonist)
    _label(surf, px + 175, py + 6, "📷", FONT_XS, (120, 220, 255))
    _label(surf, px + 8, py + 22,
           f"Gen {ag.generation}  Winters {ag.winters}", FONT_XS, (130, 135, 160))

    # ── Exploration trait bar ─────────────────────────────────
    exp  = getattr(ag, "exploration", 0.3)
    ex_w = pw - 16
    ex_h = 10
    ex_y = py + 34
    pygame.draw.rect(surf, (25, 27, 35), (px + 8, ex_y, ex_w, ex_h))
    fill_w = int(ex_w * exp)
    # Gradient: blue (homebody) → orange (explorer)
    exp_r = int(60  + exp * 195)
    exp_g = int(120 - exp * 60)
    exp_b = int(220 - exp * 170)
    pygame.draw.rect(surf, (exp_r, exp_g, exp_b), (px + 8, ex_y, fill_w, ex_h))
    pygame.draw.rect(surf, (65, 70, 95), (px + 8, ex_y, ex_w, ex_h), 1)
    _label(surf, px + 10, ex_y, f"EXPLORE {exp:.2f}", FONT_XS, (200, 200, 220))

    by = py + 48
    for stat, col in VITAL_COLS.items():
        val = getattr(ag, stat, 100)
        _bar(surf, px + 8, by, pw - 16, 15, val, col, stat[:3].upper(), FONT_XS)
        by += 18

    if ag.crafting_item and ag.craft_remaining > 0:
        from ai.agent import CRAFT_TIME
        total = CRAFT_TIME.get(ag.crafting_item, 3.0)
        pct   = 1.0 - ag.craft_remaining / max(total, 0.01)
        pygame.draw.rect(surf, (35, 38, 50),    (px+8, by, pw-16, 12))
        pygame.draw.rect(surf, (200, 150, 60),  (px+8, by, int((pw-16)*pct), 12))
        pygame.draw.rect(surf, (80, 85, 110),   (px+8, by, pw-16, 12), 1)
        _label(surf, px + 10, by, f"Crafting {ag.crafting_item}", FONT_XS, (235, 200, 110))
        by += 14

    by += 2
    aname = ACTION_NAMES[ag.current_action] if ag.current_action is not None else "?"
    acol  = ACTION_COLS.get(ag.current_action, (180, 180, 180))
    pygame.draw.rect(surf, (*acol, 60), (px+8, by, pw-16, 16))
    pygame.draw.rect(surf, acol,        (px+8, by, pw-16, 16), 1)
    _label(surf, px + 10, by + 1, f"▶ {aname}", FONT_XS, acol)
    by += 20

    _label(surf, px + 8, by, "INVENTORY", FONT_XS, (90, 98, 125))
    by += 13
    for itype, idata in list(ag.inventory.items())[:6]:
        qty   = idata["quantity"]
        fresh = idata.get("freshness")
        col   = (225, 100, 80) if (fresh is not None and fresh < 30) else (200, 195, 150)
        _label(surf, px + 8, by, f"  {itype[:14]:<14} ×{qty}", FONT_XS, col)
        by += 12

    if len(ag.inventory) > 6:
        _label(surf, px + 8, by, f"  +{len(ag.inventory)-6} more…", FONT_XS, (90, 95, 120))
        by += 12

    by += 4
    if by + 78 < py + ph - 4:
        _draw_radar(surf, px + 8, by, pw - 16, 70, ag.weights)


# ── radar / compass ───────────────────────────────────────────

def _draw_radar(surf, x, y, w, h, weights):
    cx, cy = x + w // 2, y + h // 2
    r  = min(w, h) // 2 - 6
    n  = NUM_ACTIONS
    pts_outer, pts_val = [], []
    for i in range(n):
        angle = 2 * math.pi * i / n - math.pi / 2
        ox = cx + int(r * math.cos(angle))
        oy = cy + int(r * math.sin(angle))
        pts_outer.append((ox, oy))
        v  = weights[i] / 3.0
        pts_val.append((cx + int(r * v * math.cos(angle)),
                        cy + int(r * v * math.sin(angle))))
    if len(pts_outer) >= 3:
        pygame.draw.polygon(surf, (38, 42, 60), pts_outer)
        for ring_r in [r//3, 2*r//3, r]:
            ring_pts = [(cx + int(ring_r * math.cos(2*math.pi*i/n - math.pi/2)),
                         cy + int(ring_r * math.sin(2*math.pi*i/n - math.pi/2)))
                        for i in range(n)]
            pygame.draw.polygon(surf, (55, 62, 85), ring_pts, 1)
        pygame.draw.polygon(surf, (75, 170, 115, 90), pts_val)
        pygame.draw.polygon(surf, (95, 205, 140), pts_val, 1)
        pygame.draw.polygon(surf, (65, 72, 100), pts_outer, 1)
        for i, (ox, oy) in enumerate(pts_val):
            pygame.draw.circle(surf, ACTION_COLS.get(i, (140, 145, 165)), (ox, oy), 2)


def _draw_compass(surf, cx, cy, yaw_deg):
    r  = 18
    pygame.draw.circle(surf, (18, 20, 30, 180), (cx, cy), r)
    pygame.draw.circle(surf, (55, 65, 95), (cx, cy), r, 1)
    yr = math.radians(-yaw_deg)
    nx = cx + int(r * 0.7 * math.sin(yr))
    ny = cy - int(r * 0.7 * math.cos(yr))
    pygame.draw.line(surf, (220, 80, 70), (cx, cy), (nx, ny), 2)
    sx = cx - int(r * 0.5 * math.sin(yr))
    sy = cy + int(r * 0.5 * math.cos(yr))
    pygame.draw.line(surf, (100, 110, 135), (cx, cy), (sx, sy), 2)
    t = FONT_XS.render("N", True, (220, 80, 70))
    surf.blit(t, (nx - 4, ny - 6))