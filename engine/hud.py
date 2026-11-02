"""
HUD — draws stats overlay using pygame surfaces, composited over the GL frame.

v3 additions:
  * FIX 3 — Sleep icon above sleeping colonists (drawn in draw_agent_icons)
  * FIX 4 — Pregnancy icon above pregnant colonists + inspector status
  * FIX 5 — Heart effect at colonist position when reproducing
  * FIX 6 — Friends list in inspector; returns clicked friend ID for
             selection in main.py
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
    12: (100, 140, 220),   # A_SLEEP — blue-grey
    13: (200, 220, 140),   # A_RECREATE — light green
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


# ── world-space → screen coords helper ───────────────────────

def world_to_screen(wx, wz, renderer):
    """
    Project 3-D world position (wx, wy=0, wz) to 2-D screen pixels.
    Returns (sx, sy) or None if behind the camera.
    Used to place icons above agents.
    """
    import numpy as np
    view, mvp = renderer._get_view_proj()
    # Head position: lift a bit above the tile
    pos  = np.array([wx, 1.8, wz, 1.0], dtype=np.float32)
    clip = mvp @ pos
    if clip[3] <= 0:
        return None
    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]
    sx = int((ndc_x + 1.0) * 0.5 * renderer.screen_w)
    sy = int((1.0 - (ndc_y + 1.0) * 0.5) * renderer.screen_h)
    return (sx, sy)


# ── per-agent icon rendering (FIX 3 / 4 / 5) ─────────────────

def draw_agent_icons(screen_surf, agents, renderer):
    """
    Draw sleep / pregnancy / heart icons above live agents in world-space.
    This must be called after the GL render and before pygame.display.flip().
    """
    ensure_fonts()
    for ag in agents:
        if ag.is_dead:
            continue

        wx = ag.x * TILE_SIZE + TILE_SIZE / 2
        wz = ag.y * TILE_SIZE + TILE_SIZE / 2
        screen_pos = world_to_screen(wx, wz, renderer)
        if screen_pos is None:
            continue
        sx, sy = screen_pos

        # Clip to screen
        if sx < -20 or sx > screen_surf.get_width() + 20:
            continue
        if sy < -20 or sy > screen_surf.get_height() + 20:
            continue

        offset_y = sy - 22   # baseline above head

        icons = []

        # FIX 3 — Sleep icon
        if getattr(ag, "sleeping", False):
            icons.append(("😴", (120, 160, 240)))

        # FIX 4 — Pregnancy icon
        if ag.preg_timer is not None and ag.preg_timer > 0:
            icons.append(("🤰", (240, 160, 200)))

        # FIX 5 — Heart effect (appears on both partners for a few seconds)
        if getattr(ag, "heart_effect_timer", 0) > 0:
            # Float the heart upward over its lifetime
            total   = 3.0  # HEART_EFFECT_DURATION
            frac    = ag.heart_effect_timer / total
            float_y = int((1.0 - frac) * 18)   # moves up as frac drops
            icon_surf = FONT_TITLE.render("❤", True, (255, 80, 120))
            alpha     = int(255 * frac)
            icon_surf.set_alpha(alpha)
            screen_surf.blit(icon_surf, (sx - icon_surf.get_width() // 2,
                                         offset_y - 14 - float_y))

        # Draw stacked icons left-to-right centred on sx
        total_w = len(icons) * 16
        ix = sx - total_w // 2
        for text, col in icons:
            try:
                t = FONT_SM.render(text, True, col)
            except Exception:
                t = FONT_XS.render("?", True, col)
            screen_surf.blit(t, (ix, offset_y))
            ix += t.get_width() + 2


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

        # Show sleep / pregnancy icon in list
        extra = ""
        if getattr(ag, "sleeping", False):
            extra += "😴"
        if ag.preg_timer is not None:
            extra += "🤰"
        name_txt = str(ag.id)[:8] + (extra if extra else "")
        _label(screen_surf, px + 38, ry + 2, name_txt, FONT_XS, ncol)

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
    # Returns the id of any friend that was clicked (or None).
    clicked_friend_id = None
    if selected_agent and not selected_agent.is_dead:
        clicked_friend_id = _draw_agent_panel(
            screen_surf, selected_agent, W, H,
            agents, pygame.mouse.get_pos(),
        )

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

    return clicked_friend_id   # NEW: propagate to main.py for friend-click selection


# ── agent detail panel ────────────────────────────────────────

def _draw_agent_panel(surf, ag, W, H, agents=None, mouse_pos=None):
    """
    Draws the selected-agent inspector.
    Returns the agent ID of a clicked friend entry, or None.
    (FIX 6 — friend list is clickable)
    """
    ensure_fonts()
    pw, ph = 215, 400   # taller to fit friend list
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
    _label(surf, px + 175, py + 6, "📷", FONT_XS, (120, 220, 255))
    _label(surf, px + 8, py + 22,
           f"Gen {ag.generation}  Winters {ag.winters}", FONT_XS, (130, 135, 160))

    # ── FIX 4 — Pregnancy status ──────────────────────────────
    preg_line = ""
    if ag.preg_timer is not None and ag.preg_timer > 0:
        preg_pct = int((1.0 - ag.preg_timer / 30.0) * 100)
        preg_line = f"🤰 Pregnant  {preg_pct}%"
    elif ag.postpartum_timer > 0:
        preg_line = "🍼 Postpartum rest"

    # ── FIX 3 — Sleep status ──────────────────────────────────
    sleep_line = ""
    if getattr(ag, "sleeping", False):
        remaining = int(getattr(ag, "sleep_timer", 0))
        sleep_line = f"😴 Sleeping  {remaining}s left"

    row_y = py + 34

    # Exploration bar
    exp  = getattr(ag, "exploration", 0.3)
    ex_w = pw - 16
    ex_h = 10
    pygame.draw.rect(surf, (25, 27, 35), (px + 8, row_y, ex_w, ex_h))
    fill_w = int(ex_w * exp)
    exp_r  = int(60  + exp * 195)
    exp_g  = int(120 - exp * 60)
    exp_b  = int(220 - exp * 170)
    pygame.draw.rect(surf, (exp_r, exp_g, exp_b), (px + 8, row_y, fill_w, ex_h))
    pygame.draw.rect(surf, (65, 70, 95), (px + 8, row_y, ex_w, ex_h), 1)
    _label(surf, px + 10, row_y, f"EXPLORE {exp:.2f}", FONT_XS, (200, 200, 220))
    row_y += 12

    # Status lines (pregnancy / sleep)
    if preg_line:
        _label(surf, px + 8, row_y, preg_line, FONT_XS, (255, 160, 200))
        row_y += 13
    if sleep_line:
        _label(surf, px + 8, row_y, sleep_line, FONT_XS, (120, 160, 240))
        row_y += 13

    # Vitals bars
    for stat, col in VITAL_COLS.items():
        val = getattr(ag, stat, 100)
        _bar(surf, px + 8, row_y, pw - 16, 15, val, col, stat[:3].upper(), FONT_XS)
        row_y += 18

    if ag.crafting_item and ag.craft_remaining > 0:
        from ai.agent import CRAFT_TIME
        total = CRAFT_TIME.get(ag.crafting_item, 3.0)
        pct   = 1.0 - ag.craft_remaining / max(total, 0.01)
        pygame.draw.rect(surf, (35, 38, 50),    (px+8, row_y, pw-16, 12))
        pygame.draw.rect(surf, (200, 150, 60),  (px+8, row_y, int((pw-16)*pct), 12))
        pygame.draw.rect(surf, (80, 85, 110),   (px+8, row_y, pw-16, 12), 1)
        _label(surf, px + 10, row_y, f"Crafting {ag.crafting_item}", FONT_XS, (235, 200, 110))
        row_y += 14

    row_y += 2
    aname = ACTION_NAMES[ag.current_action] if ag.current_action is not None else "?"
    acol  = ACTION_COLS.get(ag.current_action, (180, 180, 180))
    pygame.draw.rect(surf, (*acol, 60), (px+8, row_y, pw-16, 16))
    pygame.draw.rect(surf, acol,        (px+8, row_y, pw-16, 16), 1)
    _label(surf, px + 10, row_y + 1, f"▶ {aname}", FONT_XS, acol)
    row_y += 20

    # Inventory
    _label(surf, px + 8, row_y, "INVENTORY", FONT_XS, (90, 98, 125))
    row_y += 13
    for itype, idata in list(ag.inventory.items())[:5]:
        qty   = idata["quantity"]
        fresh = idata.get("freshness")
        col   = (225, 100, 80) if (fresh is not None and fresh < 30) else (200, 195, 150)
        _label(surf, px + 8, row_y, f"  {itype[:14]:<14} ×{qty}", FONT_XS, col)
        row_y += 12

    if len(ag.inventory) > 5:
        _label(surf, px + 8, row_y, f"  +{len(ag.inventory)-5} more…", FONT_XS, (90, 95, 120))
        row_y += 12

    # ── FIX 6 — Friends list ──────────────────────────────────
    clicked_friend_id = _draw_friends_list(
        surf, ag, agents, px + 8, row_y, pw - 16, H - (row_y + py),
        mouse_pos,
    )

    # Radar (only if space remains)
    row_y += _friends_list_height(ag)
    if row_y + 78 < py + ph - 4:
        _draw_radar(surf, px + 8, row_y + 4, pw - 16, 70, ag.weights)

    return clicked_friend_id


def _friends_list_height(ag):
    """Estimate pixel height the friends list will consume."""
    friends = getattr(ag, "friends", {})
    # family entries + non-family friends, capped at 6 displayed
    n = min(len(friends) + len(getattr(ag, "parent_ids", [])) + len(getattr(ag, "children_ids", [])), 6)
    if n == 0:
        return 14   # just header
    return 14 + n * 14 + 4


def _draw_friends_list(surf, ag, agents, x, y, w, max_h, mouse_pos):
    """
    FIX 6 — Render friend/family list in the inspector.
    Returns clicked friend agent ID, or None.
    """
    ensure_fonts()
    if agents is None:
        return None

    # Build display list: family first, then non-family friends sorted by score
    entries = []  # (agent_id, label, score, is_family)

    # Parent links
    for pid in getattr(ag, "parent_ids", []):
        entries.append((pid, "Parent", 999.0, True))

    # Child links
    for cid in getattr(ag, "children_ids", []):
        entries.append((cid, "Child", 999.0, True))

    # Non-family friends
    friends = getattr(ag, "friends", {})
    sorted_friends = sorted(friends.items(), key=lambda kv: -kv[1])
    for fid, score in sorted_friends:
        if fid not in getattr(ag, "parent_ids", []) and fid not in getattr(ag, "children_ids", []):
            entries.append((fid, f"{score:.0f}", score, False))

    # Cap displayed entries
    entries = entries[:6]

    # Build lookup for display names
    id_to_agent = {}
    if agents:
        for a in agents:
            id_to_agent[a.id] = a

    header_h = 14
    _label(surf, x, y, "FRIENDS / FAMILY", FONT_XS, (90, 98, 125))
    y += header_h

    if not entries:
        _label(surf, x + 4, y, "  (none yet)", FONT_XS, (60, 65, 85))
        return None

    clicked_id = None
    mx, my = mouse_pos if mouse_pos else (-1, -1)

    for entry_id, label, score, is_fam in entries:
        row_h = 14
        if y + row_h > surf.get_height() - 4:
            break

        other = id_to_agent.get(entry_id)
        dead  = (other is None or other.is_dead)

        # Row hover
        hovered = (x <= mx <= x + w and y <= my <= y + row_h)

        if hovered and not dead:
            hs = pygame.Surface((w, row_h), pygame.SRCALPHA)
            hs.fill((60, 90, 130, 110))
            surf.blit(hs, (x, y))
            clicked_id = entry_id   # return on any hover click (main checks button)

        # Colour coding
        if dead:
            name_col = (60, 65, 80)
            label_col = (50, 55, 70)
        elif is_fam:
            name_col  = (220, 200, 120)
            label_col = (160, 145, 80)
        else:
            # Colour by friendship tier
            if score >= 50:
                name_col = (130, 230, 130)
            elif score >= 25:
                name_col = (200, 220, 160)
            else:
                name_col = (160, 170, 180)
            label_col = (100, 110, 120)

        icon = "💀" if dead else ("👨‍👩‍👧" if is_fam else "🤝")
        gender_icon = ""
        if other and not dead:
            gender_icon = "♂" if other.gender == "male" else "♀"
            g_col = (110, 170, 240) if other.gender == "male" else (240, 120, 185)
        else:
            g_col = (100, 100, 100)

        name_str = str(entry_id)[:9]
        _label(surf, x + 2,  y, icon,                FONT_XS, (180, 180, 180))
        _label(surf, x + 16, y, gender_icon,          FONT_XS, g_col)
        _label(surf, x + 26, y, name_str,             FONT_XS, name_col)
        _label(surf, x + 96, y, label if is_fam else f"♥{label}", FONT_XS, label_col)

        if hovered and not dead:
            pygame.draw.rect(surf, (80, 130, 200), (x, y, w, row_h), 1)

        y += row_h

    return clicked_id


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
    
