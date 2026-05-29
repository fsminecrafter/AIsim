"""
Pre-launch settings menu.
Renders a styled pygame window where the user configures:
  - Starting population (0-1000)
  - Neural size (hidden units)
  - Terrain seed + Randomize button
  - Seasons toggle + starting season
  - GPU availability display
  - FPS meter toggle

Returns a dict of settings, or None if the user quits.
"""

import pygame
import random
import math

# ── colours ───────────────────────────────────────────────────
BG          = (12,  14,  18)
PANEL       = (20,  24,  32)
BORDER      = (40,  50,  70)
ACCENT      = (80, 160, 255)
ACCENT_DIM  = (40,  90, 160)
TEXT_HI     = (230, 235, 245)
TEXT_MID    = (150, 160, 180)
TEXT_LO     = ( 80,  90, 110)
GREEN       = ( 60, 200, 100)
RED         = (220,  70,  70)
GOLD        = (230, 180,  50)
WARNING     = (230, 140,  40)

SEASON_COLS = {
    "spring": ( 80, 200, 120),
    "summer": (240, 200,  60),
    "autumn": (210, 120,  40),
    "winter": (160, 190, 230),
}

W, H = 780, 660


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Slider:
    def __init__(self, x, y, w, lo, hi, value, label, fmt="{:.0f}", step=1):
        self.rect  = pygame.Rect(x, y, w, 4)
        self.lo, self.hi = lo, hi
        self.value = value
        self.label = label
        self.fmt   = fmt
        self.step  = step
        self.dragging = False
        self._knob_r = 9

    def _frac(self):
        return (self.value - self.lo) / max(self.hi - self.lo, 1e-9)

    def knob_pos(self):
        return (int(self.rect.x + self._frac() * self.rect.w), self.rect.centery)

    def draw(self, surf, font_sm, font_lbl):
        kx, ky = self.knob_pos()
        # track bg
        pygame.draw.rect(surf, BORDER, self.rect, border_radius=2)
        # filled portion
        fill = pygame.Rect(self.rect.x, self.rect.y, kx - self.rect.x, self.rect.h)
        pygame.draw.rect(surf, ACCENT, fill, border_radius=2)
        # knob
        col = TEXT_HI if self.dragging else ACCENT
        pygame.draw.circle(surf, col, (kx, ky), self._knob_r)
        pygame.draw.circle(surf, BG,  (kx, ky), self._knob_r - 3)
        # label + value
        lbl = font_lbl.render(self.label, True, TEXT_MID)
        surf.blit(lbl, (self.rect.x, self.rect.y - 22))
        val_s = self.fmt.format(self.value)
        val_t = font_sm.render(val_s, True, TEXT_HI)
        surf.blit(val_t, (self.rect.right - val_t.get_width(), self.rect.y - 22))

    def handle_event(self, event):
        kx, ky = self.knob_pos()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if math.hypot(event.pos[0]-kx, event.pos[1]-ky) < self._knob_r + 4:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            frac = _clamp((event.pos[0] - self.rect.x) / self.rect.w, 0, 1)
            raw  = self.lo + frac * (self.hi - self.lo)
            self.value = round(raw / self.step) * self.step
            self.value = _clamp(self.value, self.lo, self.hi)


class Toggle:
    def __init__(self, x, y, label, value=True):
        self.x, self.y = x, y
        self.label = label
        self.value = value
        self.rect  = pygame.Rect(x, y, 48, 26)

    def draw(self, surf, font):
        col  = GREEN if self.value else BORDER
        pygame.draw.rect(surf, col, self.rect, border_radius=13)
        kx = self.rect.x + (34 if self.value else 14)
        pygame.draw.circle(surf, TEXT_HI, (kx, self.rect.centery), 10)
        lbl = font.render(self.label, True, TEXT_MID)
        surf.blit(lbl, (self.rect.right + 12, self.rect.y + 4))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value
                return True
        return False


def _draw_seed_box(surf, x, y, w, seed_str, active, font, font_lbl):
    col_border = ACCENT if active else BORDER
    pygame.draw.rect(surf, PANEL, (x, y, w, 36), border_radius=6)
    pygame.draw.rect(surf, col_border, (x, y, w, 36), 1, border_radius=6)
    lbl = font_lbl.render("Terrain seed", True, TEXT_MID)
    surf.blit(lbl, (x, y - 22))
    txt = font.render(seed_str + ("|" if active and (pygame.time.get_ticks()//500)%2==0 else ""), True, TEXT_HI)
    surf.blit(txt, (x + 10, y + 8))


def _button(surf, rect, label, font, col=ACCENT, hover=False):
    bc = tuple(min(255, c + 20) for c in col) if hover else col
    pygame.draw.rect(surf, bc, rect, border_radius=6)
    t = font.render(label, True, BG if sum(col) > 300 else TEXT_HI)
    surf.blit(t, t.get_rect(center=rect.center))


def _gpu_badge(surf, x, y, available, font):
    col  = GREEN if available else RED
    text = "GPU  OpenGL 3.3" if available else "GPU  not detected  —  CPU fallback"
    icon = "✓" if available else "✗"
    r = pygame.Rect(x, y, 280, 30)
    pygame.draw.rect(surf, (*col, 40), r, border_radius=6)
    pygame.draw.rect(surf, col, r, 1, border_radius=6)
    t = font.render(f"{icon}  {text}", True, col)
    surf.blit(t, (x + 10, y + 6))


def run_menu(gpu_available=False):
    pygame.init()
    pygame.display.set_caption("Survival Simulation — Setup")
    screen = pygame.display.set_mode((W, H))
    clock  = pygame.time.Clock()

    try:
        font_title = pygame.font.SysFont("monospace", 26, bold=True)
        font_h2    = pygame.font.SysFont("monospace", 15, bold=True)
        font_body  = pygame.font.SysFont("monospace", 13)
        font_sm    = pygame.font.SysFont("monospace", 13, bold=True)
        font_btn   = pygame.font.SysFont("monospace", 13, bold=True)
    except Exception:
        font_title = font_h2 = font_body = font_sm = font_btn = pygame.font.SysFont(None, 16)

    # ── widgets ──────────────────────────────────────────────
    PAD  = 44
    COL2 = W // 2 + 10

    sl_pop = Slider(PAD, 130, W - PAD*2, 0, 1000, 12, "Starting population", "{:.0f}", step=1)
    sl_neu = Slider(PAD, 220, W - PAD*2, 4,   64, 12, "Neural size  (action weight vector)", "{:.0f}", step=4)

    seed_val    = str(random.randint(100, 9999))
    seed_active = False

    tog_seasons  = Toggle(PAD, 320, "Enable seasons")
    tog_fps      = Toggle(PAD, 370, "Show FPS meter")
    tog_debug    = Toggle(PAD, 420, "Show agent debug overlay")

    season_names = ["spring", "summer", "autumn", "winter"]
    sel_season   = 0  # index

    rand_btn   = pygame.Rect(W - PAD - 120, 296, 120, 32)
    launch_btn = pygame.Rect(W//2 - 100, H - 68, 200, 44)

    season_btns = []
    for i, s in enumerate(season_names):
        bx = PAD + i * ((W - PAD*2) // 4)
        season_btns.append(pygame.Rect(bx, 490, (W - PAD*2)//4 - 8, 34))

    hover_rand   = False
    hover_launch = False
    hover_season = [-1]

    # ── particle bg ──────────────────────────────────────────
    particles = [(random.uniform(0, W), random.uniform(0, H),
                  random.uniform(0.3, 1.2), random.uniform(0, math.pi*2))
                 for _ in range(60)]

    running = True
    result  = None

    while running:
        mx, my = pygame.mouse.get_pos()
        hover_rand   = rand_btn.collidepoint(mx, my)
        hover_launch = launch_btn.collidepoint(mx, my)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None
                if seed_active:
                    if event.key == pygame.K_BACKSPACE:
                        seed_val = seed_val[:-1]
                    elif event.key in (pygame.K_RETURN, pygame.K_TAB):
                        seed_active = False
                    elif event.unicode.isdigit() and len(seed_val) < 9:
                        seed_val += event.unicode
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # seed box click
                seed_box = pygame.Rect(PAD, 296, W - PAD*2 - 130, 36)
                seed_active = seed_box.collidepoint(event.pos)
                # randomize button
                if rand_btn.collidepoint(event.pos):
                    seed_val = str(random.randint(1, 999999))
                # launch
                if launch_btn.collidepoint(event.pos):
                    result = {
                        "population":    int(sl_pop.value),
                        "neural_size":   int(sl_neu.value),
                        "terrain_seed":  int(seed_val) if seed_val else 42,
                        "seasons":       tog_seasons.value,
                        "start_season":  sel_season if tog_seasons.value else 1,
                        "fps_meter":     tog_fps.value,
                        "debug_overlay": tog_debug.value,
                        "gpu":           gpu_available,
                    }
                    running = False
                # season select
                if tog_seasons.value:
                    for i, br in enumerate(season_btns):
                        if br.collidepoint(event.pos):
                            sel_season = i
                # toggles
                tog_seasons.handle_event(event)
                tog_fps.handle_event(event)
                tog_debug.handle_event(event)

            sl_pop.handle_event(event)
            sl_neu.handle_event(event)

        # ── draw ─────────────────────────────────────────────
        screen.fill(BG)

        # subtle particle layer
        t = pygame.time.get_ticks() / 1000.0
        for i, (px, py, spd, phase) in enumerate(particles):
            ny = (py - spd * 0.4) % H
            particles[i] = (px, ny, spd, phase)
            alpha = int(18 + 12 * math.sin(t * spd + phase))
            pygame.draw.circle(screen, (*TEXT_LO, alpha), (int(px), int(ny)), 1)

        # title bar
        pygame.draw.rect(screen, PANEL, (0, 0, W, 72))
        pygame.draw.line(screen, BORDER, (0, 72), (W, 72), 1)
        ttl = font_title.render("SURVIVAL  SIMULATION", True, TEXT_HI)
        screen.blit(ttl, (PAD, 20))
        sub = font_body.render("GPU Edition  ·  Pre-launch setup", True, TEXT_LO)
        screen.blit(sub, (PAD, 48))

        # GPU badge
        _gpu_badge(screen, W - PAD - 285, 22, gpu_available, font_body)

        # sliders
        sl_pop.draw(screen, font_sm, font_body)
        sl_neu.draw(screen, font_sm, font_body)

        # neural size hint
        hint_col = WARNING if sl_neu.value > 32 else TEXT_LO
        hint = font_body.render(
            "larger = richer behaviour  /  slower per-agent tick" if sl_neu.value > 12 else
            "default = 12  (original)",
            True, hint_col)
        screen.blit(hint, (PAD, 232))

        # seed row
        seed_box_r = pygame.Rect(PAD, 296, W - PAD*2 - 130, 36)
        _draw_seed_box(screen, PAD, 296, W - PAD*2 - 130, seed_val, seed_active, font_sm, font_body)
        _button(screen, rand_btn, "Randomize", font_btn, col=ACCENT_DIM, hover=hover_rand)

        # toggles
        lbl_opts = font_h2.render("OPTIONS", True, TEXT_LO)
        screen.blit(lbl_opts, (PAD, 300 + 50))
        pygame.draw.line(screen, BORDER, (PAD + 80, 312 + 50), (W - PAD, 312 + 50), 1)

        tog_seasons.draw(screen, font_body)
        tog_fps.draw(screen, font_body)
        tog_debug.draw(screen, font_body)

        # starting season
        lbl_s = font_h2.render("STARTING SEASON", True, TEXT_LO)
        screen.blit(lbl_s, (PAD, 464))
        pygame.draw.line(screen, BORDER, (PAD + 148, 476), (W - PAD, 476), 1)

        for i, (s_name, br) in enumerate(zip(season_names, season_btns)):
            sc     = SEASON_COLS[s_name]
            active = (i == sel_season)
            dim    = not tog_seasons.value
            alpha_col = tuple(c // 3 for c in sc) if dim else sc
            bg_col = alpha_col if active else PANEL
            pygame.draw.rect(screen, bg_col, br, border_radius=6)
            border_col = alpha_col if active else BORDER
            pygame.draw.rect(screen, border_col, br, 1, border_radius=6)
            txt_col = TEXT_HI if (active and not dim) else TEXT_LO
            st = font_body.render(s_name.upper(), True, txt_col)
            screen.blit(st, st.get_rect(center=br.center))

        if not tog_seasons.value:
            note = font_body.render("seasons disabled — always summer", True, WARNING)
            screen.blit(note, (PAD, 534))

        # launch button
        lc = (100, 200, 100) if hover_launch else GREEN
        pygame.draw.rect(screen, lc, launch_btn, border_radius=8)
        lt = font_title.render("LAUNCH", True, BG)
        screen.blit(lt, lt.get_rect(center=launch_btn.center))

        # bottom hint
        hint2 = font_body.render("ESC to quit  ·  click LAUNCH to start", True, TEXT_LO)
        screen.blit(hint2, hint2.get_rect(centerx=W//2, y=H - 18))

        pygame.display.flip()
        clock.tick(60)

    return result