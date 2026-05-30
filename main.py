#!/usr/bin/env python3
"""
SURVIVAL SIMULATION — 3D GPU Edition
=====================================
Full 3D perspective rendering with OpenGL 3.3 core via ModernGL.
Agents: capsules. Trees: cylinders + sphere canopy. Rocks: spheres.
Logs/planks: box meshes. Terrain heightmap. Weather particles.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math, random, time
import numpy as np
import pygame
import moderngl

from engine.config import *
from engine.renderer import Renderer
from engine.hud import draw_hud, draw_colonist_panel
from world.world import World
from ai.agent import Agent
from engine.menu import run_menu

# ─── init ─────────────────────────────────────────────────────
pygame.init()

menu_settings = run_menu(gpu_available=True)
if menu_settings is None:
    pygame.quit()
    sys.exit(0)

import engine.config as cfg
cfg.INITIAL_AGENTS = menu_settings["population"]

pygame.display.set_caption("Survival Simulation — 3D GPU Edition")
flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)

ctx = moderngl.create_context()

# HUD surface composited over GL
hud_surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
hud_tex  = ctx.texture((SCREEN_W, SCREEN_H), 4)
hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

HUD_VERT = """
#version 330 core
in vec2 in_vert;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = vec2(in_vert.x * 0.5 + 0.5, 0.5 - in_vert.y * 0.5);
}
"""
HUD_FRAG = """
#version 330 core
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 frag_color;
void main() { frag_color = texture(u_tex, v_uv); }
"""
hud_prog = ctx.program(vertex_shader=HUD_VERT, fragment_shader=HUD_FRAG)
fs_quad  = np.array([[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]], dtype=np.float32)
fs_vbo   = ctx.buffer(fs_quad.tobytes())
hud_vao  = ctx.vertex_array(hud_prog, [(fs_vbo, '2f', 'in_vert')])
hud_prog["u_tex"].value = 0

# ─── sim state ────────────────────────────────────────────────
world    = World(seed=menu_settings["terrain_seed"])
if hasattr(world, "enable_seasons"):
    world.enable_seasons  = menu_settings["seasons"]
    world.current_season  = menu_settings["start_season"]

agents   = []
renderer = Renderer(ctx, SCREEN_W, SCREEN_H)

def _spawn_agents(n):
    result = []
    for _ in range(n):
        x = random.uniform(WORLD_W * 0.2, WORLD_W * 0.8)
        y = random.uniform(WORLD_H * 0.2, WORLD_H * 0.8)
        kw = {}
        if "neural_size" in menu_settings:
            kw["hidden_size"] = menu_settings["neural_size"]
        a = Agent(x, y, **kw)
        a.inv_add("berry", 3, freshness=90.0)
        a.inv_add("rock",  2)
        result.append(a)
    return result

agents = _spawn_agents(cfg.INITIAL_AGENTS)
pop_history    = [len(agents)]
selected_agent = None
paused         = False
sim_speed_ref  = [SIM_SPEED]

clock     = pygame.time.Clock()
last_time = time.perf_counter()
frame_no  = 0

# Camera orbit mouse drag
_drag_active = False
_drag_last   = (0, 0)


def reset():
    global world, agents, pop_history, selected_agent
    seed = menu_settings["terrain_seed"] if menu_settings else random.randint(0, 9999)
    world = World(seed=seed)
    if hasattr(world, "enable_seasons"):
        world.enable_seasons = menu_settings["seasons"]
        world.current_season = menu_settings["start_season"]
    renderer._height_map = None   # rebuild height cache
    agents = _spawn_agents(cfg.INITIAL_AGENTS)
    pop_history    = [len(agents)]
    selected_agent = None


def _insert_newborns(agents, new_children):
    """
    Safely insert newborns into the population.

    Rules enforced here:
    - Never exceed MAX_AGENTS.
    - Never kill the mother by inserting a child (mother health is already
      guarded in tick_pregnancy, but we double-check here).
    - Children are inserted with valid state; we drop extras silently if
      the cap is already full (shouldn't happen often with pop pressure).
    """
    live_count = sum(1 for a in agents if not a.is_dead)
    inserted = 0
    for child in new_children:
        if live_count + inserted >= MAX_AGENTS:
            break  # population cap — drop this child quietly
        # Sanity-check child state so it cannot be born already dead
        if child.health <= 0:
            child.health = BABY_HEALTH_START
        if child.hunger <= 0:
            child.hunger = 70.0
        if child.thirst <= 0:
            child.thirst = 70.0
        child.is_dead = False  # guard against any stray flag
        agents.append(child)
        inserted += 1
    return agents


# ─── main loop ───────────────────────────────────────────────
running = True
while running:
    now      = time.perf_counter()
    real_dt  = min(now - last_time, 0.05)
    last_time = now
    fps_now  = clock.get_fps()

    # ── events ───────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if   event.key == pygame.K_ESCAPE:  running = False
            elif event.key == pygame.K_SPACE:   paused = not paused
            elif event.key == pygame.K_r:       reset()
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                sim_speed_ref[0] = min(50.0, sim_speed_ref[0] * 1.5)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                sim_speed_ref[0] = max(0.1, sim_speed_ref[0] / 1.5)
            elif event.key == pygame.K_f:
                mx, my = pygame.mouse.get_pos()
                wx, wy = renderer.screen_to_world(mx, my)
                tx, ty = int(wx / TILE_SIZE), int(wy / TILE_SIZE)
                if 0 <= ty < WORLD_H and 0 <= tx < WORLD_W:
                    world.start_fire(ty, tx)
            # Orbit with Q/E
            elif event.key == pygame.K_q:  renderer.orbit(-15, 0)
            elif event.key == pygame.K_e:  renderer.orbit( 15, 0)
            # Tilt with Z/X
            elif event.key == pygame.K_z:  renderer.orbit(0, -8)
            elif event.key == pygame.K_x:  renderer.orbit(0,  8)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mx, my = event.pos
                # ── colonist panel click ──────────────────────
                clicked_ag = draw_colonist_panel(hud_surf, mx, my, agents, selected_agent,
                                                 agent_panel_active=bool(selected_agent and not selected_agent.is_dead))
                if clicked_ag:
                    selected_agent = clicked_ag
                else:
                    # ── agent selection ───────────────────────
                    wx, wz = renderer.screen_to_world(mx, my)
                    # agent pick (tile coords)
                    tx = wx / TILE_SIZE
                    ty = wz / TILE_SIZE
                    best_d = 3.0
                    selected_agent = None
                    for ag in agents:
                        if ag.is_dead: continue
                        d = math.sqrt((ag.x - tx)**2 + (ag.y - ty)**2)
                        if d < best_d:
                            best_d = d
                            selected_agent = ag
            elif event.button == 2:
                _drag_active = True
                _drag_last   = event.pos
            elif event.button == 4:
                renderer.zoom_by(1.12)
            elif event.button == 5:
                renderer.zoom_by(1 / 1.12)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                _drag_active = False

        elif event.type == pygame.MOUSEMOTION:
            if _drag_active:
                dx = event.pos[0] - _drag_last[0]
                dy = event.pos[1] - _drag_last[1]
                renderer.orbit(dx * 0.5, dy * 0.4)
                _drag_last = event.pos

        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(event.size, flags)
            renderer.screen_w = event.w
            renderer.screen_h = event.h
            hud_surf = pygame.Surface(event.size, pygame.SRCALPHA)
            hud_tex  = ctx.texture(event.size, 4)
            hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ── keyboard pan ─────────────────────────────────────────
    keys = pygame.key.get_pressed()
    pan_spd = 6.0 * real_dt * (1.0 + renderer.cam_dist * 0.04)
    panning  = False
    if keys[pygame.K_LEFT]  or keys[pygame.K_a]: renderer.pan(-pan_spd, 0); panning = True
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: renderer.pan( pan_spd, 0); panning = True
    if keys[pygame.K_UP]    or keys[pygame.K_w]: renderer.pan(0, -pan_spd); panning = True
    if keys[pygame.K_DOWN]  or keys[pygame.K_s]: renderer.pan(0,  pan_spd); panning = True
    if panning:
        selected_agent = None

    # ── camera follow selected colonist ──────────────────────
    if selected_agent and not selected_agent.is_dead:
        target_wx = selected_agent.x * TILE_SIZE + TILE_SIZE / 2
        target_wz = selected_agent.y * TILE_SIZE + TILE_SIZE / 2
        follow_speed = min(1.0, 8.0 * real_dt)
        renderer.cam_target[0] += (target_wx - renderer.cam_target[0]) * follow_speed
        renderer.cam_target[2] += (target_wz - renderer.cam_target[2]) * follow_speed

    # ── simulation update ────────────────────────────────────
    if not paused:
        cfg.SIM_SPEED = sim_speed_ref[0]

        world.tick(real_dt)

        # Collect new children separately so we can do a safe insert
        new_children = []
        for ag in agents:
            ag.tick(world, agents, real_dt)
            # Pass sim_speed-scaled dt to pregnancy so it advances correctly
            child = ag.tick_pregnancy(world, agents, real_dt * sim_speed_ref[0])
            if child is not None:
                new_children.append(child)

        # Safe population-capped insertion — never exceeds MAX_AGENTS,
        # never lets a newborn arrive in a broken state.
        if new_children:
            agents = _insert_newborns(agents, new_children)

        # Remove dead agents
        agents = [a for a in agents if not a.is_dead]

        if frame_no % 120 == 0:
            pop_history.append(len(agents))
            if len(pop_history) > 500:
                pop_history = pop_history[-500:]

        if selected_agent and selected_agent.is_dead:
            selected_agent = None

    # ── render ───────────────────────────────────────────────
    renderer.render(world, agents, real_dt)

    # HUD composite
    hud_surf.fill((0, 0, 0, 0))
    draw_hud(hud_surf, world, agents, selected_agent, pop_history,
             sim_speed_ref, fps=fps_now, cam_yaw=renderer.cam_yaw)

    # Colonist leaderboard panel — drawn with live mouse for hover/click
    mx_now, my_now = pygame.mouse.get_pos()
    draw_colonist_panel(hud_surf, mx_now, my_now, agents, selected_agent,
                        agent_panel_active=bool(selected_agent and not selected_agent.is_dead))

    raw = pygame.image.tostring(hud_surf, "RGBA", False)
    hud_tex.write(raw)
    hud_tex.use(0)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    hud_vao.render(moderngl.TRIANGLES)
    ctx.enable(moderngl.DEPTH_TEST)

    pygame.display.flip()
    clock.tick(TARGET_FPS)
    frame_no += 1

pygame.quit()
sys.exit(0)
