#!/usr/bin/env python3
"""
SURVIVAL SIMULATION
===================
GPU-accelerated (OpenGL via ModernGL) survival world with
Numba-JIT AI kernels, full lifecycle, learning agents.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math, random, time
import numpy as np
import pygame
import moderngl

from engine.config import *
from engine.renderer import Renderer
from engine.hud import draw_hud
from world.world import World
from ai.agent import Agent
from menu import run_menu  # 1. Import the configuration menu module

# ─── init & pre-launch menu ───────────────────────────────────
pygame.init()

# 2. Launch configuration screen before binding the OpenGL context
# We pass gpu_available=True here because the engine requires ModernGL/OpenGL 3.3 core
menu_settings = run_menu(gpu_available=True)

# If the user closed the window or pressed ESC in the menu, exit cleanly
if menu_settings is None:
    pygame.quit()
    sys.exit(0)

# 3. Dynamic config updates from menu parameters
import engine.config as cfg
cfg.INITIAL_AGENTS = menu_settings["population"]

# 4. Initialize OpenGL Display Mode with config dimensions
pygame.display.set_caption("Survival Simulation — GPU Edition")
flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)

# ModernGL context from pygame GL context
ctx = moderngl.create_context()

# pygame surface for HUD (we render it as a texture on top)
hud_surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)

# create HUD texture
hud_tex = ctx.texture((SCREEN_W, SCREEN_H), 4)
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

# ─── simulation state ─────────────────────────────────────────
# 5. Apply the customized seed and seasonal parameters from menu
world = World(seed=menu_settings["terrain_seed"])

# If your World object has attributes to parse seasons, map them here:
if hasattr(world, "enable_seasons"):
    world.enable_seasons = menu_settings["seasons"]
    world.current_season = menu_settings["start_season"]

agents   = []
renderer = Renderer(ctx, SCREEN_W, SCREEN_H)

# spawn initial agents matching menu settings
# 6. Explicitly forward 'neural_size' constraint to custom agent architectures if applicable
for i in range(cfg.INITIAL_AGENTS):
    x = random.uniform(WORLD_W * 0.2, WORLD_W * 0.8)
    y = random.uniform(WORLD_H * 0.2, WORLD_H * 0.8)
    
    # Injecting custom neural parameters if supported by Agent __init__
    if "neural_size" in menu_settings:
        a = Agent(x, y, hidden_size=menu_settings["neural_size"])
    else:
        a = Agent(x, y)
        
    a.inv_add("berry", 3, freshness=90.0)
    a.inv_add("rock",  2)
    agents.append(a)

pop_history   = [len(agents)]
selected_agent = None
paused         = False
sim_speed_ref  = [SIM_SPEED]

clock      = pygame.time.Clock()
last_time  = time.perf_counter()
frame_no   = 0
pan_speed  = 40.0  # tiles per second when panning


def reset():
    global world, agents, pop_history, selected_agent
    # Generate a random seed during hot resets or track menu settings
    new_seed = random.randint(0, 999) if menu_settings is None else menu_settings["terrain_seed"]
    world = World(seed=new_seed)
    
    if hasattr(world, "enable_seasons"):
        world.enable_seasons = menu_settings["seasons"]
        world.current_season = menu_settings["start_season"]

    agents = []
    for i in range(cfg.INITIAL_AGENTS):
        x = random.uniform(WORLD_W * 0.2, WORLD_W * 0.8)
        y = random.uniform(WORLD_H * 0.2, WORLD_H * 0.8)
        
        if "neural_size" in menu_settings:
            a = Agent(x, y, hidden_size=menu_settings["neural_size"])
        else:
            a = Agent(x, y)
            
        a.inv_add("berry", 3, freshness=90.0)
        a.inv_add("rock",  2)
        agents.append(a)
    pop_history    = [len(agents)]
    selected_agent = None


# ─── main loop ────────────────────────────────────────────────
running = True
while running:
    now     = time.perf_counter()
    real_dt = min(now - last_time, 0.05)   # cap at 50ms to prevent spiral
    last_time = now

    # ── events ────────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                reset()
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                sim_speed_ref[0] = min(50.0, sim_speed_ref[0] * 1.5)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                sim_speed_ref[0] = max(0.1, sim_speed_ref[0] / 1.5)
            elif event.key == pygame.K_f:
                # start fire at camera centre
                mx, my = pygame.mouse.get_pos()
                wx, wy = renderer.screen_to_world(mx, my)
                tx, ty = int(wx / TILE_SIZE), int(wy / TILE_SIZE)
                if 0 <= ty < WORLD_H and 0 <= tx < WORLD_W:
                    world.start_fire(ty, tx)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mx, my = event.pos
                wx, wy = renderer.screen_to_world(mx, my)
                tx = wx / TILE_SIZE
                ty = wy / TILE_SIZE
                # find nearest agent
                best_d = 2.5
                selected_agent = None
                for ag in agents:
                    if ag.is_dead: continue
                    d = math.sqrt((ag.x - tx)**2 + (ag.y - ty)**2)
                    if d < best_d:
                        best_d = d
                        selected_agent = ag
            elif event.button == 4:
                renderer.zoom_by(1.12)
            elif event.button == 5:
                renderer.zoom_by(1 / 1.12)

        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(event.size, flags)
            renderer.screen_w = event.w
            renderer.screen_h = event.h
            hud_surf = pygame.Surface(event.size, pygame.SRCALPHA)
            hud_tex = ctx.texture(event.size, 4)
            hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ── keyboard pan ──────────────────────────────────────────
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]  or keys[pygame.K_a]: renderer.pan(-pan_speed * real_dt * TILE_SIZE, 0)
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: renderer.pan( pan_speed * real_dt * TILE_SIZE, 0)
    if keys[pygame.K_UP]    or keys[pygame.K_w]: renderer.pan(0,  pan_speed * real_dt * TILE_SIZE)
    if keys[pygame.K_DOWN]  or keys[pygame.K_s]: renderer.pan(0, -pan_speed * real_dt * TILE_SIZE)

    # ── simulation update ─────────────────────────────────────
    if not paused:
        # update sim speed in config
        import engine.config as cfg
        cfg.SIM_SPEED = sim_speed_ref[0]

        world.tick(real_dt)

        new_children = []
        for ag in agents:
            ag.tick(world, agents, real_dt)
            child = ag.tick_pregnancy(world, agents, real_dt * sim_speed_ref[0])
            if child and len(agents) + len(new_children) < MAX_AGENTS:
                new_children.append(child)

        agents.extend(new_children)

        # prune dead
        agents = [a for a in agents if not a.is_dead]

        # record population every ~2 real seconds
        if frame_no % 120 == 0:
            pop_history.append(len(agents))
            if len(pop_history) > 500:
                pop_history = pop_history[-500:]

        # clean selected
        if selected_agent and selected_agent.is_dead:
            selected_agent = None

    # ── render ────────────────────────────────────────────────
    renderer.render(world, agents)

    # HUD: render to pygame surface, upload to GL texture, composite
    hud_surf.fill((0, 0, 0, 0))
    draw_hud(hud_surf, world, agents, selected_agent, pop_history, sim_speed_ref)

    # upload hud surface to GL texture
    raw = pygame.image.tostring(hud_surf, "RGBA", False)
    hud_tex.write(raw)
    hud_tex.use(0)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    hud_vao.render(moderngl.TRIANGLES)

    pygame.display.flip()
    clock.tick(TARGET_FPS)
    frame_no += 1

pygame.quit()
sys.exit(0)
