"""
GPU Renderer — ModernGL (OpenGL 3.3 core)
Uses instanced drawing so the entire world grid + all agents are
rendered in a handful of draw calls, with the GPU handling
colour / position transforms.

Architecture:
  - World tiles: one quad per tile, colour from terrain/object/fire
    packed into a float32 instance buffer uploaded each frame.
  - Agents:      small circle quads, position + colour instanced.
  - Dropped items: tiny dot quads.
  - Overlay:     fire glow (additive blending), day/night tint.
"""

import numpy as np
import moderngl
from engine.config import *
from world.world import (T_GRASS, T_DIRT, T_ROCK, T_RIVER,
                          O_NONE, O_TREE, O_BERRY, O_WHEAT, O_ROCK_OBJ,
                          S_NONE, S_HOUSE, S_FIREPLACE)


# ─── GLSL shaders ────────────────────────────────────────────

TILE_VERT = """
#version 330 core
in vec2 in_vert;          // unit quad [-0.5, 0.5]
in vec4 in_pos_size;      // instance: (world_x, world_y, size, pad)
in vec4 in_color;         // instance: RGBA

uniform mat4 u_proj;

out vec4 v_color;
out vec2 v_uv;

void main() {
    float sz = in_pos_size.z;
    vec2 world_pos = vec2(in_pos_size.x + in_vert.x * sz,
                          in_pos_size.y + in_vert.y * sz);
    gl_Position = u_proj * vec4(world_pos, 0.0, 1.0);
    v_color = in_color;
    v_uv = in_vert + 0.5;
}
"""

TILE_FRAG = """
#version 330 core
in vec4 v_color;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
"""

AGENT_VERT = """
#version 330 core
in vec2 in_vert;
in vec3 in_pos;           // (world_x, world_y, radius)
in vec4 in_color;
in float in_health;       // 0-1 for health bar

uniform mat4 u_proj;

out vec4 v_color;
out vec2 v_uv;
out float v_health;

void main() {
    vec2 wpos = in_pos.xy + in_vert * in_pos.z;
    gl_Position = u_proj * vec4(wpos, 0.0, 1.0);
    v_color = in_color;
    v_uv    = in_vert;
    v_health = in_health;
}
"""

AGENT_FRAG = """
#version 330 core
in vec4 v_color;
in vec2 v_uv;
in float v_health;
out vec4 frag_color;

void main() {
    float dist = length(v_uv);
    if (dist > 0.5) discard;
    // soft edge
    float alpha = 1.0 - smoothstep(0.38, 0.5, dist);
    // health ring — outer 10% of radius
    if (dist > 0.38) {
        vec3 hcol = mix(vec3(0.9,0.1,0.1), vec3(0.1,0.9,0.2), v_health);
        frag_color = vec4(hcol, alpha);
    } else {
        frag_color = vec4(v_color.rgb, v_color.a * alpha);
    }
}
"""

FIRE_VERT = """
#version 330 core
in vec2 in_vert;
in vec4 in_pos_intensity;  // (x, y, intensity 0-100, pad)

uniform mat4 u_proj;
out float v_intensity;
out vec2  v_uv;

void main() {
    float sz = 1.2 + in_pos_intensity.z * 0.02;
    vec2 wpos = in_pos_intensity.xy + in_vert * sz;
    gl_Position = u_proj * vec4(wpos, 0.0, 1.0);
    v_intensity = in_pos_intensity.z / 100.0;
    v_uv = in_vert;
}
"""

FIRE_FRAG = """
#version 330 core
in float v_intensity;
in vec2  v_uv;
out vec4 frag_color;

void main() {
    float dist = length(v_uv);
    if (dist > 0.5) discard;
    float core = 1.0 - dist * 2.0;
    vec3 fire = mix(vec3(0.9, 0.3, 0.0), vec3(1.0, 0.9, 0.1), core);
    float alpha = v_intensity * core * 0.85;
    frag_color = vec4(fire, alpha);
}
"""

OVERLAY_VERT = """
#version 330 core
in vec2 in_vert;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_vert * 0.5 + 0.5;
}
"""

OVERLAY_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform float u_night;     // 0=day, 1=night
uniform float u_season;    // 0=spring 1=summer 2=autumn 3=winter
void main() {
    vec4 day_col   = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 night_col = vec4(0.02, 0.04, 0.12, 0.55);
    // winter blue tint
    vec4 winter_col = vec4(0.05, 0.08, 0.18, 0.22 * step(2.5, u_season));
    vec4 col = mix(day_col, night_col, u_night);
    col += winter_col;
    frag_color = col;
}
"""


def ortho(left, right, bottom, top):
    """2D orthographic projection matrix (column-major)."""
    m = np.eye(4, dtype=np.float32)
    m[0,0] =  2.0 / (right - left)
    m[1,1] =  2.0 / (top - bottom)
    m[2,2] = -1.0
    m[3,0] = -(right + left) / (right - left)
    m[3,1] = -(top + bottom) / (top - bottom)
    return m


class Renderer:
    def __init__(self, ctx: moderngl.Context, screen_w, screen_h):
        self.ctx      = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h

        # camera
        self.cam_x    = WORLD_W * TILE_SIZE / 2
        self.cam_y    = WORLD_H * TILE_SIZE / 2
        self.zoom     = 1.0           # tiles per pixel roughly
        self._view_w  = screen_w / 14.0   # tiles visible
        self._view_h  = screen_h / 14.0

        # compile programs
        self.prog_tile  = ctx.program(vertex_shader=TILE_VERT,  fragment_shader=TILE_FRAG)
        self.prog_agent = ctx.program(vertex_shader=AGENT_VERT, fragment_shader=AGENT_FRAG)
        self.prog_fire  = ctx.program(vertex_shader=FIRE_VERT,  fragment_shader=FIRE_FRAG)
        self.prog_over  = ctx.program(vertex_shader=OVERLAY_VERT, fragment_shader=OVERLAY_FRAG)

        # unit quad
        quad = np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],
                         [-0.5,-0.5],[0.5,0.5],[-0.5,0.5]], dtype=np.float32)
        self.quad_vbo = ctx.buffer(quad.tobytes())

        # tile instances (pre-allocate max)
        N = WORLD_W * WORLD_H
        self._tile_pos_color = np.zeros((N, 8), dtype=np.float32)  # x y size pad r g b a
        self.tile_inst_buf   = ctx.buffer(reserve=N * 8 * 4, dynamic=True)
        self.tile_vao = ctx.vertex_array(self.prog_tile, [
            (self.quad_vbo,     '2f',  'in_vert'),
            (self.tile_inst_buf,'4f/i','in_pos_size'),
            (self.tile_inst_buf,'4f/i','in_color'),
        ])

        # agent instances
        MAXAG = MAX_AGENTS
        self.agent_inst_buf = ctx.buffer(reserve=MAXAG * (3+4+1) * 4, dynamic=True)
        self.agent_vao = ctx.vertex_array(self.prog_agent, [
            (self.quad_vbo,      '2f',  'in_vert'),
            (self.agent_inst_buf,'3f/i','in_pos'),
            (self.agent_inst_buf,'4f/i','in_color'),
            (self.agent_inst_buf,'1f/i','in_health'),
        ])

        # fire instances
        self.fire_inst_buf = ctx.buffer(reserve=WORLD_W*WORLD_H * 4*4, dynamic=True)
        self.fire_vao = ctx.vertex_array(self.prog_fire, [
            (self.quad_vbo,     '2f',  'in_vert'),
            (self.fire_inst_buf,'4f/i','in_pos_intensity'),
        ])

        # fullscreen quad for overlay
        fs_quad = np.array([[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]], dtype=np.float32)
        self.fs_vbo = ctx.buffer(fs_quad.tobytes())
        self.over_vao = ctx.vertex_array(self.prog_over, [(self.fs_vbo,'2f','in_vert')])

        # GL state
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # ── camera ───────────────────────────────────────────────
    def pan(self, dx, dy):
        self.cam_x += dx
        self.cam_y += dy

    def zoom_by(self, factor):
        self.zoom = max(0.3, min(3.0, self.zoom * factor))

    def _proj(self):
        hw = self._view_w * TILE_SIZE / (2 * self.zoom)
        hh = self._view_h * TILE_SIZE / (2 * self.zoom)
        return ortho(self.cam_x - hw, self.cam_x + hw,
                     self.cam_y - hh, self.cam_y + hh)

    # ── render frame ─────────────────────────────────────────
    def render(self, world, agents):
        ctx = self.ctx
        ctx.clear(0.05, 0.05, 0.05, 1.0)

        proj = self._proj()

        # ── 1. Build tile instance data ───────────────────────
        N   = WORLD_W * WORLD_H
        buf = np.zeros((N, 8), dtype=np.float32)
        idx = 0
        for gy in range(WORLD_H):
            for gx in range(WORLD_W):
                wx = (gx + 0.5) * TILE_SIZE
                wy = (gy + 0.5) * TILE_SIZE
                r, g, b = self._tile_color(world, gy, gx)
                buf[idx, 0] = wx
                buf[idx, 1] = wy
                buf[idx, 2] = TILE_SIZE * 0.98
                buf[idx, 3] = 0.0
                buf[idx, 4] = r
                buf[idx, 5] = g
                buf[idx, 6] = b
                buf[idx, 7] = 1.0
                idx += 1

        self.tile_inst_buf.write(buf.tobytes())
        self.prog_tile["u_proj"].write(proj.tobytes())
        self.tile_vao.render(moderngl.TRIANGLES, instances=N)

        # ── 2. Fire glow (additive blend) ─────────────────────
        fire_tiles = np.argwhere(world.fire_intensity > 0)
        if len(fire_tiles) > 0:
            fbuf = np.zeros((len(fire_tiles), 4), dtype=np.float32)
            for i, (fy, fx) in enumerate(fire_tiles):
                fbuf[i, 0] = (fx + 0.5) * TILE_SIZE
                fbuf[i, 1] = (fy + 0.5) * TILE_SIZE
                fbuf[i, 2] = world.fire_intensity[fy, fx]
                fbuf[i, 3] = 0.0
            self.fire_inst_buf.write(fbuf.tobytes())
            # additive blend for glow
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            self.prog_fire["u_proj"].write(proj.tobytes())
            self.fire_vao.render(moderngl.TRIANGLES, instances=len(fire_tiles))
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ── 3. Agents ─────────────────────────────────────────
        live = [a for a in agents if not a.is_dead]
        if live:
            abuf = np.zeros((len(live), 8), dtype=np.float32)
            for i, ag in enumerate(live):
                wx = (ag.x + 0.5) * TILE_SIZE
                wy = (ag.y + 0.5) * TILE_SIZE
                r, g, b = self._agent_color(ag)
                abuf[i, 0] = wx
                abuf[i, 1] = wy
                abuf[i, 2] = TILE_SIZE * AGENT_RADIUS * 2
                abuf[i, 3] = r
                abuf[i, 4] = g
                abuf[i, 5] = b
                abuf[i, 6] = 1.0
                abuf[i, 7] = ag.health / 100.0
            self.agent_inst_buf.write(abuf.tobytes())
            self.prog_agent["u_proj"].write(proj.tobytes())
            self.agent_vao.render(moderngl.TRIANGLES, instances=len(live))

        # ── 4. Night/season overlay ───────────────────────────
        tod = world.time_of_day
        night = max(0.0, min(1.0, abs(tod - 0.5) * 4 - 1.0))
        self.prog_over["u_night"].value  = night
        self.prog_over["u_season"].value = float(world.season_idx)
        self.over_vao.render(moderngl.TRIANGLES)

    # ── colour helpers ────────────────────────────────────────
    def _tile_color(self, world, gy, gx):
        t  = world.terrain[gy, gx]
        ot = world.obj_type[gy, gx]
        og = world.obj_growth[gy, gx]
        st = world.structures[gy, gx]

        # base terrain
        if t == T_GRASS:  r,g,b = 0.22, 0.45, 0.15
        elif t == T_DIRT: r,g,b = 0.55, 0.38, 0.22
        elif t == T_ROCK: r,g,b = 0.50, 0.50, 0.52
        else:             r,g,b = 0.15, 0.40, 0.75  # river

        # structure
        if st == S_HOUSE:      r,g,b = 0.62, 0.48, 0.32
        elif st == S_FIREPLACE: r,g,b = 0.80, 0.30, 0.00

        # object overlay (blend based on growth)
        elif ot == O_TREE:
            gr = og
            r = 0.22*(1-gr) + 0.08*gr
            g = 0.45*(1-gr) + 0.35*gr
            b = 0.15*(1-gr) + 0.08*gr
        elif ot == O_BERRY:
            r = r*(1-og) + 0.65*og
            g = g*(1-og) + 0.10*og
            b = b*(1-og) + 0.10*og
        elif ot == O_WHEAT:
            r = r*(1-og) + 0.85*og
            g = g*(1-og) + 0.75*og
            b = b*(1-og) + 0.05*og
        elif ot == O_ROCK_OBJ:
            r,g,b = 0.48, 0.46, 0.44

        # fire tint
        fi = world.fire_intensity[gy, gx]
        if fi > 0:
            fi_t = min(fi / 80.0, 1.0)
            r = r*(1-fi_t) + 0.9*fi_t
            g = g*(1-fi_t) + 0.2*fi_t
            b = b*(1-fi_t) + 0.0*fi_t

        return r, g, b

    def _agent_color(self, ag):
        stage_cols = {
            "baby":  (0.95, 0.85, 0.40),
            "adult": (0.90, 0.90, 0.90) if ag.gender == "male" else (0.95, 0.65, 0.75),
            "elder": (0.70, 0.70, 0.80),
        }
        base = stage_cols.get(ag.stage, (0.9, 0.9, 0.9))
        # tint by current action
        from ai.agent import A_FLEE_FIRE, A_FIND_FOOD, A_FIND_WATER, A_BUILD, A_SEEK_MATE
        if ag.current_action == A_FLEE_FIRE:  return (1.0, 0.2, 0.0)
        if ag.current_action == A_FIND_WATER: return (0.2, 0.6, 1.0)
        if ag.current_action == A_FIND_FOOD:  return (0.6, 1.0, 0.2)
        if ag.current_action == A_BUILD:      return (0.9, 0.7, 0.3)
        if ag.current_action == A_SEEK_MATE:  return (1.0, 0.4, 0.8)
        return base

    # ── screen → world coords ─────────────────────────────────
    def screen_to_world(self, sx, sy):
        hw = self._view_w * TILE_SIZE / (2 * self.zoom)
        hh = self._view_h * TILE_SIZE / (2 * self.zoom)
        nx = (sx / self.screen_w - 0.5) * 2
        ny = (sy / self.screen_h - 0.5) * 2
        wx = self.cam_x + nx * hw
        wy = self.cam_y + ny * hh
        return wx, wy
