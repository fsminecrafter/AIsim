"""
3D GPU Renderer — ModernGL (OpenGL 3.3 core)
============================================
Full perspective-projection 3D rendering:
  - Terrain heightmap with per-tile 3D quads (raised ground)
  - Sand strips near rivers
  - Trees  → green cylinders + sphere canopy
  - Berry bushes → small sphere clusters
  - Wheat → thin tall quads (billboard-ish)
  - Rocks → sphere-ish scaled cubes
  - Agents → capsule (cylinder + 2 hemispheres)
  - Wooden planks → flat rectangular block
  - Fireplace → dark stone ring mesh
  - House → box structure
  - Fire glow overlay (additive, billboard quads)
  - Day/night ambient tint
  - Simple shadow tint under agents
"""

import numpy as np
import moderngl
import math
from engine.config import *
from world.world import (T_GRASS, T_DIRT, T_ROCK, T_RIVER,
                          O_NONE, O_TREE, O_BERRY, O_WHEAT, O_ROCK_OBJ,
                          S_NONE, S_HOUSE, S_FIREPLACE)

# ─── helpers ─────────────────────────────────────────────────

def _perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, center, up):
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] =  np.dot(f, eye)
    return m


def _mat_mul(a, b):
    return (a @ b).astype(np.float32)


def _translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x; m[1, 3] = y; m[2, 3] = z
    return m


def _scale(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = x; m[1, 1] = y; m[2, 2] = z
    return m


def _rotate_x(a):
    m = np.eye(4, dtype=np.float32)
    c, s = math.cos(a), math.sin(a)
    m[1, 1] = c; m[1, 2] = -s
    m[2, 1] = s; m[2, 2] =  c
    return m


# ─── mesh builders ───────────────────────────────────────────

def _box_mesh():
    """Unit cube centred at origin. Returns (verts, normals, indices)."""
    # 6 faces × 4 verts, triangulated
    faces = [
        # pos_x
        ([ 0.5,-0.5,-0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5], [1,0,0]),
        # neg_x
        ([-0.5,-0.5, 0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[-0.5, 0.5, 0.5], [-1,0,0]),
        # pos_y
        ([-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5], [0,1,0]),
        # neg_y
        ([-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5], [0,-1,0]),
        # pos_z
        ([-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5], [0,0,1]),
        # neg_z
        ([ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5], [0,0,-1]),
    ]
    verts = []
    norms = []
    idxs  = []
    base = 0
    for f in faces:
        v0, v1, v2, v3, n = f
        for v in [v0, v1, v2, v3]:
            verts.extend(v)
            norms.extend(n)
        idxs.extend([base, base+1, base+2, base, base+2, base+3])
        base += 4
    return (np.array(verts, dtype=np.float32).reshape(-1, 3),
            np.array(norms, dtype=np.float32).reshape(-1, 3),
            np.array(idxs,  dtype=np.uint32))


def _cylinder_mesh(segs=10):
    """Unit cylinder: radius 0.5, height 1.0, centred at y=0.5. Returns (verts, normals, indices)."""
    verts, norms, idxs = [], [], []
    base_i = 0
    for i in range(segs):
        a0 = 2 * math.pi * i / segs
        a1 = 2 * math.pi * (i + 1) / segs
        x0, z0 = 0.5 * math.cos(a0), 0.5 * math.sin(a0)
        x1, z1 = 0.5 * math.cos(a1), 0.5 * math.sin(a1)
        # side quad
        for x, z in [(x0, z0), (x1, z1), (x1, z1), (x0, z0)]:
            verts.extend([x, 0.0, z])
            verts.extend([x, 1.0, z])
            n = [x * 2, 0, z * 2]
            norms.extend(n); norms.extend(n)
        idxs.extend([base_i, base_i+1, base_i+2, base_i, base_i+2, base_i+3])
        base_i += 4
    # top cap
    cx = 0.0; cy = 1.0; cz = 0.0
    for i in range(segs):
        a0 = 2 * math.pi * i / segs
        a1 = 2 * math.pi * (i + 1) / segs
        verts.extend([0, 1, 0]); norms.extend([0, 1, 0])
        verts.extend([0.5*math.cos(a0), 1, 0.5*math.sin(a0)]); norms.extend([0, 1, 0])
        verts.extend([0.5*math.cos(a1), 1, 0.5*math.sin(a1)]); norms.extend([0, 1, 0])
        idxs.extend([base_i, base_i+1, base_i+2])
        base_i += 3
    return (np.array(verts, dtype=np.float32).reshape(-1, 3),
            np.array(norms, dtype=np.float32).reshape(-1, 3),
            np.array(idxs,  dtype=np.uint32))


def _sphere_mesh(segs=10):
    """Unit sphere, radius 0.5."""
    verts, norms, idxs = [], [], []
    rings = segs // 2
    idx = 0
    for i in range(rings):
        lat0 = math.pi * (-0.5 + i / rings)
        lat1 = math.pi * (-0.5 + (i+1) / rings)
        for j in range(segs):
            lon0 = 2 * math.pi * j / segs
            lon1 = 2 * math.pi * (j+1) / segs
            for (lat, lon) in [(lat0, lon0), (lat0, lon1), (lat1, lon1), (lat1, lon0)]:
                x = 0.5 * math.cos(lat) * math.cos(lon)
                y = 0.5 * math.sin(lat)
                z = 0.5 * math.cos(lat) * math.sin(lon)
                verts.extend([x, y, z])
                norms.extend([x*2, y*2, z*2])
            idxs.extend([idx, idx+1, idx+2, idx, idx+2, idx+3])
            idx += 4
    return (np.array(verts, dtype=np.float32).reshape(-1, 3),
            np.array(norms, dtype=np.float32).reshape(-1, 3),
            np.array(idxs,  dtype=np.uint32))


def _quad_mesh():
    """Unit billboard quad, centred, facing +Z."""
    verts = np.array([
        [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]
    ], dtype=np.float32)
    norms = np.zeros_like(verts); norms[:, 2] = 1
    idxs  = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return verts, norms, idxs


def _upload_mesh(ctx, verts, norms, idxs):
    """Pack verts + normals into single VBO, return (vbo, ibo, stride)."""
    data = np.hstack([verts, norms]).astype(np.float32)
    vbo = ctx.buffer(data.tobytes())
    ibo = ctx.buffer(idxs.tobytes())
    return vbo, ibo, len(idxs)


# ─── GLSL shaders ────────────────────────────────────────────

VERT_3D = """
#version 330 core
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_inst_pos;    // world position (x,y,z)
in vec3 in_inst_scale;  // (sx,sy,sz)
in vec4 in_inst_color;  // RGBA

uniform mat4 u_mvp;
uniform mat4 u_view;
uniform vec3 u_light_dir;  // normalised, world space

out vec4 v_color;
out float v_fog;

void main() {
    vec3 world = in_inst_pos + in_pos * in_inst_scale;
    gl_Position = u_mvp * vec4(world, 1.0);

    // simple diffuse + ambient
    float diff = max(dot(normalize(in_normal), normalize(u_light_dir)), 0.0);
    float light = 0.35 + 0.65 * diff;
    v_color = vec4(in_inst_color.rgb * light, in_inst_color.a);

    // distance fog
    float dist = length(world.xz - vec2(u_mvp[3][0], u_mvp[3][2]));
    v_fog = clamp((length(world) * 0.004), 0.0, 0.5);
}
"""

FRAG_3D = """
#version 330 core
in vec4 v_color;
in float v_fog;
out vec4 frag_color;
uniform float u_night;  // 0=day 1=night
uniform float u_weather_dark;  // 0-0.4 for storms
void main() {
    vec3 night_fog = vec3(0.02, 0.04, 0.1);
    vec3 day_fog   = vec3(0.55, 0.65, 0.75);
    vec3 fog_col   = mix(day_fog, night_fog, u_night);
    vec3 col = mix(v_color.rgb, fog_col, v_fog);
    // night darkening
    col = mix(col, col * vec3(0.15, 0.18, 0.3), u_night * 0.7);
    col *= (1.0 - u_weather_dark);
    frag_color = vec4(col, v_color.a);
}
"""

# Fire billboard shader
FIRE_VERT = """
#version 330 core
in vec2 in_quad;
in vec3 in_pos;
in float in_intensity;

uniform mat4 u_mvp;
uniform mat4 u_view;

out float v_intensity;
out vec2  v_uv;
out float v_flicker;

void main() {
    float sz = 0.5 + in_intensity * 0.015;
    // billboard: extract camera right/up from view matrix
    vec3 cam_right = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 cam_up    = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);
    vec3 world = in_pos + cam_right * in_quad.x * sz + cam_up * in_quad.y * sz * 1.6;
    gl_Position = u_mvp * vec4(world, 1.0);
    v_intensity = in_intensity / 80.0;
    v_uv = in_quad * 0.5 + 0.5;
    v_flicker = sin(in_pos.x * 13.1 + in_pos.z * 7.3) * 0.15;
}
"""

FIRE_FRAG = """
#version 330 core
in float v_intensity;
in vec2  v_uv;
in float v_flicker;
out vec4 frag_color;
void main() {
    vec2 d = v_uv - vec2(0.5, 0.35);
    float r = length(d * vec2(1.0, 1.4));
    if (r > 0.5) discard;
    float core = 1.0 - r * 2.0 + v_flicker;
    core = clamp(core, 0.0, 1.0);
    vec3 fire_col = mix(vec3(0.95, 0.3, 0.0), vec3(1.0, 0.95, 0.2), core);
    float alpha = v_intensity * core * 0.9;
    frag_color = vec4(fire_col, alpha);
}
"""

# Overlay (day/night tint, full-screen quad)
OVERLAY_VERT = """
#version 330 core
in vec2 in_vert;
void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }
"""
OVERLAY_FRAG = """
#version 330 core
out vec4 frag_color;
uniform float u_night;
uniform float u_weather_dark;
void main() {
    float a = u_night * 0.45 + u_weather_dark * 0.35;
    vec3 col = mix(vec3(0.0), vec3(0.02, 0.04, 0.15), u_night);
    frag_color = vec4(col, a);
}
"""

# Particle effects (rain, snow)
PARTICLE_VERT = """
#version 330 core
in vec3 in_pos;
in float in_alpha;
uniform mat4 u_mvp;
out float v_alpha;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = 2.0;
    v_alpha = in_alpha;
}
"""
PARTICLE_FRAG = """
#version 330 core
in float v_alpha;
out vec4 frag_color;
void main() {
    frag_color = vec4(0.7, 0.8, 1.0, v_alpha);
}
"""


class Renderer:
    def __init__(self, ctx: moderngl.Context, screen_w, screen_h):
        self.ctx      = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h

        # ── camera ─────────────────────────────────────────────
        self.cam_pitch   = -55.0   # degrees, negative = look down
        self.cam_yaw     = 0.0
        self.cam_dist    = 28.0
        self.cam_target  = np.array([WORLD_W * TILE_SIZE / 2,
                                      0.0,
                                      WORLD_H * TILE_SIZE / 2], dtype=np.float32)
        self.zoom        = 1.0

        # ── programs ──────────────────────────────────────────
        self.prog_3d     = ctx.program(vertex_shader=VERT_3D,      fragment_shader=FRAG_3D)
        self.prog_fire   = ctx.program(vertex_shader=FIRE_VERT,    fragment_shader=FIRE_FRAG)
        self.prog_over   = ctx.program(vertex_shader=OVERLAY_VERT, fragment_shader=OVERLAY_FRAG)
        self.prog_part   = ctx.program(vertex_shader=PARTICLE_VERT, fragment_shader=PARTICLE_FRAG)

        # ── meshes ────────────────────────────────────────────
        box_v, box_n, box_i       = _box_mesh()
        cyl_v, cyl_n, cyl_i       = _cylinder_mesh(12)
        sph_v, sph_n, sph_i       = _sphere_mesh(12)
        quad_v, quad_n, quad_i    = _quad_mesh()

        self._box_vbo,  self._box_ibo,  self._box_count  = _upload_mesh(ctx, box_v,  box_n,  box_i)
        self._cyl_vbo,  self._cyl_ibo,  self._cyl_count  = _upload_mesh(ctx, cyl_v,  cyl_n,  cyl_i)
        self._sph_vbo,  self._sph_ibo,  self._sph_count  = _upload_mesh(ctx, sph_v,  sph_n,  sph_i)
        self._quad_vbo, self._quad_ibo, self._quad_count = _upload_mesh(ctx, quad_v, quad_n, quad_i)

        # max instances per batch
        MAX_INST = WORLD_W * WORLD_H * 3
        # per-instance: pos(3) + scale(3) + color(4) = 10 floats = 40 bytes
        self._inst_buf = ctx.buffer(reserve=MAX_INST * 10 * 4, dynamic=True)

        # build VAOs for each mesh type (instanced)
        self._vao_box  = self._make_instanced_vao(self.prog_3d, self._box_vbo,  self._box_ibo)
        self._vao_cyl  = self._make_instanced_vao(self.prog_3d, self._cyl_vbo,  self._cyl_ibo)
        self._vao_sph  = self._make_instanced_vao(self.prog_3d, self._sph_vbo,  self._sph_ibo)
        self._vao_quad = self._make_instanced_vao(self.prog_3d, self._quad_vbo, self._quad_ibo)

        # fire billboards
        fire_quad = np.array([[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]], dtype=np.float32)
        self._fire_quad_vbo = ctx.buffer(fire_quad.tobytes())
        self._fire_inst_buf = ctx.buffer(reserve=WORLD_W*WORLD_H*4*4, dynamic=True)
        self._fire_vao = ctx.vertex_array(self.prog_fire, [
            (self._fire_quad_vbo, '2f',  'in_quad'),
            (self._fire_inst_buf, '3f/i','in_pos'),
            (self._fire_inst_buf, '1f/i','in_intensity'),
        ])

        # fullscreen overlay quad
        fs = np.array([[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]], dtype=np.float32)
        self._fs_vbo  = ctx.buffer(fs.tobytes())
        self._over_vao = ctx.vertex_array(self.prog_over, [(self._fs_vbo,'2f','in_vert')])

        # particles
        self._part_vbo = ctx.buffer(reserve=4000*4*4, dynamic=True)
        self._part_vao = ctx.vertex_array(self.prog_part, [(self._part_vbo,'3f 1f','in_pos','in_alpha')])
        self._particles = []
        self._part_timer = 0.0

        # GL state
        ctx.enable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # precomputed terrain heights
        self._height_map = None

    def _make_instanced_vao(self, prog, vbo, ibo):
        """Create instanced VAO: mesh verts from vbo, per-instance from self._inst_buf."""
        return self.ctx.vertex_array(prog, [
            (vbo,             '3f 3f',   'in_pos', 'in_normal'),
            (self._inst_buf,  '3f/i 3f/i 4f/i', 'in_inst_pos', 'in_inst_scale', 'in_inst_color'),
        ], ibo)

    # ── camera interface ─────────────────────────────────────
    def pan(self, dx, dy):
        # pan in world XZ based on current yaw
        yr = math.radians(self.cam_yaw)
        self.cam_target[0] += math.cos(yr) * dx - math.sin(yr) * dy
        self.cam_target[2] += math.sin(yr) * dx + math.cos(yr) * dy

    def zoom_by(self, factor):
        self.cam_dist = max(5.0, min(80.0, self.cam_dist / factor))

    def orbit(self, dyaw, dpitch):
        self.cam_yaw   = (self.cam_yaw + dyaw) % 360
        self.cam_pitch = max(-85, min(-15, self.cam_pitch + dpitch))

    def screen_to_world(self, sx, sy):
        """Approximate: unproject screen coord to world XZ at y=0 plane."""
        # simple: return cam target offset by screen-space delta
        norm_x = (sx / self.screen_w - 0.5) * 2
        norm_y = -(sy / self.screen_h - 0.5) * 2
        scale = self.cam_dist * 0.3
        return (self.cam_target[0] + norm_x * scale,
                self.cam_target[2] + norm_y * scale)

    def _get_view_proj(self):
        aspect = self.screen_w / max(1, self.screen_h)
        proj   = _perspective(45.0 / self.zoom, aspect, 0.5, 600.0)

        pr = math.radians(self.cam_pitch)
        yr = math.radians(self.cam_yaw)
        eye_offset = np.array([
            self.cam_dist * math.cos(pr) * math.sin(yr),
            -self.cam_dist * math.sin(pr),
            self.cam_dist * math.cos(pr) * math.cos(yr),
        ], dtype=np.float32)
        eye = self.cam_target + eye_offset
        view = _look_at(eye, self.cam_target, [0, 1, 0])
        return view, _mat_mul(proj, view)

    def _light_dir(self, world):
        # sun angle follows time_of_day
        tod = world.time_of_day
        sun_angle = math.pi * tod
        lx = math.cos(sun_angle)
        ly = math.sin(sun_angle) + 0.3
        lz = 0.5
        ln = math.sqrt(lx*lx + ly*ly + lz*lz)
        return [lx/ln, ly/ln, lz/ln]

    # ── main render ──────────────────────────────────────────
    def render(self, world, agents, dt=0.016):
        ctx = self.ctx
        ctx.clear(0.45, 0.60, 0.75, 1.0)   # sky blue
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = '<'

        if self._height_map is None:
            self._build_height_map(world)

        view, mvp = self._get_view_proj()
        light = self._light_dir(world)
        tod = world.time_of_day
        night = max(0.0, min(1.0, abs(tod - 0.5) * 4 - 1.0))
        weather_dark = {"storm": 0.3, "blizzard": 0.25, "rain": 0.1}.get(world.weather, 0.0)

        # set common uniforms
        mvp_bytes = mvp.T.tobytes()
        view_bytes = view.T.tobytes()
        for prog in [self.prog_3d, self.prog_fire]:
            prog["u_mvp"].write(mvp_bytes)
            if "u_view" in prog:
                prog["u_view"].write(view_bytes)

        self.prog_3d["u_light_dir"].value = tuple(light)
        self.prog_3d["u_night"].value = night
        self.prog_3d["u_weather_dark"].value = weather_dark

        # ── 1. Terrain ────────────────────────────────────────
        self._draw_terrain(world)

        # ── 2. Objects (trees, berries, wheat, rocks) ─────────
        self._draw_objects(world)

        # ── 3. Structures (houses, fireplaces, planks) ────────
        self._draw_structures(world)

        # ── 4. Fire glows (additive) ──────────────────────────
        fire_tiles = np.argwhere(world.fire_intensity > 0)
        if len(fire_tiles) > 0:
            fbuf = np.zeros((len(fire_tiles), 4), dtype=np.float32)
            for i, (fy, fx) in enumerate(fire_tiles):
                fbuf[i, 0] = fx * TILE_SIZE
                fbuf[i, 1] = self._h(fy, fx) + 0.5
                fbuf[i, 2] = fy * TILE_SIZE
                fbuf[i, 3] = world.fire_intensity[fy, fx]
            self._fire_inst_buf.write(fbuf.tobytes())
            self.prog_fire["u_view"].write(view_bytes)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            ctx.disable(moderngl.DEPTH_TEST)
            self._fire_vao.render(moderngl.TRIANGLES, instances=len(fire_tiles))
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ── 5. Agents (capsules) ──────────────────────────────
        self._draw_agents(agents)

        # ── 6. Particles (rain / snow) ────────────────────────
        self._update_draw_particles(world, dt)

        # ── 7. Night/weather overlay ─────────────────────────
        ctx.disable(moderngl.DEPTH_TEST)
        self.prog_over["u_night"].value = night
        self.prog_over["u_weather_dark"].value = weather_dark
        self._over_vao.render(moderngl.TRIANGLES)
        ctx.enable(moderngl.DEPTH_TEST)

    # ── terrain ──────────────────────────────────────────────
    def _build_height_map(self, world):
        H, W = WORLD_H, WORLD_W
        self._height_map = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                t = world.terrain[y, x]
                if t == T_ROCK:
                    self._height_map[y, x] = 0.9 + (hash((y*97+x*13)) % 10) * 0.05
                elif t == T_RIVER:
                    self._height_map[y, x] = -0.12
                elif t == T_DIRT:
                    self._height_map[y, x] = 0.0
                else:
                    self._height_map[y, x] = 0.08

    def _h(self, ty, tx):
        if self._height_map is None:
            return 0.0
        ty = max(0, min(WORLD_H-1, ty))
        tx = max(0, min(WORLD_W-1, tx))
        return float(self._height_map[ty, tx])

    def _draw_terrain(self, world):
        instances = []
        T = TILE_SIZE
        for gy in range(WORLD_H):
            for gx in range(WORLD_W):
                t = world.terrain[gy, gx]
                h = self._h(gy, gx)
                wx = gx * T
                wz = gy * T
                wy = h * 0.5 - 0.5

                # near-river sand
                is_sand = False
                if t in (T_GRASS, T_DIRT):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = gy+dy, gx+dx
                            if 0 <= ny < WORLD_H and 0 <= nx < WORLD_W:
                                if world.terrain[ny, nx] == T_RIVER:
                                    is_sand = True

                r, g, b = self._terrain_color(t, gy, gx, is_sand, world)

                fire_t = world.fire_intensity[gy, gx]
                if fire_t > 0:
                    ft = min(fire_t / 60.0, 0.7)
                    r = r*(1-ft) + 0.85*ft
                    g = g*(1-ft) + 0.15*ft
                    b = b*(1-ft) + 0.0*ft

                tile_h = max(0.3, abs(h) + 0.3)
                instances.append([wx + T/2, wy, wz + T/2,
                                   T * 0.99, tile_h, T * 0.99,
                                   r, g, b, 1.0])

        inst = np.array(instances, dtype=np.float32)
        self._inst_buf.orphan(size=len(instances)*10*4)
        self._inst_buf.write(inst.tobytes())
        self._vao_box.render(moderngl.TRIANGLES, instances=len(instances))

    def _terrain_color(self, t, gy, gx, is_sand, world):
        if t == T_RIVER:
            return 0.12, 0.38, 0.72
        if is_sand:
            return 0.85, 0.76, 0.48
        if t == T_ROCK:
            return 0.52, 0.50, 0.50
        if t == T_GRASS:
            season = world.season_name
            if season == "autumn":
                return 0.55, 0.42, 0.12
            if season == "winter":
                return 0.78, 0.82, 0.87
            return 0.22, 0.46, 0.15
        return 0.54, 0.38, 0.22  # dirt

    # ── objects ──────────────────────────────────────────────
    def _draw_objects(self, world):
        trees_cyl, trees_sph = [], []
        berries, rocks, wheat_quads = [], [], []

        T = TILE_SIZE
        for gy in range(WORLD_H):
            for gx in range(WORLD_W):
                ot = world.obj_type[gy, gx]
                if ot == O_NONE:
                    continue
                og = world.obj_growth[gy, gx]
                h  = self._h(gy, gx)
                wx = gx * T + T/2
                wz = gy * T + T/2
                wy = h * 0.5

                if ot == O_TREE:
                    # trunk: cylinder, canopy: sphere
                    trunk_h = 1.5 + og * 2.5
                    trunk_r = 0.2 + og * 0.15
                    cr = 0.3 + og * 0.12; cg = 0.6 - og * 0.05; cb = 0.12
                    trees_cyl.append([wx, wy, wz,  trunk_r*2, trunk_h, trunk_r*2,  0.52, 0.36, 0.16, 1.0])
                    # canopy on top of trunk
                    can_r = 1.2 * og + 0.3
                    trees_sph.append([wx, wy + trunk_h, wz,  can_r*2, can_r*2, can_r*2,  cr, cg, cb, 1.0])

                elif ot == O_BERRY:
                    sz = 0.45 + og * 0.35
                    # bush base
                    berries.append([wx, wy + sz*0.5, wz,  sz, sz*0.7, sz,  0.18, 0.48, 0.12, 1.0])
                    # berries: small red spheres
                    if og > 0.5:
                        for bdy, bdx in [(-0.2, 0.15), (0.15, -0.1), (0.1, 0.2)]:
                            berries.append([wx+bdx*T/5, wy+sz*0.8, wz+bdy*T/5,
                                           0.12, 0.12, 0.12,  0.80, 0.15, 0.15, 1.0])

                elif ot == O_WHEAT:
                    # 3 thin vertical quads forming a wheat stalk cluster
                    h_stalk = 0.6 + og * 0.9
                    col = (0.85, 0.72, 0.12) if og > 0.7 else (0.45, 0.70, 0.20)
                    wheat_quads.append([wx, wy + h_stalk/2, wz,
                                       0.08, h_stalk, 0.08,  col[0], col[1], col[2], 1.0])
                    wheat_quads.append([wx+0.15, wy + h_stalk/2, wz+0.15,
                                       0.07, h_stalk*0.85, 0.07,  col[0]*0.9, col[1], col[2], 1.0])
                    wheat_quads.append([wx-0.15, wy + h_stalk/2, wz-0.1,
                                       0.07, h_stalk*0.9, 0.07,  col[0], col[1]*0.9, col[2], 1.0])

                elif ot == O_ROCK_OBJ:
                    sz = 0.45
                    rocks.append([wx, wy + sz*0.45, wz,  sz*1.1, sz*0.7, sz*0.9,  0.56, 0.54, 0.52, 1.0])
                    rocks.append([wx+0.2, wy + sz*0.3, wz-0.1,  sz*0.6, sz*0.5, sz*0.6,  0.48, 0.46, 0.44, 1.0])

        def _batch_draw(vao, data):
            if data:
                arr = np.array(data, dtype=np.float32)
                self._inst_buf.orphan(size=len(data)*10*4)
                self._inst_buf.write(arr.tobytes())
                vao.render(moderngl.TRIANGLES, instances=len(data))

        _batch_draw(self._vao_cyl, trees_cyl)
        _batch_draw(self._vao_sph, trees_sph)
        _batch_draw(self._vao_sph, berries)
        _batch_draw(self._vao_box, rocks)
        _batch_draw(self._vao_box, wheat_quads)

    # ── structures ────────────────────────────────────────────
    def _draw_structures(self, world):
        houses     = []
        fireplaces = []
        planks     = []

        T = TILE_SIZE
        for gy in range(WORLD_H):
            for gx in range(WORLD_W):
                s = world.structures[gy, gx]
                if s == S_NONE:
                    continue
                h  = self._h(gy, gx)
                wx = gx * T + T/2
                wz = gy * T + T/2
                wy = h * 0.5

                if s == S_HOUSE:
                    # walls
                    houses.append([wx, wy + 0.8, wz,  T*0.9, 1.6, T*0.9,  0.62, 0.48, 0.32, 1.0])
                    # roof (darker wedge via flat box)
                    houses.append([wx, wy + 1.7, wz,  T*0.95, 0.4, T*0.95,  0.45, 0.30, 0.18, 1.0])

                elif s == S_FIREPLACE:
                    # stone ring: 4 small dark cubes in corners
                    for dy, dx in [(-0.25,-0.25),(-0.25,0.25),(0.25,-0.25),(0.25,0.25)]:
                        fireplaces.append([wx+dx, wy+0.22, wz+dy,  0.22, 0.45, 0.22,  0.32, 0.30, 0.30, 1.0])
                    # glowing embers
                    fireplaces.append([wx, wy+0.12, wz,  0.28, 0.12, 0.28,  0.95, 0.42, 0.05, 1.0])

        def _batch_draw(vao, data):
            if data:
                arr = np.array(data, dtype=np.float32)
                self._inst_buf.orphan(size=len(data)*10*4)
                self._inst_buf.write(arr.tobytes())
                vao.render(moderngl.TRIANGLES, instances=len(data))

        _batch_draw(self._vao_box, houses + fireplaces)

    # ── agents ────────────────────────────────────────────────
    def _draw_agents(self, agents):
        live = [a for a in agents if not a.is_dead]
        if not live:
            return

        cyls = []  # bodies
        sphs = []  # heads
        shad = []  # flat shadow discs

        T = TILE_SIZE
        for ag in live:
            ty, tx = int(ag.y), int(ag.x)
            ty = max(0, min(WORLD_H-1, ty))
            tx = max(0, min(WORLD_W-1, tx))
            h  = self._h(ty, tx)
            wx = ag.x * T + T/2
            wz = ag.y * T + T/2
            wy = h * 0.5

            r, g, b = self._agent_color(ag)
            scale = 0.65 if ag.stage == "baby" else (0.9 if ag.stage == "elder" else 0.8)

            # body cylinder
            body_h = 0.9 * scale
            body_r = 0.28 * scale
            cyls.append([wx, wy, wz,  body_r*2, body_h, body_r*2,  r, g, b, 1.0])
            # head sphere
            head_r = 0.22 * scale
            sphs.append([wx, wy + body_h + head_r, wz,
                         head_r*2, head_r*2, head_r*2,
                         min(1.0, r*1.1), min(1.0, g*1.1), min(1.0, b*1.1), 1.0])

            # shadow disc (flat sphere)
            shad.append([wx, wy + 0.02, wz,  body_r*2.5, 0.04, body_r*2.5,  0.0, 0.0, 0.0, 0.35])

        def _batch_draw(vao, data):
            if data:
                arr = np.array(data, dtype=np.float32)
                self._inst_buf.orphan(size=len(data)*10*4)
                self._inst_buf.write(arr.tobytes())
                vao.render(moderngl.TRIANGLES, instances=len(data))

        _batch_draw(self._vao_sph, shad)
        _batch_draw(self._vao_cyl, cyls)
        _batch_draw(self._vao_sph, sphs)

    def _agent_color(self, ag):
        from ai.actions import A_FLEE_FIRE, A_FIND_FOOD, A_FIND_WATER, A_BUILD, A_SEEK_MATE
        action_cols = {
            A_FLEE_FIRE:  (1.0, 0.25, 0.0),
            A_FIND_WATER: (0.2, 0.55, 1.0),
            A_FIND_FOOD:  (0.5, 1.0, 0.25),
            A_BUILD:      (0.9, 0.7, 0.2),
            A_SEEK_MATE:  (1.0, 0.35, 0.75),
        }
        if ag.current_action in action_cols:
            return action_cols[ag.current_action]
        if ag.stage == "baby":
            return 0.95, 0.85, 0.40
        if ag.stage == "elder":
            return 0.65, 0.62, 0.75
        return (0.82, 0.82, 0.90) if ag.gender == "male" else (0.92, 0.58, 0.68)

    # ── particles ────────────────────────────────────────────
    def _update_draw_particles(self, world, dt):
        weather = world.weather
        if weather not in ("rain", "storm", "blizzard"):
            self._particles = []
            return

        self._part_timer += dt
        rate = {"rain": 0.02, "storm": 0.01, "blizzard": 0.015}.get(weather, 0.03)
        while self._part_timer > rate:
            self._part_timer -= rate
            x = self.cam_target[0] + (hash(len(self._particles)) % 60 - 30)
            z = self.cam_target[2] + ((hash(len(self._particles) * 7 + 3)) % 60 - 30)
            y = 18.0
            vx = -0.5 if weather == "blizzard" else 0.0
            self._particles.append([x, y, z, vx, -4.5, 0.0, 0.85])

        keep = []
        for p in self._particles:
            p[0] += p[3] * dt
            p[1] += p[4] * dt
            p[2] += p[5] * dt
            if p[1] > -1.0:
                keep.append(p)
        self._particles = keep[:800]

        if self._particles:
            arr = np.array([[p[0], p[1], p[2], p[6]] for p in self._particles], dtype=np.float32)
            self._part_vbo.orphan(size=len(arr)*4*4)
            self._part_vbo.write(arr.tobytes())
            _, mvp = self._get_view_proj()
            self.prog_part["u_mvp"].write(mvp.T.tobytes())
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self._part_vao.render(moderngl.POINTS, vertices=len(self._particles))