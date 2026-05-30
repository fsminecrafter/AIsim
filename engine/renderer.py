"""
3D GPU Renderer — ModernGL (OpenGL 3.3 core)
============================================
Full perspective-projection 3D rendering:
  - Terrain heightmap with per-tile 3D box quads
  - Sand strips near rivers
  - Trees  → green cylinders + sphere canopy
  - Berry bushes → small sphere clusters
  - Wheat → thin vertical box stalks
  - Rocks → scaled box clusters
  - Agents → capsule (cylinder body + sphere head)
  - Fireplace → stone ring mesh
  - House → box structure
  - Fire glow (additive billboard quads)
  - Day/night ambient tint, weather particles
"""

import numpy as np
import moderngl
import math
from engine.config import *
from world.world import (T_GRASS, T_DIRT, T_ROCK, T_RIVER,
                          O_NONE, O_TREE, O_BERRY, O_WHEAT, O_ROCK_OBJ,
                          S_NONE, S_HOUSE, S_FIREPLACE)


# ─── math helpers ────────────────────────────────────────────

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
    eye    = np.array(eye,    dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up     = np.array(up,     dtype=np.float32)
    f = center - eye;  f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s;  m[0, 3] = -np.dot(s, eye)
    m[1, :3] = u;  m[1, 3] = -np.dot(u, eye)
    m[2, :3] = -f; m[2, 3] =  np.dot(f, eye)
    return m


def _mat_mul(a, b):
    return (a @ b).astype(np.float32)


# ─── mesh builders ───────────────────────────────────────────

def _box_mesh():
    """Unit cube centred at origin. Returns (verts_f32, normals_f32, indices_u32)."""
    faces = [
        ([ 0.5,-0.5,-0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5],[1,0,0]),
        ([-0.5,-0.5, 0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[-0.5, 0.5, 0.5],[-1,0,0]),
        ([-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5],[0,1,0]),
        ([-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[0,-1,0]),
        ([-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5],[0,0,1]),
        ([ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5],[0,0,-1]),
    ]
    verts, norms, idxs, base = [], [], [], 0
    for v0, v1, v2, v3, n in faces:
        for v in [v0, v1, v2, v3]:
            verts.extend(v); norms.extend(n)
        idxs.extend([base, base+1, base+2, base, base+2, base+3])
        base += 4
    return (np.array(verts, np.float32).reshape(-1,3),
            np.array(norms, np.float32).reshape(-1,3),
            np.array(idxs,  np.uint32))


def _cylinder_mesh(segs=12):
    """Cylinder radius 0.5, height 1, base at y=0."""
    verts, norms, idxs, base = [], [], [], 0
    for i in range(segs):
        a0 = 2*math.pi * i / segs
        a1 = 2*math.pi * (i+1) / segs
        x0,z0 = 0.5*math.cos(a0), 0.5*math.sin(a0)
        x1,z1 = 0.5*math.cos(a1), 0.5*math.sin(a1)
        for x,z,y in [(x0,z0,0),(x1,z1,0),(x1,z1,1),(x0,z0,1)]:
            verts.extend([x,y,z]); norms.extend([x*2,0,z*2])
        idxs.extend([base, base+1, base+2, base, base+2, base+3])
        base += 4
    # top cap
    for i in range(segs):
        a0 = 2*math.pi * i / segs
        a1 = 2*math.pi * (i+1) / segs
        for x,z,y in [(0,0,1),(0.5*math.cos(a0),0.5*math.sin(a0),1),(0.5*math.cos(a1),0.5*math.sin(a1),1)]:
            verts.extend([x,y,z]); norms.extend([0,1,0])
        idxs.extend([base, base+1, base+2]); base += 3
    return (np.array(verts, np.float32).reshape(-1,3),
            np.array(norms, np.float32).reshape(-1,3),
            np.array(idxs,  np.uint32))


def _sphere_mesh(segs=12):
    """Unit sphere radius 0.5 centred at origin."""
    verts, norms, idxs, base = [], [], [], 0
    rings = max(segs // 2, 3)
    for i in range(rings):
        lat0 = math.pi * (-0.5 + i / rings)
        lat1 = math.pi * (-0.5 + (i+1) / rings)
        for j in range(segs):
            lon0 = 2*math.pi * j / segs
            lon1 = 2*math.pi * (j+1) / segs
            for lat, lon in [(lat0,lon0),(lat0,lon1),(lat1,lon1),(lat1,lon0)]:
                x = 0.5*math.cos(lat)*math.cos(lon)
                y = 0.5*math.sin(lat)
                z = 0.5*math.cos(lat)*math.sin(lon)
                verts.extend([x,y,z]); norms.extend([x*2,y*2,z*2])
            idxs.extend([base, base+1, base+2, base, base+2, base+3])
            base += 4
    return (np.array(verts, np.float32).reshape(-1,3),
            np.array(norms, np.float32).reshape(-1,3),
            np.array(idxs,  np.uint32))


def _pack_mesh(verts, norms):
    """Interleave pos+normal into one float32 array: [x,y,z, nx,ny,nz, ...]"""
    return np.hstack([verts, norms]).astype(np.float32)


# ─── GLSL shaders ────────────────────────────────────────────

VERT_3D = """
#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
// per-instance (divisor=1) — locations pinned so Windows drivers match Linux
layout(location = 2) in vec3 in_inst_pos;
layout(location = 3) in vec3 in_inst_scale;
layout(location = 4) in vec4 in_inst_color;

uniform mat4 u_mvp;
uniform vec3 u_light_dir;
uniform float u_night;
uniform float u_weather_dark;

out vec4 v_color;

void main() {
    vec3 world = in_inst_pos + in_pos * in_inst_scale;
    gl_Position = u_mvp * vec4(world, 1.0);

    float diff  = max(dot(normalize(in_normal), normalize(u_light_dir)), 0.0);
    float light = 0.38 + 0.62 * diff;
    vec3  col   = in_inst_color.rgb * light;

    // distance fog
    float dist = length(world);
    float fog  = clamp(dist * 0.003, 0.0, 0.45);
    vec3  sky  = mix(vec3(0.52, 0.63, 0.76), vec3(0.02,0.04,0.12), u_night);
    col = mix(col, sky, fog);

    // night + weather dim
    col = mix(col, col * vec3(0.14,0.16,0.28), u_night * 0.68);
    col *= (1.0 - u_weather_dark * 0.4);

    v_color = vec4(col, in_inst_color.a);
}
"""

FRAG_3D = """
#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() { frag_color = v_color; }
"""

FIRE_VERT = """
#version 330 core
in vec2 in_quad;
// per-instance
in vec3 in_fire_pos;
in float in_fire_intensity;

uniform mat4 u_mvp;
uniform mat4 u_view;

out float v_intensity;
out vec2  v_uv;

void main() {
    float sz = 0.5 + in_fire_intensity * 0.014;
    vec3 cam_right = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 cam_up    = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);
    vec3 world = in_fire_pos
               + cam_right * in_quad.x * sz
               + cam_up    * in_quad.y * sz * 1.5;
    gl_Position = u_mvp * vec4(world, 1.0);
    v_intensity = in_fire_intensity / 80.0;
    v_uv = in_quad * 0.5 + 0.5;
}
"""

FIRE_FRAG = """
#version 330 core
in float v_intensity;
in vec2  v_uv;
out vec4 frag_color;
void main() {
    vec2 d = v_uv - vec2(0.5, 0.35);
    float r = length(d * vec2(1.0, 1.35));
    if (r > 0.5) discard;
    float core = clamp(1.0 - r * 2.0, 0.0, 1.0);
    vec3 col = mix(vec3(0.95,0.28,0.0), vec3(1.0,0.95,0.2), core);
    frag_color = vec4(col, v_intensity * core * 0.88);
}
"""

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
    float a = u_night * 0.42 + u_weather_dark * 0.28;
    vec3  c = mix(vec3(0.0), vec3(0.02,0.04,0.14), u_night);
    frag_color = vec4(c, a);
}
"""

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
void main() { frag_color = vec4(0.72, 0.82, 1.0, v_alpha); }
"""


# ─── Renderer ────────────────────────────────────────────────

class Renderer:
    def __init__(self, ctx: moderngl.Context, screen_w, screen_h):
        self.ctx      = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h

        # camera state
        self.cam_pitch  = -52.0
        self.cam_yaw    = 0.0
        self.cam_dist   = 28.0
        self.cam_target = np.array([WORLD_W * TILE_SIZE / 2,
                                     0.0,
                                     WORLD_H * TILE_SIZE / 2], dtype=np.float32)
        self.zoom = 1.0

        # compile programs
        self.prog_3d   = ctx.program(vertex_shader=VERT_3D,      fragment_shader=FRAG_3D)
        self.prog_fire = ctx.program(vertex_shader=FIRE_VERT,    fragment_shader=FIRE_FRAG)
        self.prog_over = ctx.program(vertex_shader=OVERLAY_VERT, fragment_shader=OVERLAY_FRAG)
        self.prog_part = ctx.program(vertex_shader=PARTICLE_VERT,fragment_shader=PARTICLE_FRAG)

        # build meshes
        box_v,  box_n,  box_i  = _box_mesh()
        cyl_v,  cyl_n,  cyl_i  = _cylinder_mesh(12)
        sph_v,  sph_n,  sph_i  = _sphere_mesh(12)

        # VBOs (interleaved pos+normal) and IBOs
        self._box_vbo  = ctx.buffer(_pack_mesh(box_v,  box_n).tobytes())
        self._cyl_vbo  = ctx.buffer(_pack_mesh(cyl_v,  cyl_n).tobytes())
        self._sph_vbo  = ctx.buffer(_pack_mesh(sph_v,  sph_n).tobytes())
        self._box_ibo  = ctx.buffer(box_i.tobytes())
        self._cyl_ibo  = ctx.buffer(cyl_i.tobytes())
        self._sph_ibo  = ctx.buffer(sph_i.tobytes())

        # shared instance buffer: pos(3)+scale(3)+color(4) = 10 floats per instance
        MAX_INST = WORLD_W * WORLD_H * 4
        self._inst_buf = ctx.buffer(reserve=MAX_INST * 10 * 4, dynamic=True)

        # VAOs — each attribute from inst_buf needs its OWN tuple entry
        self._vao_box  = self._make_vao(self._box_vbo,  self._box_ibo)
        self._vao_cyl  = self._make_vao(self._cyl_vbo,  self._cyl_ibo)
        self._vao_sph  = self._make_vao(self._sph_vbo,  self._sph_ibo)

        # fire billboard VBO + instance buffer
        fire_quad = np.array([[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]], dtype=np.float32)
        self._fire_quad_vbo = ctx.buffer(fire_quad.tobytes())
        # fire instance: pos(3)+intensity(1) = 4 floats packed
        self._fire_inst_buf = ctx.buffer(reserve=WORLD_W*WORLD_H*4*4, dynamic=True)
        self._fire_vao = ctx.vertex_array(self.prog_fire, [
            (self._fire_quad_vbo, '2f',    'in_quad'),
            (self._fire_inst_buf, '3f 1f /i', 'in_fire_pos', 'in_fire_intensity'),
        ])

        # fullscreen overlay
        fs = np.array([[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]], dtype=np.float32)
        self._fs_vbo  = ctx.buffer(fs.tobytes())
        self._over_vao = ctx.vertex_array(self.prog_over,
                                           [(self._fs_vbo, '2f', 'in_vert')])

        # particles
        self._part_vbo = ctx.buffer(reserve=4000*4*4, dynamic=True)
        self._part_vao = ctx.vertex_array(self.prog_part,
                                           [(self._part_vbo, '3f 1f', 'in_pos', 'in_alpha')])
        self._particles  = []
        self._part_timer = 0.0

        # GL state
        ctx.enable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self._height_map = None

    def _make_vao(self, mesh_vbo, ibo):
        """
        Build an instanced VAO for prog_3d.
        mesh_vbo: interleaved [pos(3), normal(3)] per vertex, stride=24 bytes
        inst_buf: [pos(3), scale(3), color(4)] per instance, stride=40 bytes
        Each instance attribute MUST be its own tuple in the content list.
        """
        return self.ctx.vertex_array(self.prog_3d, [
            # per-instance attributes: /1 means advance buffer once per instance
            (self._inst_buf, '3f 3f 4f /i', 'in_inst_pos', 'in_inst_scale', 'in_inst_color'),
            # per-vertex attributes
            (mesh_vbo,       '3f 3f',        'in_pos', 'in_normal'),
        ], ibo)

    # ── camera ───────────────────────────────────────────────
    def pan(self, dx, dy):
        yr = math.radians(self.cam_yaw)
        self.cam_target[0] += math.cos(yr) * dx - math.sin(yr) * dy
        self.cam_target[2] += math.sin(yr) * dx + math.cos(yr) * dy

    def zoom_by(self, factor):
        self.cam_dist = max(5.0, min(80.0, self.cam_dist / factor))

    def orbit(self, dyaw, dpitch):
        self.cam_yaw   = (self.cam_yaw + dyaw) % 360
        self.cam_pitch = max(-85, min(-15, self.cam_pitch + dpitch))

    def _visible_tile_range(self, margin=4):
        """
        Return (gx_min, gx_max, gy_min, gy_max) of tiles that are roughly
        visible given the current camera target and distance.
        Tiles outside this range are skipped — the single biggest perf win.
        """
        T = TILE_SIZE
        # Half-span in world units visible at current distance / zoom
        # Generous factor so we never clip things that peek in from the edge.
        half = self.cam_dist * 1.35 + 20.0
        cx = self.cam_target[0] / T
        cy = self.cam_target[2] / T
        x0 = max(0, int(cx - half / T) - margin)
        x1 = min(WORLD_W, int(cx + half / T) + margin + 1)
        y0 = max(0, int(cy - half / T) - margin)
        y1 = min(WORLD_H, int(cy + half / T) + margin + 1)
        return x0, x1, y0, y1

    def screen_to_world(self, sx, sy):
        norm_x = (sx / max(self.screen_w, 1) - 0.5) * 2
        norm_y = -(sy / max(self.screen_h, 1) - 0.5) * 2
        scale  = self.cam_dist * 0.28
        return (self.cam_target[0] + norm_x * scale,
                self.cam_target[2] - norm_y * scale)

    def _get_view_proj(self):
        aspect = max(self.screen_w, 1) / max(self.screen_h, 1)
        proj   = _perspective(45.0 / self.zoom, aspect, 0.5, 600.0)
        pr = math.radians(self.cam_pitch)
        yr = math.radians(self.cam_yaw)
        eye_off = np.array([
             self.cam_dist * math.cos(pr) * math.sin(yr),
            -self.cam_dist * math.sin(pr),
             self.cam_dist * math.cos(pr) * math.cos(yr),
        ], dtype=np.float32)
        eye  = self.cam_target + eye_off
        view = _look_at(eye, self.cam_target, [0, 1, 0])
        return view, _mat_mul(proj, view)

    def _light_dir(self, world):
        tod = world.time_of_day
        a   = math.pi * tod
        lx  = math.cos(a); ly = math.sin(a) + 0.3; lz = 0.5
        ln  = math.sqrt(lx*lx + ly*ly + lz*lz)
        return [lx/ln, ly/ln, lz/ln]

    # ── render ───────────────────────────────────────────────
    def render(self, world, agents, dt=0.016):
        ctx = self.ctx
        ctx.clear(0.45, 0.60, 0.75, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = '<'
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        if self._height_map is None:
            self._build_height_map(world)

        view, mvp = self._get_view_proj()
        mvp_b  = mvp.T.tobytes()
        view_b = view.T.tobytes()

        tod          = world.time_of_day
        night        = max(0.0, min(1.0, abs(tod - 0.5) * 4 - 1.0))
        weather_dark = {"storm": 0.35, "blizzard": 0.28, "rain": 0.12}.get(world.weather, 0.0)

        # prog_3d uniforms
        self.prog_3d["u_mvp"].write(mvp_b)
        self.prog_3d["u_light_dir"].value = tuple(self._light_dir(world))
        self.prog_3d["u_night"].value        = night
        self.prog_3d["u_weather_dark"].value = weather_dark

        # 1. Terrain
        self._draw_terrain(world)

        # 2. Objects
        self._draw_objects(world)

        # 3. Structures
        self._draw_structures(world)

        # 4. Fire (additive) — only visible tiles
        gx0, gx1, gy0, gy1 = self._visible_tile_range(margin=2)
        fire_mask = np.zeros_like(world.fire_intensity, dtype=bool)
        fire_mask[gy0:gy1, gx0:gx1] = world.fire_intensity[gy0:gy1, gx0:gx1] > 0
        fire_tiles = np.argwhere(fire_mask)
        if len(fire_tiles) > 0:
            fbuf = np.zeros((len(fire_tiles), 4), dtype=np.float32)
            for i, (fy, fx) in enumerate(fire_tiles):
                fbuf[i,0] = fx * TILE_SIZE + TILE_SIZE/2
                fbuf[i,1] = self._h(fy, fx) * 0.5 + 0.6
                fbuf[i,2] = fy * TILE_SIZE + TILE_SIZE/2
                fbuf[i,3] = world.fire_intensity[fy, fx]
            self._fire_inst_buf.orphan(size=len(fire_tiles)*4*4)
            self._fire_inst_buf.write(fbuf.tobytes())
            self.prog_fire["u_mvp"].write(mvp_b)
            self.prog_fire["u_view"].write(view_b)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            ctx.disable(moderngl.DEPTH_TEST)
            self._fire_vao.render(moderngl.TRIANGLES, instances=len(fire_tiles))
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # 5. Agents
        self._draw_agents(agents)

        # 6. Particles
        self._update_draw_particles(world, dt, mvp_b)

        # 7. Overlay
        ctx.disable(moderngl.DEPTH_TEST)
        self.prog_over["u_night"].value        = night
        self.prog_over["u_weather_dark"].value = weather_dark
        self._over_vao.render(moderngl.TRIANGLES)
        ctx.enable(moderngl.DEPTH_TEST)

    # ── batch helper ─────────────────────────────────────────
    def _batch(self, vao, instances):
        if not instances:
            return
        arr = np.array(instances, dtype=np.float32)
        sz  = len(instances) * 10 * 4
        self._inst_buf.orphan(size=sz)
        self._inst_buf.write(arr.tobytes())
        vao.render(moderngl.TRIANGLES, instances=len(instances))

    # ── height map ───────────────────────────────────────────
    def _build_height_map(self, world):
        H, W = WORLD_H, WORLD_W
        hm = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                t = world.terrain[y, x]
                if   t == T_ROCK:  hm[y,x] = 0.9 + ((y*97+x*13) % 10) * 0.06
                elif t == T_RIVER: hm[y,x] = -0.15
                elif t == T_DIRT:  hm[y,x] = 0.0
                else:              hm[y,x] = 0.08   # grass
        self._height_map = hm

    def _h(self, ty, tx):
        if self._height_map is None:
            return 0.0
        return float(self._height_map[
            max(0, min(WORLD_H-1, ty)),
            max(0, min(WORLD_W-1, tx))])

    def _is_near_river(self, world, gy, gx):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = gy+dy, gx+dx
                if 0 <= ny < WORLD_H and 0 <= nx < WORLD_W:
                    if world.terrain[ny, nx] == T_RIVER:
                        return True
        return False

    # ── terrain ──────────────────────────────────────────────
    def _draw_terrain(self, world):
        T   = TILE_SIZE
        buf = []
        gx0, gx1, gy0, gy1 = self._visible_tile_range(margin=2)
        for gy in range(gy0, gy1):
            for gx in range(gx0, gx1):
                t  = world.terrain[gy, gx]
                h  = self._h(gy, gx)
                wx = gx * T + T/2
                wz = gy * T + T/2
                wy = h * 0.5 - 0.5

                sand = (t in (T_GRASS, T_DIRT)) and self._is_near_river(world, gy, gx)
                r, g, b = self._terrain_col(t, gy, gx, sand, world)

                fi = world.fire_intensity[gy, gx]
                if fi > 0:
                    ft = min(fi / 60.0, 0.65)
                    r = r*(1-ft) + 0.85*ft
                    g = g*(1-ft) + 0.14*ft
                    b = b*(1-ft)

                tile_h = max(0.3, abs(h) + 0.3)
                buf.append([wx, wy, wz,  T*0.99, tile_h, T*0.99,  r, g, b, 1.0])

        self._batch(self._vao_box, buf)

    def _terrain_col(self, t, gy, gx, sand, world):
        if t == T_RIVER:  return 0.12, 0.37, 0.72
        if sand:          return 0.85, 0.76, 0.48
        if t == T_ROCK:   return 0.54, 0.52, 0.50
        season = world.season_name
        if t == T_GRASS:
            if season == "autumn": return 0.56, 0.42, 0.12
            if season == "winter": return 0.80, 0.83, 0.88
            return 0.22, 0.46, 0.15
        return 0.55, 0.38, 0.22   # dirt

    # ── objects ──────────────────────────────────────────────
    def _draw_objects(self, world):
        T = TILE_SIZE
        cyl_buf, sph_buf, box_buf = [], [], []
        gx0, gx1, gy0, gy1 = self._visible_tile_range(margin=3)

        for gy in range(gy0, gy1):
            for gx in range(gx0, gx1):
                ot = world.obj_type[gy, gx]
                if ot == O_NONE:
                    continue
                og = world.obj_growth[gy, gx]
                h  = self._h(gy, gx)
                wx = gx * T + T/2
                wz = gy * T + T/2
                wy = h * 0.5

                if ot == O_TREE:
                    th = 1.5 + og * 2.5
                    tr = 0.20 + og * 0.14
                    cyl_buf.append([wx, wy, wz,  tr*2, th, tr*2,  0.50, 0.34, 0.16, 1.0])
                    cr = 1.2 * og + 0.3
                    g_r, g_g, g_b = (0.28 + og*0.10, 0.60 - og*0.04, 0.12)
                    sph_buf.append([wx, wy + th, wz,  cr*2, cr*2, cr*2,  g_r, g_g, g_b, 1.0])

                elif ot == O_BERRY:
                    sz = 0.45 + og * 0.32
                    sph_buf.append([wx, wy + sz*0.5, wz,  sz, sz*0.65, sz,  0.18, 0.48, 0.14, 1.0])
                    if og > 0.5:
                        for bdy, bdx in [(-0.18, 0.14),(0.14, -0.10),(0.10, 0.18)]:
                            sph_buf.append([wx + bdx*T*0.15, wy + sz*0.82, wz + bdy*T*0.15,
                                            0.13, 0.13, 0.13,  0.82, 0.14, 0.14, 1.0])

                elif ot == O_WHEAT:
                    hs = 0.6 + og * 0.9
                    r2, g2, b2 = (0.85, 0.72, 0.12) if og > 0.7 else (0.45, 0.70, 0.22)
                    for ox, oz in [(0,0),(0.14,0.14),(-0.12,-0.08)]:
                        box_buf.append([wx + ox, wy + hs*0.5, wz + oz,
                                        0.09, hs, 0.09,  r2, g2, b2, 1.0])

                elif ot == O_ROCK_OBJ:
                    box_buf.append([wx, wy+0.22, wz,  0.50, 0.34, 0.46,  0.56,0.53,0.52, 1.0])
                    box_buf.append([wx+0.18, wy+0.15, wz-0.08,  0.30,0.24,0.28,  0.48,0.46,0.44, 1.0])

        self._batch(self._vao_cyl, cyl_buf)
        self._batch(self._vao_sph, sph_buf)
        self._batch(self._vao_box, box_buf)

    # ── structures ───────────────────────────────────────────
    def _draw_structures(self, world):
        T = TILE_SIZE
        buf = []
        gx0, gx1, gy0, gy1 = self._visible_tile_range(margin=2)
        for gy in range(gy0, gy1):
            for gx in range(gx0, gx1):
                s = world.structures[gy, gx]
                if s == S_NONE:
                    continue
                h  = self._h(gy, gx)
                wx = gx * T + T/2
                wz = gy * T + T/2
                wy = h * 0.5

                if s == S_HOUSE:
                    buf.append([wx, wy+0.8, wz,  T*0.88, 1.6, T*0.88,  0.62,0.48,0.32, 1.0])
                    buf.append([wx, wy+1.72, wz,  T*0.92, 0.38, T*0.92,  0.44,0.30,0.18, 1.0])
                elif s == S_FIREPLACE:
                    for dy, dx in [(-0.22,-0.22),(-0.22,0.22),(0.22,-0.22),(0.22,0.22)]:
                        buf.append([wx+dx, wy+0.22, wz+dy,  0.20,0.44,0.20,  0.30,0.28,0.28, 1.0])
                    buf.append([wx, wy+0.10, wz,  0.26,0.10,0.26,  0.94,0.40,0.05, 1.0])

        self._batch(self._vao_box, buf)

    # ── agents ───────────────────────────────────────────────
    def _draw_agents(self, agents):
        gx0, gx1, gy0, gy1 = self._visible_tile_range(margin=4)
        live = [a for a in agents if not a.is_dead
                and gx0 <= a.x < gx1 and gy0 <= a.y < gy1]
        if not live:
            return
        T    = TILE_SIZE
        cyls = []
        sphs = []
        shad = []

        for ag in live:
            ty = max(0, min(WORLD_H-1, int(ag.y)))
            tx = max(0, min(WORLD_W-1, int(ag.x)))
            h  = self._h(ty, tx)
            wx = ag.x * T + T/2
            wz = ag.y * T + T/2
            wy = h * 0.5

            sc = 0.65 if ag.stage == "baby" else (0.88 if ag.stage == "elder" else 0.80)
            bh = 0.90 * sc
            br = 0.27 * sc
            r, g, b = self._agent_col(ag)

            shad.append([wx, wy+0.02, wz,  br*2.4, 0.04, br*2.4,  0.0,0.0,0.0, 0.32])
            cyls.append([wx, wy,      wz,  br*2,   bh,   br*2,    r, g, b, 1.0])
            hr = 0.21 * sc
            sphs.append([wx, wy+bh+hr, wz,  hr*2, hr*2, hr*2,
                         min(1.0,r*1.08), min(1.0,g*1.08), min(1.0,b*1.08), 1.0])

        self._batch(self._vao_sph, shad)
        self._batch(self._vao_cyl, cyls)
        self._batch(self._vao_sph, sphs)

    def _agent_col(self, ag):
        from ai.actions import A_FLEE_FIRE, A_FIND_FOOD, A_FIND_WATER, A_BUILD, A_SEEK_MATE
        ac = {A_FLEE_FIRE:(1.0,0.22,0.0), A_FIND_WATER:(0.20,0.55,1.0),
              A_FIND_FOOD:(0.48,1.0,0.22), A_BUILD:(0.90,0.70,0.20),
              A_SEEK_MATE:(1.0,0.32,0.75)}
        if ag.current_action in ac:
            return ac[ag.current_action]
        if ag.stage == "baby":  return 0.95, 0.85, 0.40
        if ag.stage == "elder": return 0.65, 0.62, 0.75
        return (0.82,0.82,0.90) if ag.gender == "male" else (0.92,0.58,0.68)

    # ── weather particles ─────────────────────────────────────
    def _update_draw_particles(self, world, dt, mvp_b):
        weather = world.weather
        if weather not in ("rain", "storm", "blizzard"):
            self._particles = []
            return

        self._part_timer += dt
        rate = {"rain":0.025, "storm":0.012, "blizzard":0.018}.get(weather, 0.025)
        while self._part_timer > rate and len(self._particles) < 700:
            self._part_timer -= rate
            n = len(self._particles)
            x = self.cam_target[0] + ((n * 73 + 11) % 60 - 30)
            z = self.cam_target[2] + ((n * 37 + 5)  % 60 - 30)
            vx = -1.2 if weather == "blizzard" else 0.0
            self._particles.append([x, 20.0, z, vx, -5.5, 0.0, 0.80])

        keep = []
        for p in self._particles:
            p[0] += p[3]*dt; p[1] += p[4]*dt; p[2] += p[5]*dt
            if p[1] > -1.0:
                keep.append(p)
        self._particles = keep

        if not self._particles:
            return
        arr = np.array([[p[0],p[1],p[2],p[6]] for p in self._particles], dtype=np.float32)
        self._part_vbo.orphan(size=len(arr)*4*4)
        self._part_vbo.write(arr.tobytes())
        self.prog_part["u_mvp"].write(mvp_b)
        try:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        except Exception:
            pass
        self._part_vao.render(moderngl.POINTS, vertices=len(self._particles))