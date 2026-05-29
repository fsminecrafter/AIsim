/*
 * sim_kernels.c — C99 hot-path kernels for the survival sim
 *
 * Compiled as a shared library (.so) and called via ctypes from Python.
 *
 * Exports:
 *   fire_spread_step(...)   — one iteration of fire spread across the map
 *   bfs_find_nearest(...)   — BFS to find nearest tile of a given type
 *
 * Build:
 *   gcc -O3 -march=native -ffast-math -shared -fPIC \
 *       -o sim_kernels.so sim_kernels.c
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define WORLD_W 80
#define WORLD_H 80
#define T_RIVER 3
#define O_TREE  1
#define O_WHEAT 3

/* ─────────────────────────────────────────────────────────────
 * Lightweight LCG for per-cell pseudo-random numbers
 * ───────────────────────────────────────────────────────────*/
static inline uint32_t lcg(uint32_t seed) {
    return seed * 1664525u + 1013904223u;
}

/* ─────────────────────────────────────────────────────────────
 * fire_spread_step
 *
 * Inputs (all row-major float32 arrays of size H×W):
 *   fire_intensity  — 0 = no fire, >0 = burning
 *   fire_fuel       — fuel remaining (decremented per call)
 *   terrain         — int32 terrain type
 *   obj_type        — int32 object type
 *   moisture        — 0.0–1.0
 *   rng_seeds       — uint32 per-cell seed (updated in place)
 *
 * Scalar inputs:
 *   weather_fire_mult — float, spread probability multiplier
 *   dt                — float, seconds this tick
 *
 * Returns: number of new ignitions (int32)
 * ───────────────────────────────────────────────────────────*/
int32_t fire_spread_step(
        float   *fire_intensity,
        float   *fire_fuel,
        int32_t *terrain,
        int32_t *obj_type,
        float   *moisture,
        uint32_t *rng_seeds,
        float   weather_fire_mult,
        float   dt)
{
    /* staging buffer so we don't ignite new cells mid-iteration */
    static uint8_t new_fire[WORLD_H * WORLD_W];
    memset(new_fire, 0, sizeof(new_fire));

    int32_t n_ignitions = 0;
    float SPREAD_BASE = 0.15f;
    int   RADIUS      = 2;

    for (int y = 0; y < WORLD_H; y++) {
        for (int x = 0; x < WORLD_W; x++) {
            int i = y * WORLD_W + x;
            if (fire_intensity[i] <= 0.0f) continue;

            /* burn fuel */
            fire_fuel[i] -= 3.0f * dt;
            if (fire_fuel[i] <= 0.0f) {
                fire_intensity[i] = 0.0f;
                fire_fuel[i]      = 0.0f;
                continue;
            }

            /* spread attempt */
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny < 0 || ny >= WORLD_H || nx < 0 || nx >= WORLD_W) continue;
                    int ni = ny * WORLD_W + nx;
                    if (terrain[ni] == T_RIVER)       continue;
                    if (fire_intensity[ni] > 0.0f)    continue;
                    if (new_fire[ni])                  continue;
                    int ot = obj_type[ni];
                    if (ot != O_TREE && ot != O_WHEAT) continue;

                    float moist  = moisture[ni];
                    float chance = SPREAD_BASE * (1.0f - moist) * weather_fire_mult;

                    /* advance RNG */
                    rng_seeds[ni] = lcg(rng_seeds[ni]);
                    float rv = (float)(rng_seeds[ni] >> 16) / 65535.0f;

                    if (rv < chance * dt) {
                        new_fire[ni] = 1;
                        n_ignitions++;
                    }
                }
            }
        }
    }

    /* apply new ignitions */
    for (int i = 0; i < WORLD_H * WORLD_W; i++) {
        if (new_fire[i]) {
            fire_intensity[i] = 30.0f;
            fire_fuel[i]      = 40.0f;
        }
    }
    return n_ignitions;
}


/* ─────────────────────────────────────────────────────────────
 * bfs_find_nearest
 *
 * BFS on the grid to find nearest cell with obj_type == target_type
 * and obj_growth >= 1.0 within max_radius manhattan steps.
 *
 * Returns (out_y, out_x) via pointers. Sets both to -1 if not found.
 * ───────────────────────────────────────────────────────────*/
void bfs_find_nearest(
        int32_t *obj_type,
        float   *obj_growth,
        int32_t *terrain,
        int      start_y,
        int      start_x,
        int      target_type,
        int      max_radius,
        int     *out_y,
        int     *out_x)
{
    *out_y = -1;
    *out_x = -1;

    /* simple queue for BFS */
    static int qy[WORLD_H * WORLD_W];
    static int qx[WORLD_H * WORLD_W];
    static uint8_t visited[WORLD_H * WORLD_W];
    memset(visited, 0, sizeof(visited));

    int head = 0, tail = 0;
    qy[tail] = start_y;
    qx[tail] = start_x;
    tail++;
    visited[start_y * WORLD_W + start_x] = 1;

    int DY[4] = {-1, 1, 0, 0};
    int DX[4] = { 0, 0,-1, 1};

    while (head < tail) {
        int cy = qy[head];
        int cx = qx[head];
        head++;

        int dist = abs(cy - start_y) + abs(cx - start_x);
        if (dist > max_radius) continue;

        int ci = cy * WORLD_W + cx;
        if (obj_type[ci] == target_type && obj_growth[ci] >= 1.0f) {
            *out_y = cy;
            *out_x = cx;
            return;
        }

        for (int d = 0; d < 4; d++) {
            int ny = cy + DY[d];
            int nx = cx + DX[d];
            if (ny < 0 || ny >= WORLD_H || nx < 0 || nx >= WORLD_W) continue;
            int ni = ny * WORLD_W + nx;
            if (visited[ni]) continue;
            if (terrain[ni] == T_RIVER) continue;
            visited[ni] = 1;
            qy[tail] = ny;
            qx[tail] = nx;
            tail++;
        }
    }
}


/* ─────────────────────────────────────────────────────────────
 * agent_batch_vitals_decay
 *
 * Updates hunger/thirst/warmth/mood for an array of agents.
 * agents_data: float32 array, 8 floats per agent:
 *   [hunger, thirst, warmth, mood, health, near_fire, under_roof, stage_idx]
 *
 * Scalars:
 *   dt, w_hunger, w_thirst, w_warmth, w_mood, w_cold
 *   (where w_ = combined weather+season multiplier passed from Python)
 *
 * Writes updated values back in place.
 * Returns bitmask of agents that reached health 0 (dead).
 * ───────────────────────────────────────────────────────────*/
uint64_t agent_batch_vitals(
        float *agents_data,
        int    n_agents,
        float  dt,
        float  w_hunger,
        float  w_thirst,
        float  w_warmth,
        float  w_mood,
        float  w_cold)
{
    /* base decay rates per real second (×60 because config is per minute) */
    const float BASE_HUNGER = 1.0f / 60.0f;
    const float BASE_THIRST = 2.0f / 60.0f;
    const float BASE_WARMTH = 0.5f / 60.0f;
    const float BASE_MOOD   = 0.3f / 60.0f;

    uint64_t dead_mask = 0;

    for (int i = 0; i < n_agents && i < 64; i++) {
        float *ag = agents_data + i * 8;
        float hunger   = ag[0];
        float thirst   = ag[1];
        float warmth   = ag[2];
        float mood     = ag[3];
        float health   = ag[4];
        float near_fire = ag[5];
        float under_roof = ag[6];
        float stage    = ag[7];   /* 0=baby 1=adult 2=elder */

        float baby_mod = (stage < 0.5f) ? 0.7f : 1.0f;

        hunger = fmaxf(0.0f, hunger - BASE_HUNGER * w_hunger * dt * baby_mod * 60.0f);
        thirst = fmaxf(0.0f, thirst - BASE_THIRST * w_thirst * dt * 60.0f);
        mood   = fmaxf(0.0f, mood   - BASE_MOOD   * w_mood   * dt * 60.0f);

        float wloss = BASE_WARMTH * w_warmth * w_cold * dt * 60.0f;
        if (near_fire > 0.5f)  wloss -= 3.0f * dt;
        if (under_roof > 0.5f) wloss *= 0.5f;
        warmth = fmaxf(0.0f, fminf(100.0f, warmth - wloss));

        /* damage */
        if (warmth <= 0.0f) health -= 8.0f  * dt;
        if (hunger <= 0.0f) health -= 5.0f  * dt;
        if (thirst <= 0.0f) health -= 15.0f * dt;
        if (stage > 1.5f)   health -= 0.5f  * dt;  /* elder passive decay */

        ag[0] = hunger;
        ag[1] = thirst;
        ag[2] = warmth;
        ag[3] = mood;
        ag[4] = health;

        if (health <= 0.0f) dead_mask |= (1ULL << i);
    }
    return dead_mask;
}
