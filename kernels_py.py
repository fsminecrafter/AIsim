"""
ctypes bindings for sim_kernels.so
Falls back to Python implementations if .so not found.
"""

import ctypes
import os
import numpy as np

_SO_PATH = os.path.join(os.path.dirname(__file__), "sim_kernels.so")
_lib = None

def _load():
    global _lib
    if _lib is not None:
        return _lib
    try:
        _lib = ctypes.CDLL(_SO_PATH)

        # fire_spread_step
        _lib.fire_spread_step.restype  = ctypes.c_int32
        _lib.fire_spread_step.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # fire_intensity
            ctypes.POINTER(ctypes.c_float),   # fire_fuel
            ctypes.POINTER(ctypes.c_int32),   # terrain
            ctypes.POINTER(ctypes.c_int32),   # obj_type
            ctypes.POINTER(ctypes.c_float),   # moisture
            ctypes.POINTER(ctypes.c_uint32),  # rng_seeds
            ctypes.c_float,                   # weather_fire_mult
            ctypes.c_float,                   # dt
        ]

        # bfs_find_nearest
        _lib.bfs_find_nearest.restype  = None
        _lib.bfs_find_nearest.argtypes = [
            ctypes.POINTER(ctypes.c_int32),   # obj_type
            ctypes.POINTER(ctypes.c_float),   # obj_growth
            ctypes.POINTER(ctypes.c_int32),   # terrain
            ctypes.c_int,                     # start_y
            ctypes.c_int,                     # start_x
            ctypes.c_int,                     # target_type
            ctypes.c_int,                     # max_radius
            ctypes.POINTER(ctypes.c_int),     # out_y
            ctypes.POINTER(ctypes.c_int),     # out_x
        ]

        # agent_batch_vitals
        _lib.agent_batch_vitals.restype  = ctypes.c_uint64
        _lib.agent_batch_vitals.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # agents_data
            ctypes.c_int,                     # n_agents
            ctypes.c_float,                   # dt
            ctypes.c_float,                   # w_hunger
            ctypes.c_float,                   # w_thirst
            ctypes.c_float,                   # w_warmth
            ctypes.c_float,                   # w_mood
            ctypes.c_float,                   # w_cold
        ]

        return _lib
    except OSError as e:
        print(f"[sim_kernels] Failed to load C lib: {e}; using Python fallback")
        return None


def c_fire_spread(fire_intensity, fire_fuel, terrain, obj_type,
                  moisture, rng_seeds, weather_fire_mult, dt):
    """Call C fire spread kernel; returns n_ignitions."""
    lib = _load()
    if lib is None:
        return 0

    fi  = np.ascontiguousarray(fire_intensity, dtype=np.float32)
    ff  = np.ascontiguousarray(fire_fuel,      dtype=np.float32)
    tr  = np.ascontiguousarray(terrain,        dtype=np.int32)
    ot  = np.ascontiguousarray(obj_type,       dtype=np.int32)
    mo  = np.ascontiguousarray(moisture,       dtype=np.float32)
    rng = np.ascontiguousarray(rng_seeds,      dtype=np.uint32)

    n = lib.fire_spread_step(
        fi.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ff.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ot.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        mo.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rng.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_float(weather_fire_mult),
        ctypes.c_float(dt),
    )
    # copy back (C wrote in-place on copies — re-assign slices)
    fire_intensity[:] = fi
    fire_fuel[:] = ff
    rng_seeds[:] = rng
    return n


def c_bfs_find_nearest(obj_type, obj_growth, terrain,
                        start_y, start_x, target_type, max_radius=25):
    lib = _load()
    if lib is None:
        return None

    ot  = np.ascontiguousarray(obj_type,   dtype=np.int32)
    og  = np.ascontiguousarray(obj_growth, dtype=np.float32)
    tr  = np.ascontiguousarray(terrain,    dtype=np.int32)
    oy  = ctypes.c_int(-1)
    ox  = ctypes.c_int(-1)

    lib.bfs_find_nearest(
        ot.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        og.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(start_y),
        ctypes.c_int(start_x),
        ctypes.c_int(target_type),
        ctypes.c_int(max_radius),
        ctypes.byref(oy),
        ctypes.byref(ox),
    )
    if oy.value < 0:
        return None
    return (oy.value, ox.value)
