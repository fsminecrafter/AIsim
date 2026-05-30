@echo off
REM ============================================================
REM  Survival Simulation — 3D GPU Edition
REM  Run script for Windows
REM ============================================================

title Survival Simulation Launcher
color 0A

echo.
echo  ================================================
echo   SURVIVAL SIMULATION — 3D GPU Edition
echo  ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo  [1/4] Checking Python version...
python --version

REM Upgrade pip silently
echo  [2/4] Updating pip...
python -m pip install --upgrade pip --quiet

REM Install / verify dependencies
echo  [3/4] Installing dependencies (may take a moment)...
pip install ^
    pygame==2.5.2 ^
    moderngl ^
    numpy ^
    numba ^
    scipy ^
    noise ^
    --quiet

if errorlevel 1 (
    echo.
    echo  [WARNING] Some packages failed to install. Trying without noise...
    pip install pygame==2.5.2 moderngl numpy numba scipy --quiet
)

echo  [4/4] Compiling C kernels (optional, uses Python fallback if not available)...
where gcc >nul 2>&1
if not errorlevel 1 (
    gcc -O3 -march=native -ffast-math -shared -fPIC ^
        -o sim_kernels.so sim_kernels.c ^
        2>nul
    if not errorlevel 1 (
        echo   [OK] C kernels compiled successfully.
    ) else (
        echo   [INFO] C compilation skipped — Python fallback active.
    )
) else (
    echo   [INFO] GCC not found — using Python fallback kernels.
)

echo.
echo  ================================================
echo   Launching simulation...
echo  ================================================
echo.
echo  Controls:
echo    WASD / Arrow Keys  — Pan camera
echo    Q / E              — Orbit left / right
echo    Z / X              — Tilt camera
echo    Middle Mouse Drag  — Orbit camera
echo    Scroll Wheel       — Zoom in / out
echo    Left Click         — Select agent
echo    F                  — Start fire at cursor
echo    +/-                — Speed up / slow down
echo    Space              — Pause / resume
echo    R                  — Reset world
echo    ESC                — Quit
echo.

python main.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Simulation crashed. See above for details.
    pause
)
