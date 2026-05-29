@echo off
set PYTHONPATH=%CD%
echo Installing dependencies...
pip install pygame pyopengl numpy
echo Starting Agent AI Simulation in 3D...
python main.py
pause