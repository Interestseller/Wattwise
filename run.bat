@echo off
REM WattWise — start the Flask development server

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] .venv not found. Using system Python.
    echo           Run: python -m venv .venv ^&^& pip install -r requirements.txt
)

REM Launch the app
python run.py
