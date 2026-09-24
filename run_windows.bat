@echo off
REM One-click start for Windows: double-click this file.
REM First run creates .venv and installs packages (a few minutes). Later runs start straight away.
cd /d "%~dp0"
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY (python --version >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python was not found.
  echo Install Python 3.12 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" on the first installer screen. Then run this file again.
  pause
  exit /b 1
)
%PY% verify_install.py || (pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment with %PY% ...
  %PY% -m venv .venv || (pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package install failed. & pause & exit /b 1)
)
if not exist ".env" copy ".env.example" ".env" >nul
echo.
echo Starting AI Investigation Cockpit at http://localhost:8501   (Ctrl+C to stop)
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
