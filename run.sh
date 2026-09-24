#!/usr/bin/env bash
# One-command start for macOS/Linux:  bash run.sh
# First run creates .venv and installs packages (a few minutes). Later runs start straight away.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for cand in python3.14 python3.13 python3.12 python3.11 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11 or newer was not found (macOS's built-in python3 is usually 3.9)."
  echo "Install it with either:"
  echo "  - the installer from https://www.python.org/downloads/macos/   or"
  echo "  - Homebrew:  brew install python@3.12"
  echo "Then open a new Terminal window and run:  bash run.sh"
  exit 1
fi

"$PY" verify_install.py || exit 1

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment with $PY ($("$PY" --version))..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
[ -f .env ] || cp .env.example .env
echo "Starting AI Investigation Cockpit at http://localhost:8501  (Ctrl+C to stop)"
exec .venv/bin/python -m streamlit run app/streamlit_app.py
