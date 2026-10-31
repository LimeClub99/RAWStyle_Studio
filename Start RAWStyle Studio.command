#!/bin/zsh
set -eu
cd "$(dirname "$0")"
printf '\n\033[1mRAWStyle Studio\033[0m\nLocal RAW-to-Lightroom editing.\n\n'
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo 'Python 3.10+ is required. Install Python from https://www.python.org/downloads/macos/'
  echo 'Then double-click this launcher again.'
  read -k 1 '?Press any key to close…'
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3,10):
    raise SystemExit('Python 3.10 or newer is required. Install it from python.org.')
PY
VENV="$HOME/Library/Application Support/RAWStyle Studio/python-env"
if [ ! -f "$VENV/bin/python3" ]; then
  printf 'Setting up this app for the first time (requires internet access)…\n'
  "$PYTHON_BIN" -m venv "$VENV"
fi
if ! "$VENV/bin/python3" -c 'import rawpy,numpy,sklearn,joblib' >/dev/null 2>&1; then
  printf 'Installing RAW decoder and local model dependencies…\n'
  "$VENV/bin/python3" -m pip install --disable-pip-version-check -r requirements.txt
fi
printf 'Opening RAWStyle Studio in your browser.\nClose Terminal to stop RAWStyle.\n\n'
"$VENV/bin/python3" run.py
