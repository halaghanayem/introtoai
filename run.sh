#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f vendor/FaceNet-main/data_loader.py || ! -f models/selected_backbone.pth ]]; then
  echo 'Team source or checkpoint missing. Follow START_HERE.md and run the Colab builder first.'
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  python3.11 -m venv .venv
fi
if [[ ! -f .venv/attendance-installed.txt ]]; then
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/attendance-installed.txt
fi
.venv/bin/python manage.py check-model
exec .venv/bin/python app.py
