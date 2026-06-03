#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
export PYTHONUNBUFFERED=1

mkdir -p logs

exec >> logs/full_launchd.log 2>> logs/full_launchd_error.log

exec /usr/bin/caffeinate -i .venv/bin/python src/__init__.py --type full
