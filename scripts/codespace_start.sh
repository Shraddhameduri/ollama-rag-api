#!/bin/bash
# Codespace (re)start: ensure Ollama + API are running. Idempotent.
# Runs on every codespace start via devcontainer.json postStartCommand.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if ! pgrep -f "[o]llama serve" >/dev/null 2>&1; then
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 4
fi

if ! pgrep -f "[u]vicorn app.main:app" >/dev/null 2>&1; then
  nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &
fi

echo "START DONE"
