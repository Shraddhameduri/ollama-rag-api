#!/bin/bash
# Codespace first-boot setup: Ollama + models + Python deps + .env.
# Runs once via devcontainer.json postCreateCommand. Idempotent-ish.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "--- installing Ollama ---"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sudo sh
fi

echo "--- starting Ollama and pulling models ---"
pkill -f "[o]llama serve" 2>/dev/null || true
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 6
ollama pull llama3.2:1b
ollama pull nomic-embed-text

echo "--- installing Python deps ---"
python3 -m venv .venv
.venv/bin/pip install --quiet -r requirements.txt

echo "--- writing .env ---"
if [ ! -f .env ]; then
  if [ -z "${RAG_API_KEY:-}" ]; then
    echo "RAG_API_KEY secret not set in codespace; generating ephemeral key" >&2
    RAG_API_KEY="$(openssl rand -hex 32)"
  fi
  sed "s|^API_KEY=.*|API_KEY=$RAG_API_KEY|" .env.example > .env
  chmod 600 .env
fi

echo "SETUP DONE"
