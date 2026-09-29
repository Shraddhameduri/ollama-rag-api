#!/usr/bin/env bash
# Pull the chat + embedding models configured via env (with sane defaults).
set -euo pipefail

: "${OLLAMA_MODEL:=llama3.2:1b}"
: "${OLLAMA_EMBED_MODEL:=nomic-embed-text}"

echo "Pulling chat model:     $OLLAMA_MODEL"
ollama pull "$OLLAMA_MODEL"
echo "Pulling embedding model: $OLLAMA_EMBED_MODEL"
ollama pull "$OLLAMA_EMBED_MODEL"
echo "Done."
