# Sample document 2: quickstart (used by scripts/eval_rag.py)

# Quickstart

## Prerequisites

- Docker and Docker Compose
- An Ollama server (or let Compose run one for you)

## Run with Docker (recommended)

```bash
cp .env.example .env
# edit .env: set API_KEY to a long random string
docker compose up --build
```

This starts two services: `ollama` (the LLM server) and `api` (this project).
On first run, pull the models inside the Ollama container:

```bash
docker compose exec ollama ollama pull llama3.2:1b
docker compose exec ollama ollama pull nomic-embed-text
```

The API is then available at `http://localhost:8000`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama serve  # in another terminal, plus: ollama pull llama3.2:1b
uvicorn app.main:app --reload
```

## Configuration

All configuration is via environment variables (see `.env.example`).
The vector store is ChromaDB with persistent storage; by default data lives in
`./data/chroma` (or `/app/data/chroma` inside Docker). Delete that directory
to reset the knowledge base. Chunking defaults to 800 characters with 120
characters of overlap, and retrieval returns the top 4 chunks per query.
