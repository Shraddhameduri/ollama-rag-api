# ollama-rag-api

A production-grade **Chat API with RAG**, powered entirely by free, locally-run
[Ollama](https://ollama.com) models. OpenAI-compatible chat completions
(including SSE streaming) with cited sources, document ingestion
(PDF/TXT/Markdown), API-key auth, rate limiting, structured logging, Prometheus
metrics, and Docker Compose deployment.

## Features

- **OpenAI-compatible** `POST /v1/chat/completions` — drop-in for OpenAI clients
- **RAG with citations** — retrieved chunks are injected as numbered context;
  every response carries a `sources` array (`[{rank, doc_id, filename,
  chunk_index, score, excerpt}]`); streaming emits sources first
- **Document API** — upload `.pdf`/`.txt`/`.md`, list, delete
- **Per-session memory** — configurable history window, in-memory
- **Security** — `X-API-Key` auth (constant-time compare), per-key/IP rate
  limiting; `/health`, `/ready`, `/metrics` stay open
- **Ops-ready** — JSON structured logs with request IDs, Prometheus metrics,
  liveness/readiness probes, Docker `HEALTHCHECK`, multi-stage slim image,
  non-root user
- **Quality gates** — ruff, pytest (Ollama mocked — no server needed), CI with
  Docker build

## Architecture

```mermaid
flowchart LR
    Client -->|X-API-Key| MW[Middleware\nrequest-id · access log · CORS]
    MW --> Chat[POST /v1/chat/completions]
    MW --> Docs[POST /v1/documents]
    MW --> Ops[/health · /ready · /metrics]

    Chat --> MEM[(Session memory)]
    Chat -->|embed query| EMB[Ollama\nembeddings]
    EMB --> VEC[(ChromaDB\nvector store)]
    VEC -->|top-k chunks| Chat
    Docs -->|chunk + embed| VEC
    Chat -->|prompt + history| LLM[Ollama\nchat model]
    LLM -->|SSE / JSON| Client
```

## Quickstart

### Docker (recommended)

```bash
cp .env.example .env
# set API_KEY to a long random string in .env
docker compose up --build -d

# pull models once (inside the ollama container)
docker compose exec ollama ollama pull llama3.2:1b
docker compose exec ollama ollama pull nomic-embed-text

curl http://localhost:8000/health
```

### Local

```bash
make install          # creates .venv and installs requirements
ollama serve          # terminal 1 (then: ollama pull llama3.2:1b nomic-embed-text)
make run              # terminal 2 -> http://localhost:8000
```

## API reference

All `/v1/*` routes require `X-API-Key: <API_KEY>` when `API_KEY` is set.

### Chat

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"messages":[{"role":"user","content":"What is ollama-rag-api?"}]}' | jq .
```

Request fields: `model?`, `messages[]`, `stream=false`, `temperature=0.2`,
`top_k=4`, `session_id?`, `use_rag=true`. The **last user message** is used as
the retrieval query; the last `HISTORY_WINDOW` turns of `session_id` are
included as history.

Response is OpenAI-compatible (`id`, `object: "chat.completion"`, `choices`,
`usage`) **plus** a top-level `sources` array. When retrieval finds nothing
useful, the model answers alone with `sources: []`.

Streaming:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'
# data: {"sources": [...]}   <- sources first
# data: {"choices":[{"delta":{"content":"Hello"}}]}
# ...
# data: [DONE]
```

### Documents

```bash
# upload
curl -s http://localhost:8000/v1/documents \
  -H "X-API-Key: $API_KEY" -F "file=@notes.pdf" | jq .

# list
curl -s http://localhost:8000/v1/documents -H "X-API-Key: $API_KEY" | jq .

# delete
curl -s -X DELETE http://localhost:8000/v1/documents/<doc_id> \
  -H "X-API-Key: $API_KEY" | jq .
```

### Ops

```bash
curl http://localhost:8000/health     # liveness -> {"status":"ok"}
curl http://localhost:8000/ready      # checks ollama + chroma (200/503)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/v1/models
curl http://localhost:8000/metrics    # prometheus exposition
```

Errors are uniform JSON: `{"error": {"code": "...", "message": "..."}}` —
stack traces are never leaked.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:1b` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_DIR` | `./data/chroma` | ChromaDB persistence dir |
| `API_KEY` | *(empty)* | `X-API-Key` value; **empty = auth disabled** |
| `RATE_LIMIT_PER_MIN` | `60` | Req/min per API key (or per IP if auth off) |
| `CORS_ORIGINS` | *(empty)* | Comma-separated allowed origins |
| `MAX_UPLOAD_MB` | `25` | Max document upload size |
| `CHUNK_SIZE` | `800` | RAG chunk size (chars) |
| `CHUNK_OVERLAP` | `120` | RAG chunk overlap (chars) |
| `TOP_K` | `4` | Retrieved chunks per query |
| `HISTORY_WINDOW` | `6` | Session history turns included |
| `LOG_LEVEL` | `INFO` | Log verbosity |

Copy `.env.example` to `.env` to configure locally.

## Project structure

```
app/
  main.py            # app factory, middleware (request-id, access logging), error handlers
  config.py          # pydantic-settings
  security.py        # X-API-Key auth (constant-time compare)
  limits.py          # slowapi limiter (per-key/IP)
  metrics.py         # Prometheus counters/histograms
  ollama_client.py   # chat/embed wrapper with retries + timeouts
  memory.py          # per-session conversation history
  rag/
    chunking.py      # recursive character splitter (no langchain)
    store.py         # ChromaDB wrapper
    pipeline.py      # ingest_text / retrieve
    prompts.py       # citation-instructed system prompt
  api/
    routes_chat.py   # POST /v1/chat/completions (+ SSE)
    routes_docs.py   # documents CRUD
    routes_ops.py    # /health /ready /v1/models /metrics
tests/               # pytest (Ollama mocked — no server needed)
scripts/             # eval_rag.py smoke eval, pull_models.sh
docker/Dockerfile    # multi-stage, non-root, HEALTHCHECK
```

## Deployment

### Hugging Face Spaces

Spaces runs the Docker image directly:

1. Create a Space with the **Docker** SDK, push this repo.
2. Add secrets: `API_KEY`, and (if using a hosted Ollama) `OLLAMA_HOST`.
   For a self-contained Space, run Ollama inside the same container via a
   startup script, or point `OLLAMA_HOST` at any reachable Ollama server.
3. The Space serves port `7860` by default — set `PORT` handling or map 8000.

> Note: model downloads (`llama3.2:1b` + `nomic-embed-text`) take several
> minutes on first boot; use a persistent volume for `/root/.ollama` and
> `/app/data` to avoid re-pulling.

### Render

1. New **Web Service** → connect repo → Runtime **Docker**,
   Dockerfile path `docker/Dockerfile`.
2. Environment: `API_KEY` (secret), `OLLAMA_HOST` → your Ollama URL
   (Render has no GPU; run Ollama on a VPS or use Ollama Cloud tunnel),
   `CHROMA_DIR=/app/data/chroma`.
3. Add a **Disk** mounted at `/app/data` so the vector store survives deploys.
4. Health check path: `/health`.

### Fly.io

```bash
fly launch --dockerfile docker/Dockerfile
fly secrets set API_KEY="$(openssl rand -hex 32)"
fly volumes create chroma_data --size 2   # mount at /app/data in fly.toml
fly deploy
```

Run Ollama on a separate Fly machine with a volume, or set `OLLAMA_HOST` to an
external Ollama endpoint.

### VPS (any Linux host)

```bash
git clone <repo> && cd ollama-rag-api
cp .env.example .env   # set API_KEY
docker compose up --build -d
docker compose exec ollama bash -c "ollama pull llama3.2:1b && ollama pull nomic-embed-text"
```

Put Caddy/Nginx in front for TLS, and back up the `chroma-data` volume.

### Cloudflare Tunnel (expose a local/VPS instance)

```bash
cloudflared tunnel --url http://localhost:8000
```

For production, create a named tunnel and route a hostname to it; keep
`API_KEY` set — the tunnel exposes the API to the internet.

## Development

```bash
make test    # pytest
make lint    # ruff
make eval    # RAG smoke eval (needs Ollama running with models pulled)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE).

## Troubleshooting

- **`httpx.InvalidURL: Invalid port` on startup**: your `no_proxy`/`NO_PROXY`
  contains bracketed IPv6 entries (e.g. `[::1]`) that httpx 0.28.x cannot
  parse when the `ollama` client initializes. Sanitize it, e.g.
  `export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1`.
- **`/ready` returns 503 `degraded`**: Ollama isn't reachable at `OLLAMA_HOST`.
  Start it (`ollama serve`) and pull the models (`bash scripts/pull_models.sh`).
- **Empty `sources`**: retrieval found nothing above the similarity floor —
  ingest relevant documents first, or check the embedding model is pulled.
